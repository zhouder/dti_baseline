# src/offline/molclr_extract.py
from __future__ import annotations
import argparse, glob, hashlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# RDKit（两种模式都需要）
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def _one_hot(x, choices):
    v = [0.0] * (len(choices) + 1)
    try:
        i = choices.index(x)
    except ValueError:
        i = len(choices)
    v[i] = 1.0
    return v

# ---------- RDKit 原子级特征（42维，Chemprop风格简化） ----------
_ATOMS = ["H","C","N","O","F","P","S","Cl","Br","I","B","Si","Se","Te"]
_DEG   = [0,1,2,3,4,5]
_CHG   = [-2,-1,0,1,2]
_NUMH  = [0,1,2,3,4]
_HYB   = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
          HybridizationType.SP3D, HybridizationType.SP3D2, HybridizationType.S]

def atom_features(a) -> list[float]:
    v = []
    v += _one_hot(a.GetSymbol(), _ATOMS)
    v += _one_hot(a.GetTotalDegree(), _DEG)
    v += _one_hot(a.GetFormalCharge(), _CHG)
    v += _one_hot(a.GetTotalNumHs(), _NUMH)
    v += _one_hot(a.GetHybridization(), _HYB)
    v += [1.0 if a.GetIsAromatic() else 0.0, 1.0 if a.IsInRing() else 0.0]
    # 手性占位（是否手性中心）
    chi = a.GetChiralTag().name
    v += [1.0 if chi in ("CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW") else 0.0, 0.0]
    v += [a.GetMass()/200.0]
    return v  # 总维度 ≈ 42

def rdkit_matrix_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros((1, 42), dtype=np.float32)
    feats = [atom_features(a) for a in mol.GetAtoms()]
    return np.asarray(feats, dtype=np.float32)

# ---------- molclr 模式用的：把分子转成“索引” ----------
# 节点：[:,0]=原子序号(0..118；0=unknown)  [:,1]=手性(0..2)
# 边   ：[:,0]=键类型(0..4；4=unknown)     [:,1]=方向(0..2)
ATOMIC_NUM_UPPER = 119
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir
CHIRAL_ENUM = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
BOND_TYPE_ENUM = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}  # 4 作为 unknown/其他
BOND_DIR_ENUM = {
    BondDir.NONE: 0,
    BondDir.ENDUPRIGHT: 1,
    BondDir.ENDDOWNRIGHT: 2,
}

def smiles_to_indices(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()==0:
        return None
    xs0, xs1 = [], []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        z = z if 1 <= z <= 118 else 0
        xs0.append(z)
        xs1.append(CHIRAL_ENUM.get(a.GetChiralTag(), 0))
    x = np.stack([np.array(xs0, np.int64), np.array(xs1, np.int64)], axis=1)  # [N,2]

    ei_u, ei_v, es0, es1 = [], [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = BOND_TYPE_ENUM.get(b.GetBondType(), 4)
        d = BOND_DIR_ENUM.get(b.GetBondDir(), 0)
        ei_u += [u, v]; ei_v += [v, u]
        es0   += [t, t]; es1   += [d, d]
    if len(ei_u) == 0:
        edge_index = np.zeros((2,0), dtype=np.int64)
        edge_attr  = np.zeros((0,2), dtype=np.int64)
    else:
        edge_index = np.stack([np.array(ei_u, np.int64), np.array(ei_v, np.int64)], axis=0)  # [2,E]
        edge_attr  = np.stack([np.array(es0, np.int64), np.array(es1, np.int64)], axis=1)    # [E,2]
    return x, edge_index, edge_attr

# ---------- 可选：PyG + GIN/GINE 的 MolCLR 模式 ----------
def _maybe_import_pyg():
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GINConv, GINEConv, BatchNorm
        return True, torch, nn, F, Data, GINConv, GINEConv, BatchNorm
    except Exception as e:
        print(f"[WARN] PyG not available: {e}", file=sys.stderr)
        return False, None, None, None, None, None, None, None

# ====== Embedding + GINEConv 版本（对齐 ckpt 的输入维 & 命名）======
def build_molclr_model(torch, nn, F, GINConv, GINEConv, BatchNorm,
                       hidden_dim=300, layers=5, use_edge=True):
    """
    - x_embedding1: 119×H（原子序号）
    - x_embedding2:   3×H（手性）
    - edge_embedding1: 5×H（键类型）
    - edge_embedding2: 3×H（键方向）
    - 每层 mlp: Linear(H,2H)→ReLU→Linear(2H,H)   （权重尺寸 ~ (600,300)/(300,600)）
    - 命名沿用 gnns.{i}.mlp.* 以增加和 ckpt 的匹配概率（strict=False 仍可容忍缺失）
    """
    class Block(nn.Module):
        def __init__(self, hidden, use_edge):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(hidden, hidden*2), nn.ReLU(),
                nn.Linear(hidden*2, hidden),
            )
            self.conv = GINEConv(self.mlp) if use_edge else GINConv(self.mlp)
            self.bn   = BatchNorm(hidden)
        def forward(self, x, edge_index, edge_attr=None):
            if edge_attr is not None and isinstance(self.conv, GINEConv):
                x = self.conv(x, edge_index, edge_attr)
            else:
                x = self.conv(x, edge_index)
            x = self.bn(x)
            return F.relu(x)

    class Net(nn.Module):
        def __init__(self, hidden, layers, use_edge):
            super().__init__()
            # Embeddings（名字与 ckpt 对齐）
            self.x_embedding1     = nn.Embedding(ATOMIC_NUM_UPPER, hidden)
            self.x_embedding2     = nn.Embedding(3, hidden)
            self.edge_embedding1  = nn.Embedding(5, hidden)
            self.edge_embedding2  = nn.Embedding(3, hidden)
            # GNN 堆叠（保持名字 gnns）
            self.gnns = nn.ModuleList([Block(hidden, use_edge) for _ in range(layers)])
        def node_embed(self, x_idx):
            # x_idx: [N,2] (long)
            return self.x_embedding1(x_idx[:,0]) + self.x_embedding2(x_idx[:,1])
        def edge_embed(self, e_idx):
            if e_idx.numel() == 0:
                return e_idx.new_zeros((0, self.x_embedding1.embedding_dim), dtype=torch.float32)
            return self.edge_embedding1(e_idx[:,0]) + self.edge_embedding2(e_idx[:,1])
        def forward(self, data):
            x = self.node_embed(data.x)
            e = self.edge_embed(data.edge_attr) if hasattr(data, "edge_attr") else None
            for blk in self.gnns:
                x = blk(x, data.edge_index, e)
            return x

    return Net(hidden_dim, layers, use_edge)

def save_npz(out_dir: Path, key: str, arr: np.ndarray, fp16: bool):
    out = out_dir / f"{key}.npz"
    if fp16:
        np.savez_compressed(out, drug_atoms=arr.astype(np.float16))
    else:
        np.savez_compressed(out, drug_atoms=arr.astype(np.float32))

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", type=str, required=True, help='如 "data/DAVIS/fold*_*.csv"')
    ap.add_argument("--out", type=str, required=True, help="输出根目录")
    ap.add_argument("--dataset", type=str, default=None, help="若提供则写到 out/<dataset>/")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","fp32","bf16"])
    ap.add_argument("--mode", type=str, default="rdkit", choices=["rdkit","molclr"])
    # molclr 专用
    ap.add_argument("--molclr_repo", type=str, default=None, help="可选：MolCLR 仓库路径（若需导入其代码）")
    ap.add_argument("--ckpt", type=str, default=None, help="可选：预训练 GIN ckpt 路径（.pth/.pt）")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--hidden", type=int, default=300, help="molclr 模式的隐层维度（通常 300）")
    ap.add_argument("--layers", type=int, default=5, help="molclr 模式的层数（通常 5）")
    ap.add_argument("--dropout", type=float, default=0.0)  # 预留
    args = ap.parse_args()

    out = Path(args.out) / args.dataset if args.dataset else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.csv_glob))
    if not files:
        print(f"[WARN] No CSV matched: {args.csv_glob}", file=sys.stderr)

    fp16 = (args.dtype == "fp16")

    # ---------- RDKit-only 路径（保留） ----------
    if args.mode == "rdkit":
        seen = set(); n_ok = n_zero = 0
        for f in files:
            df = pd.read_csv(f)
            if "smiles" not in df.columns:
                print(f"[WARN] {f} has no 'smiles' column; skip.", file=sys.stderr)
                continue
            for smi in df["smiles"].astype(str).unique():
                key = _hash(smi)
                if key in seen: continue
                seen.add(key)
                try:
                    A = rdkit_matrix_from_smiles(smi)
                    if A.size == 0: A = np.zeros((1, 42), dtype=np.float32); n_zero += 1
                    else: n_ok += 1
                except Exception as e:
                    print(f"[ERR] key={key}: {e} → zero atoms", file=sys.stderr)
                    A = np.zeros((1, 42), dtype=np.float32); n_zero += 1
                save_npz(out, key, A, fp16)
        print(f"[DONE] saved: {n_ok} real, {n_zero} zeros, out={out}")
        return

    # ---------- molclr 路径：索引→Embedding→GINEConv ----------
    ok, torch, nn, F, Data, GINConv, GINEConv, BatchNorm = _maybe_import_pyg()
    if not ok:
        print("[FALLBACK] PyG not found; switching to rdkit mode.", file=sys.stderr)
        args.mode = "rdkit"
        return main()  # 递归复用 rdkit 分支

    # 可选：把 MolCLR 仓库加到 sys.path（若你要直接 import 他们的代码）
    if args.molclr_repo and os.path.isdir(args.molclr_repo):
        sys.path.insert(0, args.molclr_repo)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    amp_dtype = dtype_map.get(args.dtype, torch.float16)
    dev = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # 构建模型（输入来自 Embedding，第一层维度=hidden；mlp 300->600->300）
    model = build_molclr_model(torch, nn, F, GINConv, GINEConv, BatchNorm,
                               hidden_dim=args.hidden, layers=args.layers, use_edge=True)
    model.to(dev).eval()

    # 载入 ckpt（宽松加载，剥离常见前缀；名字对得上就能加载，比如 x_embedding1/2, edge_embedding1/2, gnns.*.mlp.*）
    loaded = False
    if args.ckpt and os.path.exists(args.ckpt):
        try:
            sd = torch.load(args.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
            new_sd = {}
            for k, v in (sd.items() if isinstance(sd, dict) else []):
                kk = k
                for p in ("module.","model.","encoder.","net.","gnn."):
                    if kk.startswith(p): kk = kk[len(p):]
                new_sd[kk] = v
            missing, unexpected = model.load_state_dict(new_sd, strict=False)
            print(f"[CKPT] loaded with missing={len(missing)}, unexpected={len(unexpected)}", file=sys.stderr)
            loaded = True
        except Exception as e:
            print(f"[CKPT] failed to load: {e}", file=sys.stderr)
    if not loaded:
        print("[WARN] using randomly initialized GNN (ckpt not loaded).", file=sys.stderr)

    seen = set(); n_ok = n_zero = 0
    for f in files:
        df = pd.read_csv(f)
        if "smiles" not in df.columns:
            print(f"[WARN] {f} has no 'smiles' column; skip.", file=sys.stderr)
            continue
        for smi in df["smiles"].astype(str).unique():
            key = _hash(smi)
            if key in seen: continue
            seen.add(key)
            try:
                idx_pack = smiles_to_indices(smi)
                if idx_pack is None:
                    A = np.zeros((1, args.hidden), dtype=np.float32); n_zero += 1
                else:
                    x_np, ei_np, ea_np = idx_pack
                    # 转成 PyG Data（long 索引）
                    import torch as _t
                    g = Data(
                        x=_t.as_tensor(x_np, dtype=_t.long),
                        edge_index=_t.as_tensor(ei_np, dtype=_t.long),
                        edge_attr=_t.as_tensor(ea_np, dtype=_t.long),
                    ).to(dev)
                    with torch.no_grad():
                        use_amp = (amp_dtype in (torch.float16, torch.bfloat16)) and str(dev).startswith("cuda")
                        if use_amp:
                            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                                X = model(g)  # [N, hidden]
                        else:
                            X = model(g)
                        if amp_dtype in (torch.float16, torch.bfloat16):
                            X = X.to(amp_dtype)
                        A = X.detach().cpu().numpy()
                    n_ok += 1
            except Exception as e:
                print(f"[ERR] key={key}: {e} → zero atoms", file=sys.stderr)
                A = np.zeros((1, args.hidden), dtype=np.float32); n_zero += 1
            save_npz(out, key, A, fp16)

    print(f"[DONE] saved: {n_ok} real, {n_zero} zeros, out={out}")

if __name__ == "__main__":
    main()
