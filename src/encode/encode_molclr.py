#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


# -------------------------
# utils
# -------------------------
def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype == object:
        raise RuntimeError(f"Refuse to save object array: {path}")
    ensure_dir(path.parent)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def resolve_dataset_csv(data_root: Path, name: str) -> Path:
    for ext in [".csv", ".tsv"]:
        for n in [name, name.lower(), name.upper()]:
            p = data_root / f"{n}{ext}"
            if p.exists():
                return p
    d = data_root / name
    if d.exists() and d.is_dir():
        cands = list(d.glob("*.csv")) + list(d.glob("*.tsv"))
        if len(cands) == 1:
            return cands[0]
    raise FileNotFoundError(f"Cannot find dataset csv for {name} under {data_root}")


def guess_delim(path: Path) -> str:
    # 简单可靠：逗号 vs 制表符
    with path.open("rb") as f:
        head = f.read(4096)
    return "," if head.count(b",") >= head.count(b"\t") else "\t"


def pick_header(headers: List[str], names: Sequence[str]) -> str:
    """
    headers: 原始表头（字符串列表）
    names: 候选关键字，如 ["smile","smiles"]
    """
    lower = {h.lower(): h for h in headers}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    # 允许子串匹配（比如 "canonical_smiles"）
    for n in names:
        nl = n.lower()
        for hl, orig in lower.items():
            if nl in hl:
                return orig
    raise KeyError(f"Cannot find column among {list(names)} in headers={headers}")


def load_unique_smiles(csv_path: Path) -> List[Tuple[str, str]]:
    """
    返回去重后的 (did, smiles) 列表（保持首次出现顺序）
    """
    delim = guess_delim(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            headers = next(reader)
        except StopIteration:
            raise RuntimeError(f"Empty file: {csv_path}")

    smi_col = pick_header(headers, ["smile", "smiles"])

    # 用 DictReader 方便按列名取值
    uniq: "OrderedDict[str,str]" = OrderedDict()
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, delimiter=delim)
        for row in dr:
            smi = (row.get(smi_col, "") or "").strip()
            if not smi:
                continue
            did = sha1_24(smi)
            if did not in uniq:
                uniq[did] = smi

    return list(uniq.items())


# -------------------------
# graph featurization
# -------------------------
def smiles_to_graph_2feat(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      x: (N,2)  int64  [atom_type, chirality]  (OGB feature vector first 2 dims)
      edge_index: (2,E) int64
      edge_attr: (E,2) int64 [bond_type, bond_stereo] (OGB bond feature first 2 dims)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = []
    for a in mol.GetAtoms():
        feat = atom_to_feature_vector(a)
        x.append([feat[0], feat[1]])
    x = np.asarray(x, dtype=np.int64)

    edges, eattr = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_to_feature_vector(b)
        ef = [bf[0], bf[1]]
        edges.append((i, j)); eattr.append(ef)
        edges.append((j, i)); eattr.append(ef)

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, 2), dtype=np.int64)
    else:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr  = np.asarray(eattr, dtype=np.int64)
    return x, edge_index, edge_attr


def add_self_loops(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
    loop_index = torch.stack([loop, loop], dim=0)  # (2, N)
    loop_attr = torch.zeros((num_nodes, edge_attr.size(1)), device=device, dtype=edge_attr.dtype) if edge_attr.numel() else \
                torch.zeros((num_nodes, 2), device=device, dtype=torch.long)
    edge_index2 = torch.cat([edge_index, loop_index], dim=1)
    edge_attr2  = torch.cat([edge_attr, loop_attr], dim=0)
    return edge_index2, edge_attr2


# -------------------------
# model: each layer has its own edge embeddings
# -------------------------
class MolCLRGinLayer(nn.Module):
    def __init__(self, emb_dim: int, num_bond_type: int, num_bond_dir: int):
        super().__init__()
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_dir, emb_dim)
        self.nn = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.numel() == 0:
            e = torch.zeros((0, x.size(1)), device=x.device, dtype=x.dtype)
        else:
            e0 = torch.clamp(edge_attr[:, 0].long(), min=0, max=self.edge_embedding1.num_embeddings - 1)
            e1 = torch.clamp(edge_attr[:, 1].long(), min=0, max=self.edge_embedding2.num_embeddings - 1)
            e = self.edge_embedding1(e0) + self.edge_embedding2(e1)

        src, dst = edge_index[0], edge_index[1]
        m = F.relu(x[src] + e)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        out = (1.0 + self.eps) * x + agg
        return self.nn(out)


class MolCLRGINPerLayerEdge(nn.Module):
    def __init__(self, num_layers: int, emb_dim: int,
                 num_atom_type: int, num_chirality: int,
                 num_bond_type: int, num_bond_dir: int,
                 dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality, emb_dim)

        self.gnns = nn.ModuleList([
            MolCLRGinLayer(emb_dim, num_bond_type, num_bond_dir)
            for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x0 = torch.clamp(x[:, 0].long(), min=0, max=self.x_embedding1.num_embeddings - 1)
        x1 = torch.clamp(x[:, 1].long(), min=0, max=self.x_embedding2.num_embeddings - 1)
        h = self.x_embedding1(x0) + self.x_embedding2(x1)

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=h.size(0))

        for i in range(self.num_layers):
            h = self.gnns[i](h, edge_index, edge_attr)
            h = self.batch_norms[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# -------------------------
# checkpoint loading
# -------------------------
def _is_state_dict(d: Any) -> bool:
    return isinstance(d, dict) and d and all(isinstance(k, str) for k in d.keys()) and all(torch.is_tensor(v) for v in d.values())


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if _is_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "encoder", "backbone"]:
            if key in obj and _is_state_dict(obj[key]):
                return obj[key]
        for v in obj.values():
            if _is_state_dict(v):
                return v
            if isinstance(v, dict):
                for key in ["state_dict", "model_state_dict"]:
                    if key in v and _is_state_dict(v[key]):
                        return v[key]
    raise RuntimeError("Cannot locate a state_dict inside the checkpoint.")


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def canonicalize_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip_prefixes(k: str) -> str:
        prefixes = ["module.", "model.", "gnn.", "encoder.", "backbone.", "net."]
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    out = {}
    for k, v in sd.items():
        k2 = strip_prefixes(k)
        k2 = k2.replace("x_embed1.", "x_embedding1.")
        k2 = k2.replace("x_embed2.", "x_embedding2.")
        k2 = k2.replace("edge_embed1.", "edge_embedding1.")
        k2 = k2.replace("edge_embed2.", "edge_embedding2.")
        k2 = k2.replace("convs.", "gnns.")
        k2 = k2.replace("bns.", "batch_norms.")
        out[k2] = v
    return out


def infer_hparams(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int, int, int]:
    if "x_embedding1.weight" not in sd or "x_embedding2.weight" not in sd:
        raise RuntimeError("Checkpoint missing x_embedding1/x_embedding2 weights.")

    w_a1 = sd["x_embedding1.weight"]
    w_a2 = sd["x_embedding2.weight"]
    emb_dim = int(w_a1.shape[1])
    num_atom_type = int(w_a1.shape[0])
    num_chirality = int(w_a2.shape[0])

    # edge embedding vocab sizes: use layer 0 (your checkpoint style)
    e1_cand = sorted([k for k in sd if re.match(r"gnns\.\d+\.edge_embedding1\.weight$", k)])
    e2_cand = sorted([k for k in sd if re.match(r"gnns\.\d+\.edge_embedding2\.weight$", k)])
    if not e1_cand or not e2_cand:
        raise RuntimeError(
            "Cannot find per-layer edge embeddings in checkpoint.\n"
            f"First 60 keys: {list(sd.keys())[:60]}"
        )
    w_e1 = sd[e1_cand[0]]
    w_e2 = sd[e2_cand[0]]
    num_bond_type = int(w_e1.shape[0])
    num_bond_dir  = int(w_e2.shape[0])

    layer_ids = []
    for k in sd.keys():
        m = re.match(r"gnns\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 5

    return num_layers, emb_dim, num_atom_type, num_chirality, num_bond_type, num_bond_dir


def load_molclr(ckpt: Path, device: torch.device) -> MolCLRGINPerLayerEdge:
    obj = safe_torch_load(ckpt)
    sd = extract_state_dict(obj)
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    sd = canonicalize_state_dict(sd)

    num_layers, emb_dim, na, nc, nb, nd = infer_hparams(sd)
    model = MolCLRGINPerLayerEdge(num_layers, emb_dim, na, nc, nb, nd, dropout=0.0)
    model.load_state_dict(sd, strict=False)  # ignore head layers like feat_lin/out_lin
    model.to(device).eval()
    return model


# -------------------------
# main
# -------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--datasets", nargs="+", default=["drugbank", "kiba", "davis"])
    ap.add_argument("--molclr-ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_molclr(Path(args.molclr_ckpt), device)
    if args.fp16 and device.type == "cuda":
        model.half()

    data_root, out_root = Path(args.data_root), Path(args.out_root)

    for ds in args.datasets:
        csv_path = resolve_dataset_csv(data_root, ds)
        pairs = load_unique_smiles(csv_path)   # [(did, smiles), ...]

        out_dir = out_root / ds / "molclr"
        ensure_dir(out_dir)

        for did, smi in tqdm(pairs, total=len(pairs), desc=f"MolCLR[{ds}]"):
            out_path = out_dir / f"{did}.npy"
            if args.skip_existing and out_path.exists():
                continue
            try:
                x, edge_index, edge_attr = smiles_to_graph_2feat(smi)
                # --- robust tensor build: avoid torch.from_numpy ABI issues ---
                if isinstance(x, np.ndarray):
                    xt = torch.tensor(x.tolist(), dtype=torch.long, device=device)
                else:
                    xt = torch.tensor(x, dtype=torch.long, device=device)

                # edge_index: (2, E)
                if isinstance(edge_index, np.ndarray):
                    if edge_index.size == 0:
                        eit = torch.empty((2, 0), dtype=torch.long, device=device)
                    else:
                        eit = torch.tensor(edge_index.tolist(), dtype=torch.long, device=device)
                else:
                    eit = torch.tensor(edge_index, dtype=torch.long, device=device)

                # edge_attr: (E, 2)
                if isinstance(edge_attr, np.ndarray):
                    if edge_attr.size == 0:
                        eat = torch.empty((0, 2), dtype=torch.long, device=device)
                    else:
                        eat = torch.tensor(edge_attr.tolist(), dtype=torch.long, device=device)
                else:
                    eat = torch.tensor(edge_attr, dtype=torch.long, device=device)
                # -------------------------------------------------------------

                node = model(xt, eit, eat)
                arr = node.detach().to("cpu").float().contiguous().numpy()
                arr = np.asarray(arr, dtype=np.float32)

                # 强制保证是 2D
                if arr.ndim != 2:
                    raise RuntimeError(f"MolCLR output ndim={arr.ndim}, shape={arr.shape} (expected 2D)")

                save_npy(out_path, arr)
            except Exception as e:
                print(f"[MolCLR] failed did={did}: {e}")


if __name__ == "__main__":
    main()