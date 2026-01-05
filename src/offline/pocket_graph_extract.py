# src/offline/pocket_graph_extract.py
# -*- coding: utf-8 -*-
"""
从蛋白 PDB 结构构建残基级图（暂时视为“口袋图”）并缓存到 .npz。
- 输入：
    - CSV：至少包含 `protein` 列（与训练时一致）
    - PDB：默认放在 /root/lanyun-tmp/structures 下，可以通过 --structures_dir 指定
- 输出：
    - {out}/{dataset}/{key}.npz
    - key = sha1_24(protein_string)（与数据加载侧完全一致）
- 每个 .npz 包含字段（建议）：
    - pid:        str，蛋白 ID（这里用 key）
    - node_scalar_feat: [N, F_node]  残基级节点标量特征（目前只含 AA 类型 one-hot）
    - coords:           [N, 3]      残基 Cα 坐标
    - edge_index:       [2, E]     残基级边（i->j）
    - edge_scalar_feat: [E, 1]     边标量特征（CA–CA 距离）
    - res_idx:          [N]        残基在“全长序列”中的索引（当前用 0..N-1 占位）
    - chain_id:         [N]        残基所属链 ID（字符串）
后续你可以在模型里读取这些字段，接 GVP / GraphTransformer 等结构编码器。
"""

from __future__ import annotations
import argparse, glob, hashlib, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from Bio.PDB import PDBParser, is_aa
except Exception:
    PDBParser = is_aa = None

# ------------------- 通用工具 -------------------

def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]

_AMINO = set("ACDEFGHIKLMNPQRSTVWY")

def prot_variants(s: str) -> List[str]:
    """与 datamodule.py 中 _prot_variants 逻辑对齐（简化版）"""
    raw = s or ""
    v = [raw, raw.strip(), "".join(raw.split()), "".join(raw.split()).upper()]
    only_aa = "".join(ch for ch in raw if ch.upper() in _AMINO)
    if only_aa:
        v += [only_aa, only_aa.upper()]
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def index_structures(struct_dir: Path) -> Dict[str, Path]:
    """
    遍历 structures_dir，建立 {stem -> path} 映射：
    - stem 即文件名去掉扩展名，如 'abcd1234' 对应 abcd1234.pdb
    - 支持 .pdb / .ent / .cif
    """
    tbl: Dict[str, Path] = {}
    if not struct_dir.exists():
        return tbl
    exts = (".pdb", ".ent", ".cif", ".PDB", ".CIF")
    for p in struct_dir.rglob("*"):
        if p.is_file() and p.suffix in exts:
            tbl[p.stem] = p
    return tbl

# ------------------- 残基图构建 -------------------

# 3-letter -> 1-letter 映射
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

def aa3_to_index(resname: str) -> int:
    aa1 = AA_3TO1.get(resname.upper(), "X")
    if aa1 in AA_TO_INDEX:
        return AA_TO_INDEX[aa1]
    else:
        return len(AA_ORDER)  # unknown

def parse_pdb_residues(pdb_path: Path):
    """用 Bio.PDB 解析 PDB，返回标准氨基酸残基列表。"""
    if PDBParser is None or is_aa is None:
        raise RuntimeError("Bio.PDB 未安装，请先 `pip install biopython`")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    residues = []
    for model in structure:
        # 默认只用第一个 model
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                resname = residue.get_resname()
                res_id = residue.id  # (hetfield, resseq, icode)
                residues.append((chain_id, res_id, resname, residue))
        break
    return residues

def build_residue_graph(residues, dist_thresh: float = 10.0):
    """
    根据残基列表构建残基级图：
    - 节点：每个含 CA 的残基
    - 节点特征：AA 类型 one-hot
    - 边：CA–CA 距离 <= dist_thresh
    - 边特征：距离值（1 维）
    """
    coords = []
    aa_indices = []
    chain_ids = []
    res_idx = []

    for i, (chain_id, res_id, resname, residue) in enumerate(residues):
        if "CA" not in residue:
            continue
        ca = residue["CA"]
        coords.append(ca.coord)
        aa_indices.append(aa3_to_index(resname))
        chain_ids.append(chain_id)
        res_idx.append(i)  # 暂时用顺序 index，当作“全长残基索引”的占位

    if len(coords) == 0:
        raise ValueError("No residues with CA atoms in PDB")

    coords = np.asarray(coords, dtype=np.float32)
    aa_indices = np.asarray(aa_indices, dtype=np.int64)
    chain_ids = np.asarray(chain_ids)
    res_idx = np.asarray(res_idx, dtype=np.int64)

    n = coords.shape[0]
    num_aa_types = len(AA_ORDER) + 1
    node_scalar_feat = np.zeros((n, num_aa_types), dtype=np.float32)
    for i, idx in enumerate(aa_indices):
        node_scalar_feat[i, min(idx, num_aa_types-1)] = 1.0

    # pairwise 距离
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # [n, n]

    edge_src, edge_dst, edge_feat = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(dists[i, j])
            if d <= dist_thresh:
                edge_src.append(i)
                edge_dst.append(j)
                edge_feat.append([d])

    if len(edge_src) == 0:
        # 至少保证有个伪边，否则有的 GNN 结构可能报错
        if n >= 2:
            edge_src = [0, 1]
            edge_dst = [1, 0]
            edge_feat = [[np.linalg.norm(coords[0]-coords[1])]] * 2
        else:
            edge_src = [0]
            edge_dst = [0]
            edge_feat = [[0.0]]

    edge_index = np.asarray([edge_src, edge_dst], dtype=np.int64)
    edge_scalar_feat = np.asarray(edge_feat, dtype=np.float32)

    return node_scalar_feat, coords, edge_index, edge_scalar_feat, res_idx, chain_ids

# ------------------- 主逻辑 -------------------

def read_proteins_from_csv(csv_path: str) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "protein" not in df.columns:
        raise ValueError(f"{csv_path} 中未找到 'protein' 列")
    return df["protein"].astype(str).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True,
                    help='如 "/data/DAVIS/all.csv"（可带通配符）')
    ap.add_argument("--structures_dir", type=str, default="/root/lanyun-tmp/structures",
                    help="PDB 结构所在根目录（递归扫描）")
    ap.add_argument("--out", required=True,
                    help="输出根目录，如 /root/lanyun-tmp/cache")
    ap.add_argument("--dataset", required=True,
                    help="数据集名（用于输出子目录），如 DAVIS / BindingDB / BioSNAP")
    ap.add_argument("--dist_thresh", type=float, default=10.0,
                    help="CA–CA 距离阈值（Å），默认 10.0")
    args = ap.parse_args()

    struct_dir = Path(args.structures_dir)
    out_root = Path(args.out) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    if PDBParser is None:
        print("[ERR] 需要 biopython：请先 `pip install biopython`", file=sys.stderr)
        return

    # 建立 PDB 索引
    struct_tbl = index_structures(struct_dir)
    if not struct_tbl:
        print(f"[WARN] 没在 {struct_dir} 下找到任何 PDB/ENT/CIF 文件", file=sys.stderr)
    else:
        print(f"[INFO] indexed {len(struct_tbl)} structure files under {struct_dir}")

    # 从 CSV 读 protein 序列/ID
    files = sorted(glob.glob(args.csv_glob))
    if not files:
        print(f"[WARN] No CSV matched: {args.csv_glob}", file=sys.stderr)
        return

    seen_keys = set()
    n_total = n_ok = n_skip = 0

    for csv_path in files:
        prots = read_proteins_from_csv(csv_path)
        for pro in prots:
            n_total += 1
            canonical = str(pro)
            key = sha1_24(canonical)
            if key in seen_keys:
                continue

            # 找到对应的 PDB 文件
            pdb_path = None
            cands = prot_variants(canonical)
            name_cands = []
            for s in cands:
                name_cands.append(s)
                name_cands.append(sha1_24(s))
            # 去重
            uniq_name_cands, _seen = [], set()
            for s in name_cands:
                if s and s not in _seen:
                    _seen.add(s); uniq_name_cands.append(s)

            for stem in uniq_name_cands:
                p = struct_tbl.get(stem)
                if p is not None:
                    pdb_path = p
                    break

            if pdb_path is None:
                # 找不到 PDB，跳过
                if key not in seen_keys:
                    print(f"[WARN] no PDB found for protein={canonical[:20]}..., key={key}", file=sys.stderr)
                seen_keys.add(key)
                n_skip += 1
                continue

            seen_keys.add(key)

            # 输出文件：直接放在 out_root 下，不再建 ab/0f 这种子目录
            out_path = out_root / f"{key}.npz"
            if out_path.exists():
                # 已经处理过（防止重复）
                continue

            try:
                residues = parse_pdb_residues(pdb_path)
                (node_scalar_feat,
                 coords,
                 edge_index,
                 edge_scalar_feat,
                 res_idx,
                 chain_ids) = build_residue_graph(residues, dist_thresh=args.dist_thresh)
            except Exception as e:
                print(f"[ERR] failed to build graph for {pdb_path}: {e}", file=sys.stderr)
                n_skip += 1
                continue

            np.savez_compressed(
                out_path,
                pid=key,
                node_scalar_feat=node_scalar_feat.astype(np.float32),
                coords=coords.astype(np.float32),
                edge_index=edge_index.astype(np.int64),
                edge_scalar_feat=edge_scalar_feat.astype(np.float32),
                res_idx=res_idx.astype(np.int64),
                chain_id=chain_ids.astype("U4"),
            )
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[INFO] saved {n_ok} graphs (skip={n_skip})")

    print(f"[DONE] total_proteins={n_total}, saved={n_ok}, skipped={n_skip}, out={out_root}")

if __name__ == "__main__":
    main()
