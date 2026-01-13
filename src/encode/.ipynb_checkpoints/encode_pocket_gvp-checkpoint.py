#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser


AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA1)}
UNK_IDX = len(AA1)

AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sniff_delim(path: Path) -> str:
    head = path.read_bytes()[:4096]
    return "," if head.count(b",") >= head.count(b"\t") else "\t"


def pick_col(headers: List[str], candidates: List[str]) -> str:
    low = {h.lower(): h for h in headers}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for c in candidates:
        cl = c.lower()
        for hl, orig in low.items():
            if cl in hl:
                return orig
    raise KeyError(f"Cannot find columns {candidates} in header={headers}")


def read_unique_pid_seq(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Return: pid -> (uid, raw_seq)
    pid规则：sha1(raw_seq)[:24]，与你前面 pdb 命名一致
    """
    sep = sniff_delim(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f, delimiter=sep)
        headers = next(r)

    uid_col = pick_col(headers, ["uid"])
    seq_col = pick_col(headers, ["seq", "sequence", "protein"])

    pid2 = {}
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f, delimiter=sep)
        for row in dr:
            uid = (row.get(uid_col, "") or "").strip()
            raw = (row.get(seq_col, "") or "").strip()
            if not raw:
                continue
            pid = sha1_24(raw)
            if pid not in pid2:
                pid2[pid] = (uid, raw)
    return pid2


def parse_pdb_residues(pdb_path: Path, prefer_chain: str = "A"):
    """
    Parse PDB, return residue list with backbone atoms.
    Each residue returns dict:
      - aa (1-letter)
      - coords: dict of atom->(3,)
      - plddt: float (use CA B-factor)
      - res_idx: 0-based index in chain order
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))

    model = next(struct.get_models())
    chains = list(model.get_chains())
    if not chains:
        return []

    chain = None
    for c in chains:
        if c.id == prefer_chain:
            chain = c
            break
    if chain is None:
        chain = chains[0]

    out = []
    idx = 0
    for res in chain.get_residues():
        # skip hetero/water
        hetflag = res.id[0]
        if hetflag not in (" ", ""):
            continue

        name3 = (res.get_resname() or "").upper()
        aa = AA3_TO_AA1.get(name3, "X")

        coords = {}
        for atom_name in ("N", "CA", "C", "O", "CB"):
            if atom_name in res:
                coords[atom_name] = res[atom_name].get_coord().astype(np.float32)

        # need at least N,CA,C for frame; CA for graph distance
        if "CA" not in coords or "N" not in coords or "C" not in coords:
            idx += 1
            continue

        # pLDDT from CA B-factor (ESMFold/AF2 style)
        plddt = float(res["CA"].get_bfactor()) if "CA" in res else 0.0

        out.append({
            "aa": aa,
            "coords": coords,
            "plddt": plddt,
            "res_idx": idx,
        })
        idx += 1

    return out


def block_neighbor_count(coords: np.ndarray, radius: float, block: int = 256) -> np.ndarray:
    """
    coords: (N,3)
    return neighbor_count (N,) within radius (excluding self)
    blockwise to control memory.
    """
    N = coords.shape[0]
    r2 = float(radius * radius)
    out = np.zeros((N,), dtype=np.int32)

    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        diff = coords[i0:i1, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        cnt = (d2 < r2).sum(axis=1) - 1
        out[i0:i1] = cnt.astype(np.int32)
    return out


def select_pocket_residues(
    ca: np.ndarray,
    plddt: np.ndarray,
    topk: int,
    neighbor_radius: float,
    min_plddt: float,
) -> np.ndarray:
    """
    Select pocket residues by score = (plddt/100) * (1/(1+neighbor_count))
    Return indices (Lp,)
    """
    nb = block_neighbor_count(ca, radius=neighbor_radius, block=256).astype(np.float32)
    surf = 1.0 / (1.0 + nb)
    conf = np.clip(pldddt := (plddt / 100.0), 0.0, 1.0)

    mask = (plddt >= min_plddt)
    score = conf * surf
    score = np.where(mask, score, -1.0)

    k = min(topk, ca.shape[0])
    idx = np.argpartition(-score, kth=max(0, k - 1))[:k]
    idx = idx[np.argsort(-score[idx])]

    # remove invalid (score < 0)
    idx = idx[score[idx] >= 0]
    return idx.astype(np.int64)


def aa_onehot(aa_list: List[str]) -> np.ndarray:
    N = len(aa_list)
    x = np.zeros((N, len(AA1) + 1), dtype=np.float32)
    for i, a in enumerate(aa_list):
        j = AA_TO_IDX.get(a, UNK_IDX)
        x[i, j] = 1.0
    return x


def safe_unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def build_node_features(residues: List[dict], sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    node_s: onehot(21) + plddt(1) + relpos(1)
    node_v: [CA->N, CA->C, CA->O, CA->CB]  (V=4)  (missing O/CB use zeros)
    res_idx: selected residue indices in full chain (0-based)
    """
    sel_res = [residues[i] for i in sel.tolist()]
    aa_list = [r["aa"] for r in sel_res]
    plddt = np.array([r["plddt"] for r in sel_res], dtype=np.float32)

    # full chain length (use last res_idx + 1)
    full_len = residues[-1]["res_idx"] + 1 if residues else len(sel_res)
    res_idx = np.array([r["res_idx"] for r in sel_res], dtype=np.int64)
    relpos = (res_idx.astype(np.float32) / max(1.0, float(full_len - 1))).reshape(-1, 1)

    onehot = aa_onehot(aa_list)
    node_s = np.concatenate([onehot, (plddt / 100.0).reshape(-1, 1), relpos], axis=1).astype(np.float32)

    V = 4
    node_v = np.zeros((len(sel_res), V, 3), dtype=np.float32)
    for i, r in enumerate(sel_res):
        c = r["coords"]
        ca = c["CA"]
        node_v[i, 0] = (c["N"] - ca)
        node_v[i, 1] = (c["C"] - ca)
        node_v[i, 2] = (c.get("O", ca) - ca) if "O" in c else 0.0
        node_v[i, 3] = (c.get("CB", ca) - ca) if "CB" in c else 0.0

    node_v = safe_unit(node_v)
    return node_s, node_v, res_idx


def rbf_expand(d: np.ndarray, D_count: int = 16, d_min: float = 0.0, d_max: float = 10.0) -> np.ndarray:
    """
    RBF expansion of distances: (E,) -> (E, D_count)
    """
    centers = np.linspace(d_min, d_max, D_count, dtype=np.float32)
    widths = (centers[1] - centers[0]) if D_count > 1 else (d_max - d_min + 1e-6)
    gamma = 1.0 / (widths * widths + 1e-8)
    d = d.astype(np.float32).reshape(-1, 1)
    return np.exp(-gamma * (d - centers.reshape(1, -1)) ** 2).astype(np.float32)


def build_edges(
    ca_sel: np.ndarray,
    res_idx_sel: np.ndarray,
    cutoff: float,
    rbf_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build directed edges for selected residues within cutoff (Å).
    edge_index: (2,E)
    edge_s: (E, rbf_dim + 1)  (rbf(distance) + seq_sep_norm)
    edge_v: (E, 1, 3) unit displacement vector
    """
    Lp = ca_sel.shape[0]
    if Lp <= 1:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, rbf_dim + 1), dtype=np.float32),
            np.zeros((0, 1, 3), dtype=np.float32),
        )

    diff = ca_sel[:, None, :] - ca_sel[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    mask = (d2 > 1e-8) & (d2 <= cutoff * cutoff)

    src, dst = np.where(mask)
    if src.size == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, rbf_dim + 1), dtype=np.float32),
            np.zeros((0, 1, 3), dtype=np.float32),
        )

    d = np.sqrt(d2[src, dst]).astype(np.float32)
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)

    rbf = rbf_expand(d, D_count=rbf_dim, d_min=0.0, d_max=float(cutoff))

    seq_sep = np.abs(res_idx_sel[src] - res_idx_sel[dst]).astype(np.float32)
    seq_sep_norm = (seq_sep / max(1.0, float(seq_sep.max()))).reshape(-1, 1)

    edge_s = np.concatenate([rbf, seq_sep_norm], axis=1).astype(np.float32)

    disp = ca_sel[dst] - ca_sel[src]
    disp_u = safe_unit(disp).reshape(-1, 1, 3).astype(np.float32)
    edge_v = disp_u

    return edge_index, edge_s, edge_v


def save_npz(out_path: Path, **arrays):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    # 用一个不带 .npz 的临时前缀，避免 numpy 自动加后缀导致找不到文件
    tmp_prefix = out_path.with_suffix("")  # 去掉 .npz
    tmp_prefix = Path(str(tmp_prefix) + ".tmp")  # e.g. xxx.tmp

    tmp_file = Path(str(tmp_prefix) + ".npz")    # numpy 实际会写成这个
    np.savez_compressed(tmp_prefix, **arrays)    # 注意：传 prefix，不传 .npz

    os.replace(tmp_file, out_path)


def process_one(
    pid: str,
    uid: str,
    raw_seq: str,
    pdb_path: Path,
    out_path: Path,
    prefer_chain: str,
    topk: int,
    neighbor_radius: float,
    min_plddt: float,
    edge_cutoff: float,
    rbf_dim: int,
) -> Tuple[bool, str]:
    if not pdb_path.exists():
        return False, f"missing_pdb: {pdb_path}"

    try:
        residues = parse_pdb_residues(pdb_path, prefer_chain=prefer_chain)
        if len(residues) < 2:
            return False, "too_few_residues"

        ca = np.stack([r["coords"]["CA"] for r in residues], axis=0).astype(np.float32)
        plddt = np.array([r["plddt"] for r in residues], dtype=np.float32)

        sel = select_pocket_residues(
            ca=ca,
            plddt=plddt,
            topk=topk,
            neighbor_radius=neighbor_radius,
            min_plddt=min_plddt,
        )
        if sel.size < 2:
            # fallback: just take first min(64, N)
            k = min(64, ca.shape[0])
            sel = np.arange(k, dtype=np.int64)

        node_s, node_v, res_idx = build_node_features(residues, sel)
        ca_sel = ca[sel]
        edge_index, edge_s, edge_v = build_edges(
            ca_sel=ca_sel,
            res_idx_sel=res_idx,
            cutoff=edge_cutoff,
            rbf_dim=rbf_dim,
        )

        save_npz(
            out_path,
            node_s=node_s.astype(np.float32),
            node_v=node_v.astype(np.float32),
            edge_index=edge_index.astype(np.int64),
            edge_s=edge_s.astype(np.float32),
            edge_v=edge_v.astype(np.float32),
            res_idx=res_idx.astype(np.int64),
        )
        return True, "ok"
    except Exception as e:
        return False, f"error: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser("Encode pocket graph with GVP-style features (from ESMFold PDB)")
    ap.add_argument("--datasets", nargs="+", required=True, help="e.g., davis kiba drugbank")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs", help="where {dataset}.csv lives")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-fs", help="where {dataset}/pdb and output go")
    ap.add_argument("--prefer-chain", type=str, default="A")
    ap.add_argument("--topk", type=int, default=256, help="pocket residues topK")
    ap.add_argument("--neighbor-radius", type=float, default=12.0, help="surface neighbor radius (Å)")
    ap.add_argument("--min-plddt", type=float, default=50.0, help="min pLDDT to be eligible for pocket")
    ap.add_argument("--edge-cutoff", type=float, default=10.0, help="edge cutoff in Å")
    ap.add_argument("--rbf-dim", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for ds in args.datasets:
        csv_path = data_root / f"{ds}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing csv: {csv_path}")

        pid2 = read_unique_pid_seq(csv_path)

        pdb_dir = out_root / ds / "pdb"
        out_dir = out_root / ds / "pocket_graph"
        ensure_dir(out_dir)

        pids = list(pid2.keys())
        pbar = tqdm(pids, desc=f"PocketGVP[{ds}]", ncols=120)

        ok, fail = 0, 0
        for pid in pbar:
            uid, raw_seq = pid2[pid]
            pdb_path = pdb_dir / f"{pid}.pdb"
            out_path = out_dir / f"{pid}.npz"

            if args.skip_existing and out_path.exists():
                ok += 1
                continue

            success, msg = process_one(
                pid=pid,
                uid=uid,
                raw_seq=raw_seq,
                pdb_path=pdb_path,
                out_path=out_path,
                prefer_chain=args.prefer_chain,
                topk=args.topk,
                neighbor_radius=args.neighbor_radius,
                min_plddt=args.min_plddt,
                edge_cutoff=args.edge_cutoff,
                rbf_dim=args.rbf_dim,
            )
            if success:
                ok += 1
            else:
                fail += 1
                print(f"[PocketGVP] failed pid={pid} uid={uid}: {msg}")

        print(f"[Done] {ds}: ok={ok}, fail={fail}, wrote to {out_dir}")


if __name__ == "__main__":
    main()
