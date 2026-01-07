# -*- coding: utf-8 -*-
"""run_baseline.py

DrugBAN baseline runner with **your own all.csv** + **strict warm/cold split** + K-fold CV.

Why this script exists
----------------------
DrugBAN's original `main.py` expects pre-split csv files under `./datasets/<name>/<split>/`.
In many benchmarking setups (like your UGCA-DTI pipeline), you instead have a unified
`all.csv` and you want to evaluate baselines under the SAME split protocol:

  - warm (edge split)
  - cold-protein
  - cold-drug
  - cold-pair
  - cold-both (STRICT both-cold; mixed edges are excluded)

This script implements that protocol and produces per-fold checkpoints + metrics.

Notes
-----
* Expected CSV columns in all.csv (case-insensitive):
    - smiles
    - protein (or sequence / target / target sequence)
    - label (or y)
  They will be mapped to DrugBAN's required columns: SMILES / Protein / Y

* For BindingDB-like data, invalid SMILES are filtered using RDKit.

* Model selection is by **val AUPRC** (matches your previous scripts).

* Test-time threshold is chosen by maximizing val F1 (optional but reported).

Run example
-----------
python run_baseline.py \
  --dataset DAVIS --data-root /root/lanyun-tmp \
  --out /root/lanyun-tmp/drugban-runs \
  --split-mode cold-protein --cv-folds 5 --seed 42 \
  --cfg configs/DrugBAN.yaml \
  --epochs 100 --batch-size 64 --lr 5e-5 --patience 10
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# sklearn metrics
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        accuracy_score,
        recall_score,
        precision_score,
        matthews_corrcoef,
    )

    _HAS_SK = True
except Exception:
    _HAS_SK = False

# RDKit for SMILES validation
try:
    from rdkit import Chem

    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# DrugBAN modules (relative imports)
from configs import get_cfg_defaults
from dataloader import DTIDataset
import types as _types

def _maybe_patch_dataset_truncation(truncate: bool, max_nodes: int):
    if not truncate:
        return
    try:
        import dgl
    except Exception:
        return
    # Patch only once
    if getattr(DTIDataset, "_patched_truncate_large", False):
        return
    orig_getitem = DTIDataset.__getitem__
    def _getitem(self, idx):
        out = orig_getitem(self, idx)
        # Expect (bg_d, v_p, y) from DrugBAN dataloader
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            bg_d = out[0]
            # truncate if graph too large
            try:
                if hasattr(bg_d, "num_nodes") and int(bg_d.num_nodes()) > int(max_nodes):
                    keep = list(range(int(max_nodes)))
                    bg_d = dgl.node_subgraph(bg_d, keep)
                    out = (bg_d, *out[1:]) if isinstance(out, tuple) else [bg_d, *out[1:]]
            except Exception:
                pass
        return out
    DTIDataset.__getitem__ = _getitem
    DTIDataset._patched_truncate_large = True

from models import DrugBAN
from utils import graph_collate_func


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    low2col = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low2col:
            return low2col[c.lower()]
    return None


def load_all_csv(all_csv: Path) -> pd.DataFrame:
    """Load and normalize all.csv to columns: SMILES / Protein / Y."""
    df = pd.read_csv(all_csv)

    c_smiles = _find_col(df, ["smiles", "drug", "compound", "ligand"])
    c_prot = _find_col(df, ["protein", "sequence", "target", "target sequence"])
    c_y = _find_col(df, ["label", "y"])
    if c_smiles is None or c_prot is None or c_y is None:
        raise ValueError(
            f"all.csv 缺列：需要 smiles/protein/label（大小写不敏感）。当前列={list(df.columns)}"
        )

    df = df[[c_smiles, c_prot, c_y]].rename(columns={c_smiles: "SMILES", c_prot: "Protein", c_y: "Y"})
    df = df.dropna().reset_index(drop=True)
    df["SMILES"] = df["SMILES"].astype(str)
    df["Protein"] = df["Protein"].astype(str)
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).reset_index(drop=True)

    # LABEL_CHECK: ensure classification labels are 0/1; otherwise require binarization threshold
    y_vals = df["Y"].to_numpy(dtype=float)
    uniq = np.unique(y_vals)
    is_binary = (len(uniq) <= 2) and set(uniq.tolist()).issubset({0.0, 1.0})
    if not is_binary:
        # keep original; caller may binarize with args.label_thr
        pass

    # Filter invalid smiles (important for BindingDB)
    if _HAS_RDKIT:
        ok = []
        for s in df["SMILES"].values:
            try:
                ok.append(Chem.MolFromSmiles(s) is not None)
            except Exception:
                ok.append(False)
        ok = np.asarray(ok, dtype=bool)
        before = len(df)
        df = df.loc[ok].reset_index(drop=True)
        if before != len(df):
            print(f"[CLEAN] invalid SMILES filtered: {before} -> {len(df)}")
    else:
        print("[WARN] RDKit not available; skip SMILES validity filtering.")

    return df


def compute_metrics(prob: np.ndarray, lab: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    lab = np.asarray(lab, dtype=np.float32).reshape(-1)
    y_true = lab.astype(np.int32)
    y_pred = (prob >= thr).astype(np.int32)

    out: Dict[str, float] = {}
    out["acc"] = float((y_pred == y_true).mean())

    if not _HAS_SK or len(np.unique(y_true)) < 2:
        out.update({"auc": float("nan"), "auprc": float("nan"), "f1": float("nan"), "recall": float("nan"),
                    "precision": float("nan"), "mcc": float("nan")})
        return out

    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        out["f1"] = float("nan")
    try:
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        out["recall"] = float("nan")
    try:
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        out["precision"] = float("nan")
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = float("nan")
    return out


def find_best_threshold(prob: np.ndarray, lab: np.ndarray) -> float:
    if not _HAS_SK:
        return 0.5
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    lab = np.asarray(lab, dtype=np.float32).reshape(-1).astype(np.int32)
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 199):
        try:
            f1 = f1_score(lab, (prob >= t).astype(np.int32), zero_division=0)
        except Exception:
            f1 = -1.0
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


# ------------------------- split utilities -------------------------


def _chunk_unique(arr: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    uniq = np.unique(arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [uniq[k::n_splits] for k in range(n_splits)]


def make_outer_splits(
    mode: str,
    cv_folds: int,
    seed: int,
    drug_key: np.ndarray,
    prot_key: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_pool_idx, test_idx)."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    N = len(labels)
    all_idx = np.arange(N)

    if mode == "cold-protein":
        gkf = GroupKFold(n_splits=cv_folds)
        return [(tr, te) for tr, te in gkf.split(all_idx, groups=prot_key)]
    if mode == "cold-drug":
        gkf = GroupKFold(n_splits=cv_folds)
        return [(tr, te) for tr, te in gkf.split(all_idx, groups=drug_key)]
    if mode == "cold-pair":
        pair = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)
        gkf = GroupKFold(n_splits=cv_folds)
        return [(tr, te) for tr, te in gkf.split(all_idx, groups=pair)]
    if mode == "cold-both":
        # STRICT both-cold: choose heldout drug fold k and heldout protein fold k as test block.
        # Mixed edges (d in heldout_drugs XOR p in heldout_prots) are excluded from the pool.
        drug_folds = _chunk_unique(drug_key, cv_folds, seed)
        prot_folds = _chunk_unique(prot_key, cv_folds, seed + 7)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(cv_folds):
            td = set(drug_folds[k])
            tp = set(prot_folds[k])
            test_mask = np.array([(d in td) and (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            pool_mask = np.array([(d not in td) and (p not in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            te = np.where(test_mask)[0]
            tr_pool = np.where(pool_mask)[0]
            if len(te) == 0 or len(tr_pool) == 0:
                raise RuntimeError(
                    f"cold-both fold{k+1}: test or train_pool is empty. "
                    f"Try fewer folds or a different split mode."
                )
            out.append((tr_pool, te))
        return out

    # warm/hot (edge split)
    if len(np.unique(labels)) == 2:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(all_idx, labels)]
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(all_idx)]


def sample_val_indices(
    mode: str,
    pool_idx: np.ndarray,
    val_frac_in_pool: float,
    seed: int,
    drug_key: np.ndarray,
    prot_key: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) from pool_idx."""
    from sklearn.model_selection import StratifiedShuffleSplit

    pool_idx = np.asarray(pool_idx, dtype=int)

    if mode in ("warm", "hot"):
        if len(np.unique(labels)) == 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_in_pool, random_state=seed)
            tr_sub, va_sub = next(sss.split(pool_idx, labels[pool_idx]))
            return pool_idx[tr_sub], pool_idx[va_sub]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(pool_idx)
        n_va = max(1, int(round(val_frac_in_pool * len(pool_idx))))
        return perm[n_va:], perm[:n_va]

    if mode == "cold-protein":
        g = prot_key[pool_idx]
        uniq = np.unique(g)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val])
        mask = np.array([x in val_g for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-drug":
        g = drug_key[pool_idx]
        uniq = np.unique(g)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val])
        mask = np.array([x in val_g for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-pair":
        pair = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)[pool_idx]
        uniq = np.unique(pair)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val])
        mask = np.array([x in val_g for x in pair], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-both":
        # STRICT: pick val drugs + val prots within pool, then take only (val_drug AND val_prot) edges for val
        rng = np.random.default_rng(seed)
        d_pool = np.unique(drug_key[pool_idx]); rng.shuffle(d_pool)
        p_pool = np.unique(prot_key[pool_idx]); rng.shuffle(p_pool)
        nd = max(1, int(round(val_frac_in_pool * len(d_pool))))
        np_ = max(1, int(round(val_frac_in_pool * len(p_pool))))
        val_d = set(d_pool[:nd]); val_p = set(p_pool[:np_])

        d_sub = drug_key[pool_idx]
        p_sub = prot_key[pool_idx]
        va_mask = np.array([(d in val_d) and (p in val_p) for d, p in zip(d_sub, p_sub)], dtype=bool)
        tr_mask = np.array([(d not in val_d) and (p not in val_p) for d, p in zip(d_sub, p_sub)], dtype=bool)
        va_idx = pool_idx[va_mask]
        tr_idx = pool_idx[tr_mask]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            # fallback to cold-pair (still leakage-free w.r.t pair)
            return sample_val_indices("cold-pair", pool_idx, val_frac_in_pool, seed, drug_key, prot_key, labels)
        return tr_idx, va_idx

    raise ValueError(f"Unknown mode: {mode}")


def summarize_split(name: str, idx: np.ndarray, drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray):
    idx = np.asarray(idx, dtype=int)
    d = drug_key[idx]
    p = prot_key[idx]
    y = labels[idx]
    msg = (
        f"[{name}] n={len(idx)} | uniq_drug={len(np.unique(d))} | uniq_prot={len(np.unique(p))}"
    )
    if len(np.unique(labels)) == 2:
        msg += f" | pos_ratio={float(np.mean(y)):.4f}"
    print(msg)


@torch.no_grad()
def eval_epoch(model: DrugBAN, loader: DataLoader, device: torch.device, crit: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(False)
    # crit is passed in
    loss_sum = 0.0
    n = 0
    probs, labs = [], []

    it = loader
    if tqdm is not None:
        it = tqdm(loader, total=len(loader), ncols=120, leave=False, desc="train")
    for bg_d, v_p, y in it:
        bg_d = bg_d.to(device)
        v_p = v_p.to(device)
        y = y.float().to(device).view(-1, 1)
        _, _, score, _att = model(bg_d, v_p, mode="eval")
        loss = crit(score, y)
        bs = int(y.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
        probs.append(torch.sigmoid(score).detach().float().cpu().numpy().reshape(-1))
        labs.append(y.detach().float().cpu().numpy().reshape(-1))

    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab = np.concatenate(labs, axis=0) if labs else np.zeros((0,), dtype=np.float32)
    return loss_sum / max(1, n), prob, lab


def train_epoch(model: DrugBAN, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, crit: nn.Module) -> float:
    model.train(True)
    # crit is passed in
    loss_sum = 0.0
    n = 0
    it = loader
    if tqdm is not None:
        it = tqdm(loader, total=len(loader), ncols=120, leave=False, desc="val")
    for bg_d, v_p, y in it:
        bg_d = bg_d.to(device)
        v_p = v_p.to(device)
        y = y.float().to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        _vd, _vp, _f, score = model(bg_d, v_p, mode="train")
        loss = crit(score, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = int(y.shape[0])
        loss_sum += float(loss.item()) * bs
        n += bs
    return loss_sum / max(1, n)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="DAVIS / BindingDB / BioSNAP (folder name under data-root)")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-tmp")
    ap.add_argument("--out", type=str, default="/root/lanyun-tmp/drugban-runs")
    ap.add_argument("--cfg", type=str, default="configs/DrugBAN.yaml", help="DrugBAN yaml config")

    ap.add_argument("--split-mode",
                    choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
                    default="warm")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-drug-nodes", type=int, default=290,
                    help="Drug 分子图固定节点上限（DrugBAN 原实现会 padding 到该长度；BioSNAP 最大原子数可能>290）。")
    ap.add_argument("--auto-max-drug-nodes", action="store_true",
                    help="扫描数据集 SMILES 的最大原子数；若大于 --max-drug-nodes 则自动抬高到该值（更稳，但更吃显存）。")
    ap.add_argument("--truncate-large-mols", action="store_true",
                    help="若分子原子数超过 max_drug_nodes，则截断到前 max_drug_nodes 个节点（更省显存，但损失信息）。")
    ap.add_argument("--label-thr", type=float, default=None,
                    help="若 all.csv 的 label 不是 0/1，提供阈值把它二值化（与 --label-thr-op 配合）。")
    ap.add_argument("--label-thr-op", choices=["ge", "le"], default="ge",
                    help="二值化规则：ge 表示 label>=thr 记为1；le 表示 label<=thr 记为1（亲和力数值越小越强时常用 le）。")
    ap.add_argument("--pos-weight", choices=["auto", "none"], default="auto",
                    help="类别不平衡时 BCE 的 pos_weight；auto=按 train 折自动设为 neg/pos。")
    ap.add_argument("--suppress-warnings", action="store_true",
                    help="屏蔽常见 FutureWarning（不影响训练）。")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=1e-4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    _maybe_patch_dataset_truncation(bool(args.truncate_large_mols), int(args.max_drug_nodes))

    # Safer DataLoader multiprocessing for RDKit/DGL
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    ds_lower = args.dataset.strip().lower()
    ds = {"bindingdb": "BindingDB", "davis": "DAVIS", "biosnap": "BioSNAP"}.get(ds_lower, args.dataset)

    data_root = Path(args.data_root)
    all_csv = data_root / ds / "all.csv"
    if not all_csv.exists():
        raise FileNotFoundError(f"未找到 {all_csv}")

    df_all = load_all_csv(all_csv)
    print(f"[ALL] loaded {len(df_all)} rows from {all_csv}")

    drug_key = df_all["SMILES"].values.astype(object)
    prot_key = df_all["Protein"].values.astype(object)
    labels = df_all["Y"].values.astype(np.float32)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    # override key solver params
    cfg.SOLVER.MAX_EPOCH = int(args.epochs)
    cfg.SOLVER.BATCH_SIZE = int(args.batch_size)
    cfg.SOLVER.NUM_WORKERS = int(args.workers)
    cfg.SOLVER.LR = float(args.lr)
    cfg.SOLVER.SEED = int(args.seed)
    cfg.DA.USE = False  # baseline only (no domain adaptation)

    # outer splits
    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer_splits = make_outer_splits(split_mode, args.cv_folds, args.seed, drug_key, prot_key, labels)

    K = len(outer_splits)
    val_frac_in_pool = 0.10 / (1.0 - 1.0 / K)  # ensure overall 0.7/0.1/0.2 when K=5
    print(f"[SPLIT] target train/val/test = 0.70/0.10/0.20 | K={K} | val_frac_in_pool={val_frac_in_pool:.4f}")

    run_dir = Path(args.out) / f"{ds}_{args.split_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {run_dir}")

    keys = ["auc", "auprc", "f1", "acc", "recall", "precision", "mcc"]
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_pool_idx, test_idx) in enumerate(outer_splits, start=1):
        fold_id = fold - 1
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        fold_dir = run_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_pt = fold_dir / "best.pt"
        last_pt = fold_dir / "last.pt"
        log_csv = fold_dir / "log.csv"
        result_csv = fold_dir / "result.csv"

        print("=" * 80)
        print(f"[Fold {fold}/{K}] train_pool={len(train_pool_idx)} test={len(test_idx)} mode={split_mode}")

        # split train/val within pool
        tr_idx, va_idx = sample_val_indices(
            split_mode,
            train_pool_idx,
            val_frac_in_pool,
            args.seed + 100 + fold,
            drug_key,
            prot_key,
            labels,
        )

        print(f"[COUNT] train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)}")
        summarize_split("train", tr_idx, drug_key, prot_key, labels)
        summarize_split("val", va_idx, drug_key, prot_key, labels)
        summarize_split("test", test_idx, drug_key, prot_key, labels)

        # build dataframes for DTIDataset; reset index so dataset indexing is 0..n-1
        df_tr = df_all.iloc[tr_idx].reset_index(drop=True)
        df_va = df_all.iloc[va_idx].reset_index(drop=True)
        df_te = df_all.iloc[test_idx].reset_index(drop=True)
        train_set = DTIDataset(np.arange(len(df_tr)), df_tr, max_drug_nodes=int(args.max_drug_nodes))
        val_set   = DTIDataset(np.arange(len(df_va)), df_va, max_drug_nodes=int(args.max_drug_nodes))
        test_set  = DTIDataset(np.arange(len(df_te)), df_te, max_drug_nodes=int(args.max_drug_nodes))

        dl_params_train = dict(
            batch_size=int(cfg.SOLVER.BATCH_SIZE),
            num_workers=int(cfg.SOLVER.NUM_WORKERS),
            collate_fn=graph_collate_func,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        dl_params_eval = dict(
            batch_size=int(cfg.SOLVER.BATCH_SIZE),
            num_workers=int(cfg.SOLVER.NUM_WORKERS),
            collate_fn=graph_collate_func,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )

        # Safer multiprocessing for RDKit/DGL graph construction
        if int(cfg.SOLVER.NUM_WORKERS) > 0:
            dl_params_train["multiprocessing_context"] = "spawn"
            dl_params_eval["multiprocessing_context"] = "spawn"

        train_loader = DataLoader(train_set, shuffle=True, **dl_params_train)
        val_loader = DataLoader(val_set, shuffle=False, **dl_params_eval)
        test_loader = DataLoader(test_set, shuffle=False, **dl_params_eval)

        model = DrugBAN(**cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.SOLVER.LR))

        # Loss (handle imbalance if requested)
        if args.pos_weight == "auto":
            y_tr = df_tr["Y"].to_numpy(dtype=float)
            pos = float((y_tr >= 0.5).sum())
            neg = float(len(y_tr) - pos)
            if pos > 0:
                pw = max(1.0, neg / pos)
                crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
                print(f"[LOSS] pos_weight=neg/pos={pw:.3f} (pos={int(pos)} neg={int(neg)})")
            else:
                crit = nn.BCEWithLogitsLoss()
                print("[LOSS] pos_weight skipped (no positive in train split)")
        else:
            crit = nn.BCEWithLogitsLoss()


        # log header
        if not log_csv.exists():
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "lr",
                    "train_loss",
                    "val_loss", "val_auc", "val_auprc", "val_thr",
                    "val_f1", "val_acc", "val_recall", "val_precision", "val_mcc",
                    "is_best_val_auprc",
                    "time_sec",
                ])

        best_val_auprc = -1.0
        bad_cnt = 0
        start_epoch = 1

        if args.resume and last_pt.exists():
            ck = torch.load(last_pt, map_location="cpu")
            model.load_state_dict(ck["model"], strict=False)
            optimizer.load_state_dict(ck["optim"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_val_auprc = float(ck.get("best_val_auprc", -1.0))
            bad_cnt = int(ck.get("bad_cnt", 0))
            print(f"[RESUME] start_epoch={start_epoch} best_val_auprc={best_val_auprc:.6f} bad_cnt={bad_cnt}")

        for ep in range(start_epoch, int(cfg.SOLVER.MAX_EPOCH) + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, device, optimizer, crit)
            va_loss, va_prob, va_lab = eval_epoch(model, val_loader, device, crit)
            thr_now = find_best_threshold(va_prob, va_lab)
            va_met = compute_metrics(va_prob, va_lab, thr=thr_now)

            cur = float(va_met.get("auprc", float("nan")))
            improved = (np.isfinite(cur) and (cur > best_val_auprc + args.es_min_delta))
            if improved:
                prev = best_val_auprc
                best_val_auprc = cur
                bad_cnt = 0
                torch.save({"model": model.state_dict()}, best_pt)
                print(f">>> [BEST] val AUPRC improved: {prev:.6f} -> {best_val_auprc:.6f} | saved best.pt")
            else:
                bad_cnt += 1

            # save last (for resume)
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_val_auprc": best_val_auprc,
                    "bad_cnt": bad_cnt,
                },
                last_pt,
            )

            lr_now = float(optimizer.param_groups[0]["lr"])
            dt = time.time() - t0
            print(
                f"[Epoch {ep:03d}] train_loss={tr_loss:.4f} | val_auprc={va_met['auprc']:.4f} "
                f"val_auc={va_met['auc']:.4f} val_loss={va_loss:.4f} thr={thr_now:.3f} "
                f"bad={bad_cnt}/{args.patience} time={dt:.1f}s"
            )

            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ep, lr_now,
                    float(tr_loss),
                    float(va_loss), float(va_met["auc"]), float(va_met["auprc"]), float(thr_now),
                    float(va_met["f1"]), float(va_met["acc"]), float(va_met["recall"]),
                    float(va_met["precision"]), float(va_met["mcc"]),
                    int(improved),
                    float(dt),
                ])

            if args.patience > 0 and bad_cnt >= args.patience:
                print(f"[EarlyStop] no improve for {bad_cnt} epochs. stop.")
                break

        # test with best
        used = "best.pt" if best_pt.exists() else "last.pt"
        ckpt = best_pt if best_pt.exists() else last_pt
        try:
            ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            ck = torch.load(ckpt, map_location="cpu")
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
        else:
            model.load_state_dict(ck, strict=False)

        # compute best thr from val under best checkpoint
        va_loss, va_prob, va_lab = eval_epoch(model, val_loader, device, crit)
        thr_best = find_best_threshold(va_prob, va_lab)

        te_loss, te_prob, te_lab = eval_epoch(model, test_loader, device, crit)
        te_met = compute_metrics(te_prob, te_lab, thr=thr_best)
        fold_metrics.append(te_met)

        print(
            f"[TEST] used={used} thr={thr_best:.3f} loss={te_loss:.4f} "
            f"auc={te_met['auc']:.4f} auprc={te_met['auprc']:.4f} f1={te_met['f1']:.4f}"
        )

        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fold", fold_id])
            w.writerow(["used_ckpt", used])
            w.writerow(["best_val_auprc", best_val_auprc])
            w.writerow(["best_thr", thr_best])
            w.writerow(["test_loss", float(te_loss)])
            for k in keys:
                w.writerow([k, float(te_met.get(k, float("nan")))])

    # summary
    mean = {k: float(np.nanmean([m.get(k, float("nan")) for m in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([m.get(k, float("nan")) for m in fold_metrics])) for k in keys}
    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for i, m in enumerate(fold_metrics):
            w.writerow([i] + [m.get(k, float("nan")) for k in keys])
        w.writerow(["mean"] + [mean[k] for k in keys])
        w.writerow(["std"] + [std[k] for k in keys])

    print("=" * 80)
    print(f"[Saved] {summary_csv}")
    for k in keys:
        print(f"  {k}: {mean[k]:.4f} ± {std[k]:.4f}")


if __name__ == "__main__":
    main()
