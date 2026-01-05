# -*- coding: utf-8 -*-
"""
GraphDTA baseline training with STRICT splits (warm/cold) + BindingDB-safe preprocessing.

Key fixes for BindingDB IndexError:
- Build dataset from a cleaned df (dropna + RDKit-valid SMILES + reset_index)
- Make splits based on the same cleaned df (so split indices always < len(dataset))
- Optional cache versioning to avoid stale processed cache mismatch

Split modes:
- warm: random edge split (KFold/StratifiedKFold)
- strict-warm: warm split + enforce val/test entities (drug & protein) are seen in train
- cold-protein: proteins are disjoint across train/val/test
- cold-drug: drugs are disjoint across train/val/test
- cold-pair: (drug, protein) pairs are disjoint across train/val/test
- cold-both: strict both-cold diagonal block (D_k × P_k) with leakage-free pool; mixed edges are excluded

Example:
python run.py --dataset BindingDB --data-root /root/lanyun-tmp --output-dir /root/out \
  --model GIN --split-mode cold-protein --epochs 200 --batch-size 64 --lr 5e-4 --patience 30 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, StratifiedShuffleSplit
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix
)

# --- Silence warnings ---
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ---- Local imports (keep same as your project layout) ----
from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from src.baseline_data import GraphDTADataset


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Data loading & cleaning (BindingDB-safe)
# ------------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    low2col = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low2col:
            return low2col[c.lower()]
    return None


def load_and_clean_df(csv_path: str) -> pd.DataFrame:
    """
    Return a df that is BOTH:
    1) cleaned (dropna + rdkit-valid smiles + reset_index)
    2) compatible with GraphDTADataset which often expects columns:
       'smiles', 'protein', 'label'  (lowercase)
    """
    df = pd.read_csv(csv_path)

    def _find_col(df_, candidates):
        low2col = {c.lower(): c for c in df_.columns}
        for c in candidates:
            if c.lower() in low2col:
                return low2col[c.lower()]
        return None

    drug_col  = _find_col(df, ["smiles", "drug", "compound", "ligand", "SMILES"])
    prot_col  = _find_col(df, ["protein", "sequence", "target", "target sequence"])
    label_col = _find_col(df, ["label", "y"])

    if drug_col is None or prot_col is None or label_col is None:
        raise ValueError(
            f"CSV列名不匹配，需要包含 smiles/protein/label（大小写不敏感）。当前列={list(df.columns)}"
        )

    # 统一到内部标准列名（先做一份大写/易读版本）
    df = df[[drug_col, prot_col, label_col]].rename(
        columns={drug_col: "SMILES", prot_col: "Target Sequence", label_col: "Label"}
    )

    df = df.dropna().reset_index(drop=True)

    # label 转数值
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"]).reset_index(drop=True)

    # RDKit 过滤无效 SMILES（BindingDB 很常见）
    smiles = df["SMILES"].astype(str).values
    ok = np.zeros(len(df), dtype=bool)
    for i, s in enumerate(smiles):
        try:
            ok[i] = Chem.MolFromSmiles(s) is not None
        except Exception:
            ok[i] = False
    df = df.loc[ok].reset_index(drop=True)

    # ✅ 关键：补齐 GraphDTADataset 常用的小写列名
    df["smiles"]  = df["SMILES"].astype(str)
    df["protein"] = df["Target Sequence"].astype(str)
    df["label"]   = df["Label"].astype(float)

    return df



def make_cache_tag(csv_path: str, df_len: int) -> str:
    """
    Avoid stale processed cache mismatch.
    Uses file size + mtime + cleaned df_len to form a short tag.
    """
    st = os.stat(csv_path)
    raw = f"{os.path.basename(csv_path)}|{st.st_size}|{int(st.st_mtime)}|{df_len}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def calculate_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_score)
    except Exception:
        auprc = 0.0

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Accuracy": float(acc),
        "Sensitivity": float(rec),
        "Specificity": float(spec),
        "Precision": float(prec),
        "MCC": float(mcc),
    }


def find_best_threshold(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.0, 1.0, 101):
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


# ------------------------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)

        loss = criterion(logits.view(-1), data.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item()) * int(data.num_graphs)
        y_true_all.extend(data.y.detach().cpu().numpy().reshape(-1).tolist())
        y_score_all.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = calculate_metrics(y_true_all, y_score_all)
    metrics["loss"] = float(avg_loss)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = criterion(logits.view(-1), data.y.view(-1))

        total_loss += float(loss.item()) * int(data.num_graphs)
        y_true_all.extend(data.y.detach().cpu().numpy().reshape(-1).tolist())
        y_score_all.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = calculate_metrics(y_true_all, y_score_all)
    metrics["loss"] = float(avg_loss)

    y_true = np.asarray(y_true_all, dtype=int)
    y_score = np.asarray(y_score_all, dtype=float)
    return metrics, y_true, y_score


# ------------------------------------------------------------------------------
# STRICT splits
# ------------------------------------------------------------------------------
def _chunk_unique(arr: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    uniq = np.unique(arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [uniq[k::n_splits] for k in range(n_splits)]


def _enforce_strict_warm(train_idx: np.ndarray, other_idx: np.ndarray,
                         drug_key: np.ndarray, prot_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure every (drug,protein) in other_idx exists in train entities.
    Violating samples are moved into train.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    other_idx = np.asarray(other_idx, dtype=int)

    tr_d = set(drug_key[train_idx].tolist())
    tr_p = set(prot_key[train_idx].tolist())

    keep = np.array([(drug_key[i] in tr_d) and (prot_key[i] in tr_p) for i in other_idx], dtype=bool)
    moved = other_idx[~keep]
    kept = other_idx[keep]
    if len(moved) > 0:
        train_idx = np.concatenate([train_idx, moved])
    return train_idx, kept


def _assert_no_overlap(a_idx, b_idx, key_arr, what: str, name_a: str, name_b: str):
    sa = set(key_arr[np.asarray(a_idx, dtype=int)].tolist())
    sb = set(key_arr[np.asarray(b_idx, dtype=int)].tolist())
    inter = sa & sb
    if len(inter) != 0:
        raise AssertionError(f"[STRICT CHECK FAIL] {what} overlap {name_a}∩{name_b} = {len(inter)}")


def make_strict_splits(
    df: pd.DataFrame,
    split_mode: str,
    n_folds: int,
    seed: int,
    overall_val: float = 0.10,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, val_idx, test_idx), indices are into df rows.

    overall_val is overall fraction (e.g. 0.10). With K folds, outer test=1/K,
    so inner val fraction in pool is overall_val / (1 - 1/K).
    """
    drug_key = df["smiles"].astype(str).values
    prot_key = df["protein"].astype(str).values
    y = df["label"].values
    N = len(df)
    all_idx = np.arange(N)

    is_binary = len(np.unique(y)) == 2
    K = int(n_folds)
    overall_test = 1.0 / K
    val_frac_in_pool = overall_val / (1.0 - overall_test)

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if split_mode in ("warm", "hot", "strict-warm"):
        if is_binary:
            outer = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
            outer_iter = outer.split(all_idx, y)
        else:
            outer = KFold(n_splits=K, shuffle=True, random_state=seed)
            outer_iter = outer.split(all_idx)

        for fold, (tr_pool, te_idx) in enumerate(outer_iter, start=1):
            tr_pool = np.asarray(tr_pool, dtype=int)
            te_idx = np.asarray(te_idx, dtype=int)

            # inner val
            if is_binary:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_in_pool, random_state=seed + 1000 + fold)
                tr_sub_rel, va_rel = next(sss.split(tr_pool, y[tr_pool]))
                tr_idx = tr_pool[tr_sub_rel]
                va_idx = tr_pool[va_rel]
            else:
                rng = np.random.default_rng(seed + 1000 + fold)
                perm = rng.permutation(tr_pool)
                n_va = max(1, int(round(val_frac_in_pool * len(tr_pool))))
                va_idx = perm[:n_va]
                tr_idx = perm[n_va:]

            if split_mode == "strict-warm":
                tr_idx, va_idx = _enforce_strict_warm(tr_idx, va_idx, drug_key, prot_key)
                tr_idx, te_idx = _enforce_strict_warm(tr_idx, te_idx, drug_key, prot_key)
                tr_idx, va_idx = _enforce_strict_warm(tr_idx, va_idx, drug_key, prot_key)

            splits.append((tr_idx, va_idx, te_idx))

        return splits

    if split_mode == "cold-protein":
        gkf = GroupKFold(n_splits=K)
        for fold, (tr_pool, te_idx) in enumerate(gkf.split(all_idx, groups=prot_key), start=1):
            tr_pool = np.asarray(tr_pool, dtype=int)
            te_idx = np.asarray(te_idx, dtype=int)

            uniq_p = np.unique(prot_key[tr_pool])
            rng = np.random.default_rng(seed + 2000 + fold)
            rng.shuffle(uniq_p)
            n_val_g = max(1, int(round(val_frac_in_pool * len(uniq_p))))
            val_p = set(uniq_p[:n_val_g])

            va_mask = np.array([prot_key[i] in val_p for i in tr_pool], dtype=bool)
            va_idx = tr_pool[va_mask]
            tr_idx = tr_pool[~va_mask]

            _assert_no_overlap(tr_idx, va_idx, prot_key, "protein", "train", "val")
            _assert_no_overlap(tr_idx, te_idx, prot_key, "protein", "train", "test")
            _assert_no_overlap(va_idx, te_idx, prot_key, "protein", "val", "test")
            splits.append((tr_idx, va_idx, te_idx))
        return splits

    if split_mode == "cold-drug":
        gkf = GroupKFold(n_splits=K)
        for fold, (tr_pool, te_idx) in enumerate(gkf.split(all_idx, groups=drug_key), start=1):
            tr_pool = np.asarray(tr_pool, dtype=int)
            te_idx = np.asarray(te_idx, dtype=int)

            uniq_d = np.unique(drug_key[tr_pool])
            rng = np.random.default_rng(seed + 3000 + fold)
            rng.shuffle(uniq_d)
            n_val_g = max(1, int(round(val_frac_in_pool * len(uniq_d))))
            val_d = set(uniq_d[:n_val_g])

            va_mask = np.array([drug_key[i] in val_d for i in tr_pool], dtype=bool)
            va_idx = tr_pool[va_mask]
            tr_idx = tr_pool[~va_mask]

            _assert_no_overlap(tr_idx, va_idx, drug_key, "drug", "train", "val")
            _assert_no_overlap(tr_idx, te_idx, drug_key, "drug", "train", "test")
            _assert_no_overlap(va_idx, te_idx, drug_key, "drug", "val", "test")
            splits.append((tr_idx, va_idx, te_idx))
        return splits

    if split_mode == "cold-pair":
        pair_key = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)
        gkf = GroupKFold(n_splits=K)
        for fold, (tr_pool, te_idx) in enumerate(gkf.split(all_idx, groups=pair_key), start=1):
            tr_pool = np.asarray(tr_pool, dtype=int)
            te_idx = np.asarray(te_idx, dtype=int)

            uniq_pair = np.unique(pair_key[tr_pool])
            rng = np.random.default_rng(seed + 4000 + fold)
            rng.shuffle(uniq_pair)
            n_val_g = max(1, int(round(val_frac_in_pool * len(uniq_pair))))
            val_pair = set(uniq_pair[:n_val_g])

            va_mask = np.array([pair_key[i] in val_pair for i in tr_pool], dtype=bool)
            va_idx = tr_pool[va_mask]
            tr_idx = tr_pool[~va_mask]

            _assert_no_overlap(tr_idx, va_idx, pair_key, "pair", "train", "val")
            _assert_no_overlap(tr_idx, te_idx, pair_key, "pair", "train", "test")
            _assert_no_overlap(va_idx, te_idx, pair_key, "pair", "val", "test")
            splits.append((tr_idx, va_idx, te_idx))
        return splits

    if split_mode == "cold-both":
        # strict both-cold diagonal block; mixed edges excluded from each fold
        drug_folds = _chunk_unique(drug_key, K, seed)
        prot_folds = _chunk_unique(prot_key, K, seed + 7)

        for k in range(K):
            td = set(drug_folds[k])
            tp = set(prot_folds[k])

            te_mask = np.array([(d in td) and (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            te_idx = np.where(te_mask)[0]

            pool_mask = np.array([(d not in td) and (p not in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            pool_idx = np.where(pool_mask)[0]

            if len(te_idx) == 0 or len(pool_idx) == 0:
                raise RuntimeError(
                    f"cold-both fold{k+1}: test或pool为空（数据过稀/折数太多）。"
                    f"建议减少fold或改repeated holdout。"
                )

            # val within pool: strict both-cold, fallback to cold-pair if empty
            rng = np.random.default_rng(seed + 5000 + k)
            d_pool = np.unique(drug_key[pool_idx]); rng.shuffle(d_pool)
            p_pool = np.unique(prot_key[pool_idx]); rng.shuffle(p_pool)

            nd = max(1, int(round(val_frac_in_pool * len(d_pool))))
            np_ = max(1, int(round(val_frac_in_pool * len(p_pool))))
            val_d = set(d_pool[:nd])
            val_p = set(p_pool[:np_])

            va_mask = np.array([(d in val_d) and (p in val_p)
                                for d, p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)
            tr_mask = np.array([(d not in val_d) and (p not in val_p)
                                for d, p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)

            va_idx = pool_idx[va_mask]
            tr_idx = pool_idx[tr_mask]

            if len(va_idx) == 0 or len(tr_idx) == 0:
                pair_key = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)
                uniq_pair = np.unique(pair_key[pool_idx]); rng.shuffle(uniq_pair)
                n_val_g = max(1, int(round(val_frac_in_pool * len(uniq_pair))))
                val_pair = set(uniq_pair[:n_val_g])
                va_mask2 = np.array([pair_key[i] in val_pair for i in pool_idx], dtype=bool)
                va_idx = pool_idx[va_mask2]
                tr_idx = pool_idx[~va_mask2]

            # strict checks: no entity overlaps across train/val/test
            _assert_no_overlap(tr_idx, te_idx, drug_key, "drug", "train", "test")
            _assert_no_overlap(tr_idx, te_idx, prot_key, "protein", "train", "test")
            _assert_no_overlap(va_idx, te_idx, drug_key, "drug", "val", "test")
            _assert_no_overlap(va_idx, te_idx, prot_key, "protein", "val", "test")
            _assert_no_overlap(tr_idx, va_idx, drug_key, "drug", "train", "val")
            _assert_no_overlap(tr_idx, va_idx, prot_key, "protein", "train", "val")

            splits.append((tr_idx, va_idx, te_idx))

        return splits

    raise ValueError(f"Unknown split_mode: {split_mode}")


def summarize_split(df: pd.DataFrame, idx: np.ndarray, name: str):
    idx = np.asarray(idx, dtype=int)
    d = df.iloc[idx]["SMILES"].astype(str).values
    p = df.iloc[idx]["Target Sequence"].astype(str).values
    y = df.iloc[idx]["Label"].values
    msg = f"[{name}] n={len(idx)} | uniq_drug={len(np.unique(d))} | uniq_prot={len(np.unique(p))}"
    if len(np.unique(df["Label"].values)) == 2:
        msg += f" | pos_ratio={float(np.mean(y)):.4f}"
    print(msg)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def build_model(name: str):
    if name == "GIN":
        return GINConvNet()
    if name == "GAT":
        return GATNet()
    if name == "GCN":
        return GCNNet()
    if name == "GAT_GCN":
        return GAT_GCN()
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="GIN", choices=["GIN", "GAT", "GCN", "GAT_GCN"])
    parser.add_argument("--split-mode", type=str, default="cold-protein",
                        choices=["warm", "hot", "strict-warm", "cold-protein", "cold-drug", "cold-pair", "cold-both"])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--overall-val", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cache-base", type=str, default="baseline_cache_v2",
                        help="cache folder under data-root")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="delete cache dir and reprocess dataset")
    args = parser.parse_args()

    set_seed(args.seed)

    # -------- load + clean df (critical for BindingDB) --------
    csv_path = os.path.join(args.data_root, args.dataset, "all.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = load_and_clean_df(csv_path)
    print(f"[ALL] cleaned df rows = {len(df)} from {csv_path}")

    # -------- cache handling (avoid stale processed mismatch) --------
    cache_tag = make_cache_tag(csv_path, len(df))
    cache_root = os.path.join(args.data_root, args.cache_base, args.dataset, cache_tag)
    os.makedirs(cache_root, exist_ok=True)

    if args.force_reprocess and os.path.exists(cache_root):
        print(f"[CACHE] force reprocess: deleting {cache_root}")
        shutil.rmtree(cache_root, ignore_errors=True)
        os.makedirs(cache_root, exist_ok=True)

    print(f"[CACHE] root = {cache_root}")

    # -------- build dataset from cleaned df --------
    dataset = GraphDTADataset(root=cache_root, df=df)

    # Defensive check
    if len(dataset) != len(df):
        # This *shouldn't* happen with our cleaning + fresh cache tag, but warn loudly.
        print(f"[WARN] len(dataset)={len(dataset)} != len(df)={len(df)}. "
              f"This suggests GraphDTADataset is dropping rows internally. "
              f"Splits will still be based on df; you may need to align mapping in GraphDTADataset.")
        # To avoid crash, we hard-stop (better than silent wrong split).
        raise RuntimeError("Dataset length mismatch. Consider --force-reprocess or inspect GraphDTADataset filtering logic.")

    # Check node feature dim if you expect 78
    try:
        if dataset[0].x.shape[1] != 78:
            raise RuntimeError(f"Data dim is {dataset[0].x.shape[1]}, expected 78.")
    except Exception as e:
        print(f"[WARN] Could not validate node feature dim: {e}")

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

    split_mode = "warm" if args.split_mode == "hot" else args.split_mode

    # -------- strict splits (train/val/test indices over cleaned df) --------
    splits = make_strict_splits(
        df=df,
        split_mode=split_mode,
        n_folds=args.cv_folds,
        seed=args.seed,
        overall_val=args.overall_val,
    )

    run_dir = os.path.join(args.output_dir, f"{args.dataset}_{split_mode}_{args.model}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[OUT] {run_dir}")

    summary_metrics: List[Dict[str, float]] = []
    keys = ["AUROC", "AUPRC", "F1", "Accuracy", "Sensitivity", "Specificity", "Precision", "MCC"]

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n========== Fold {fold} / {len(splits)} ==========")
        fold_dir = os.path.join(run_dir, f"fold_{fold-1}")
        os.makedirs(fold_dir, exist_ok=True)

        summarize_split(df, train_idx, "train")
        summarize_split(df, val_idx, "val")
        summarize_split(df, test_idx, "test")

        # Safety: ensure indices are in range (should always be true now)
        if np.max(train_idx) >= len(dataset) or np.max(val_idx) >= len(dataset) or np.max(test_idx) >= len(dataset):
            raise RuntimeError("Index out of range: split indices exceed dataset length. Check preprocessing & cache.")

        train_loader = DataLoader(dataset[np.asarray(train_idx, dtype=int)],
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[np.asarray(val_idx, dtype=int)],
                                batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[np.asarray(test_idx, dtype=int)],
                                 batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        log_path = os.path.join(fold_dir, "log.csv")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,split,loss,AUROC,AUPRC,F1,Accuracy,Sensitivity,Specificity,Precision,MCC,time\n")

        best_val_auprc = -1.0
        best_model_path = os.path.join(fold_dir, "best.pt")
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_m, va_y, va_s = evaluate(model, val_loader, criterion, device)
            dt = time.time() - t0

            def fmt_line(ep: int, split: str, m: Dict[str, float], sec: float) -> str:
                return (
                    f"{ep},{split},{m['loss']:.6f},{m['AUROC']:.6f},{m['AUPRC']:.6f},{m['F1']:.6f},"
                    f"{m['Accuracy']:.6f},{m['Sensitivity']:.6f},{m['Specificity']:.6f},{m['Precision']:.6f},"
                    f"{m['MCC']:.6f},{sec:.2f}\n"
                )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(fmt_line(epoch, "train", tr_m, dt))
                f.write(fmt_line(epoch, "val", va_m, dt))

            print(f"\rEp {epoch:03d} | Val AUPRC {va_m['AUPRC']:.4f} | Val loss {va_m['loss']:.4f}", end="")

            if va_m["AUPRC"] > best_val_auprc:
                best_val_auprc = float(va_m["AUPRC"])
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f" -> Early stopping at epoch {epoch} (best val AUPRC={best_val_auprc:.4f})")
                break

        # ---------- test using best checkpoint + threshold from val ----------
        print("\nEvaluating on Test Set...")
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        model = model.to(device)

        _, va_y, va_s = evaluate(model, val_loader, criterion, device)
        best_thr = find_best_threshold(va_y, va_s)

        _, te_y, te_s = evaluate(model, test_loader, criterion, device)
        te_m = calculate_metrics(te_y, te_s, threshold=best_thr)
        te_m["threshold"] = float(best_thr)
        te_m["fold"] = int(fold)

        pd.DataFrame([te_m]).to_csv(os.path.join(fold_dir, "result.csv"), index=False)
        summary_metrics.append(te_m)

        print(f"[TEST/fold{fold}] thr={best_thr:.3f} | AUPRC {te_m['AUPRC']:.4f} | AUROC {te_m['AUROC']:.4f}")

    # ---------- summary ----------
    summ_df = pd.DataFrame(summary_metrics)
    summ_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("FINAL REPORT (mean ± std)")
    print("=" * 60)
    for k in keys:
        mean = float(summ_df[k].mean())
        std = float(summ_df[k].std())
        print(f"{k:<12} {mean:.4f} ± {std:.4f}")
    print("=" * 60)

    report = []
    for k in keys:
        report.append({"Metric": k, "Mean": float(summ_df[k].mean()), "Std": float(summ_df[k].std())})
    pd.DataFrame(report).to_csv(os.path.join(run_dir, "final_report.csv"), index=False)
    print(f"[Saved] {os.path.join(run_dir, 'final_report.csv')}")


if __name__ == "__main__":
    main()
