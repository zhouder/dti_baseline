# -*- coding: utf-8 -*-
"""
GraphDTA 5-fold CV training (STRICT split version)

✅ 支持严格划分：
- warm / hot / strict-warm:
  - 外层：KFold / StratifiedKFold（按样本边切）
  - 内层：StratifiedShuffleSplit / 随机切
  - strict-warm：强制保证 val/test 中的 drug 和 protein 都在 train 出现过（违规样本挪回 train）

- cold-protein:
  - train/val/test 的 protein 两两不重叠（val 也严格 cold）

- cold-drug:
  - train/val/test 的 drug 两两不重叠（val 也严格 cold）

- cold-pair:
  - train/val/test 的 pair 两两不重叠（val 也严格 cold）

- cold-both（严格 both-cold，diagonal block）:
  - 每折 test = D_k × P_k 交集边
  - train/val pool = 既不含 D_k 也不含 P_k 的边（避免 test 实体泄漏）
  - val 在 pool 内再做 strict both-cold（必要时退化为 cold-pair 防止空集）
  ⚠️ 注意：严格 both-cold 的 K-fold 很难覆盖全边集，本实现是 diagonal block，
          会丢弃 “mixed 边”（drug 在 heldout drugs 但 protein 不在 / 反过来）

用法示例：
python train_graphdta_strict.py --dataset DAVIS --data-root /path/to/data --output-dir /path/to/out \
  --model GIN --split-mode cold-protein --epochs 1000 --batch-size 64 --lr 5e-4 --patience 100 --device cuda:0
"""

from __future__ import annotations
import argparse
import os
import time
import warnings

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

# --- 🔇 SILENCE WARNINGS (保持清静) ---
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
# ------------------------------------

# Import local modules
from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from src.baseline_data import GraphDTADataset  # 只用 Dataset，不再用 get_kfold_indices


# ------------------------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# 1. Metrics Calculation
# ------------------------------------------------------------------------------
def calculate_metrics(y_true, y_score, threshold: float = 0.5):
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


def find_best_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    best_f1 = -1.0
    best_thr = 0.5
    thresholds = np.linspace(0.0, 1.0, 101)
    for thr in thresholds:
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


# ------------------------------------------------------------------------------
# 2. Training Helper
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)  # logits

        loss = criterion(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(data.num_graphs)
        y_true_all.extend(data.y.detach().cpu().numpy().reshape(-1).tolist())
        y_score_all.extend(torch.sigmoid(output).detach().cpu().numpy().reshape(-1).tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = calculate_metrics(np.array(y_true_all), np.array(y_score_all))
    metrics["loss"] = float(avg_loss)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_all, y_score_all = [], []

    for data in loader:
        data = data.to(device)
        output = model(data)  # logits
        loss = criterion(output.view(-1), data.y.view(-1))

        total_loss += float(loss.item()) * int(data.num_graphs)
        y_true_all.extend(data.y.detach().cpu().numpy().reshape(-1).tolist())
        y_score_all.extend(torch.sigmoid(output).detach().cpu().numpy().reshape(-1).tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    y_true = np.array(y_true_all, dtype=int)
    y_score = np.array(y_score_all, dtype=float)

    metrics = calculate_metrics(y_true, y_score)
    metrics["loss"] = float(avg_loss)
    return metrics, y_true, y_score


# ------------------------------------------------------------------------------
# 3. STRICT SPLITS (outer + inner val)
# ------------------------------------------------------------------------------
def _get_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    drug_col = cols.get("smiles") or cols.get("drug") or cols.get("compound") or cols.get("ligand")
    prot_col = cols.get("protein") or cols.get("sequence") or cols.get("target") or cols.get("target sequence")
    label_col = cols.get("label") or cols.get("y")

    if drug_col is None or prot_col is None or label_col is None:
        raise ValueError(
            f"CSV列名不匹配，需要包含 smiles/protein/label（大小写不敏感）。当前列={list(df.columns)}"
        )
    return drug_col, prot_col, label_col


def _chunk_unique(arr: np.ndarray, n_splits: int, seed: int):
    uniq = np.unique(arr)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [uniq[k::n_splits] for k in range(n_splits)]


def _enforce_strict_warm(train_idx: np.ndarray, other_idx: np.ndarray,
                         drug_key: np.ndarray, prot_key: np.ndarray):
    """确保 other_idx(=val 或 test) 中的 drug/protein 都在 train_idx 出现过；违规样本挪回 train。"""
    train_idx = np.asarray(train_idx, dtype=int)
    other_idx = np.asarray(other_idx, dtype=int)

    tr_d = set(drug_key[train_idx].tolist())
    tr_p = set(prot_key[train_idx].tolist())

    keep_mask = np.array([
        (drug_key[i] in tr_d) and (prot_key[i] in tr_p)
        for i in other_idx
    ], dtype=bool)

    moved = other_idx[~keep_mask]
    kept = other_idx[keep_mask]
    if len(moved) > 0:
        train_idx = np.concatenate([train_idx, moved])
    return train_idx, kept


def _assert_no_overlap(a_idx, b_idx, key_arr, what: str, name_a: str, name_b: str):
    sa = set(key_arr[np.asarray(a_idx, dtype=int)].tolist())
    sb = set(key_arr[np.asarray(b_idx, dtype=int)].tolist())
    inter = sa & sb
    if len(inter) != 0:
        raise AssertionError(f"[STRICT CHECK FAIL] {what} overlap {name_a}∩{name_b} = {len(inter)}")


def make_strict_splits(df: pd.DataFrame, split_mode: str, n_folds: int, seed: int,
                       overall_val: float = 0.10):
    """
    返回 List[(train_idx, val_idx, test_idx)]，都是 df 的行索引。

    overall_val：总体 val 比例（默认 0.10）。外层 test=1/K 时，
    pool 内 val_frac_in_pool = overall_val / (1 - 1/K)
    """
    drug_col, prot_col, label_col = _get_cols(df)
    drug_key = df[drug_col].astype(str).values
    prot_key = df[prot_col].astype(str).values
    y = df[label_col].values
    N = len(df)
    all_idx = np.arange(N)

    is_binary = len(np.unique(y)) == 2
    K = int(n_folds)
    overall_test = 1.0 / K
    val_frac_in_pool = overall_val / (1.0 - overall_test)

    splits = []

    # ----- warm / hot / strict-warm -----
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
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_frac_in_pool, random_state=seed + 1000 + fold
                )
                tr_sub_rel, va_rel = next(sss.split(tr_pool, y[tr_pool]))
                tr_idx = tr_pool[tr_sub_rel]
                va_idx = tr_pool[va_rel]
            else:
                rng = np.random.default_rng(seed + 1000 + fold)
                perm = rng.permutation(tr_pool)
                n_va = max(1, int(round(val_frac_in_pool * len(tr_pool))))
                va_idx = perm[:n_va]
                tr_idx = perm[n_va:]

            # strict-warm：强制 val/test 的实体都在 train 出现过
            if split_mode == "strict-warm":
                tr_idx, va_idx = _enforce_strict_warm(tr_idx, va_idx, drug_key, prot_key)
                tr_idx, te_idx = _enforce_strict_warm(tr_idx, te_idx, drug_key, prot_key)
                tr_idx, va_idx = _enforce_strict_warm(tr_idx, va_idx, drug_key, prot_key)

            splits.append((tr_idx, va_idx, te_idx))
        return splits

    # ----- cold-protein -----
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

    # ----- cold-drug -----
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

    # ----- cold-pair -----
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

    # ----- cold-both (strict both-cold, diagonal block) -----
    if split_mode == "cold-both":
        drug_folds = _chunk_unique(drug_key, K, seed)
        prot_folds = _chunk_unique(prot_key, K, seed + 7)

        for k in range(K):
            td = set(drug_folds[k])
            tp = set(prot_folds[k])

            te_mask = np.array([(d in td) and (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            te_idx = np.where(te_mask)[0]

            # pool: neither in heldout drug nor heldout protein (avoid leakage)
            pool_mask = np.array([(d not in td) and (p not in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            pool_idx = np.where(pool_mask)[0]

            if len(te_idx) == 0 or len(pool_idx) == 0:
                raise RuntimeError(
                    f"cold-both fold{k+1}: test或pool为空（数据太稀疏/折数过多）。"
                    f"建议减少fold或改repeated holdout。"
                )

            # inner val: strict both-cold within pool
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

            # fallback if empty
            if len(va_idx) == 0 or len(tr_idx) == 0:
                pair_key = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)
                uniq_pair = np.unique(pair_key[pool_idx]); rng.shuffle(uniq_pair)
                n_val_g = max(1, int(round(val_frac_in_pool * len(uniq_pair))))
                val_pair = set(uniq_pair[:n_val_g])
                va_mask2 = np.array([pair_key[i] in val_pair for i in pool_idx], dtype=bool)
                va_idx = pool_idx[va_mask2]
                tr_idx = pool_idx[~va_mask2]

            # strict checks: drugs & proteins are disjoint across train/val/test
            _assert_no_overlap(tr_idx, te_idx, drug_key, "drug", "train", "test")
            _assert_no_overlap(tr_idx, te_idx, prot_key, "protein", "train", "test")
            _assert_no_overlap(va_idx, te_idx, drug_key, "drug", "val", "test")
            _assert_no_overlap(va_idx, te_idx, prot_key, "protein", "val", "test")
            _assert_no_overlap(tr_idx, va_idx, drug_key, "drug", "train", "val")
            _assert_no_overlap(tr_idx, va_idx, prot_key, "protein", "train", "val")

            splits.append((tr_idx, va_idx, te_idx))

        return splits

    raise ValueError(f"Unknown split_mode: {split_mode}")


def summarize_fold(df: pd.DataFrame, idx: np.ndarray, name: str):
    idx = np.asarray(idx, dtype=int)
    drug_col, prot_col, label_col = _get_cols(df)
    d = df.iloc[idx][drug_col].astype(str).values
    p = df.iloc[idx][prot_col].astype(str).values
    y = df.iloc[idx][label_col].values
    n = len(idx)
    msg = f"[{name}] n={n} | uniq_drug={len(np.unique(d))} | uniq_prot={len(np.unique(p))}"
    if len(np.unique(y)) == 2:
        msg += f" | pos_ratio={float(np.mean(y)):.4f}"
    print(msg)


# ------------------------------------------------------------------------------
# 4. Main Loop
# ------------------------------------------------------------------------------
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
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup
    csv_path = os.path.join(args.data_root, args.dataset, "all.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)

    # Build dataset cache
    cache_path = os.path.join(args.data_root, "baseline_cache_v2", args.dataset)
    os.makedirs(cache_path, exist_ok=True)
    dataset = GraphDTADataset(root=cache_path, df=df)

    # sanity check on node feature dim
    if dataset[0].x.shape[1] != 78:
        print(f"❌ Critical Error: Data dim is {dataset[0].x.shape[1]}, expected 78.")
        raise SystemExit(1)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[Device] {device}")

    # Strict splits (train/val/test)
    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    splits = make_strict_splits(df, split_mode, n_folds=args.cv_folds, seed=args.seed, overall_val=args.overall_val)

    run_dir = os.path.join(args.output_dir, f"{args.dataset}_{split_mode}_{args.model}")
    os.makedirs(run_dir, exist_ok=True)

    summary_metrics = []

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        print(f"\n========== Fold {fold} / {len(splits)} ==========")
        fold_dir = os.path.join(run_dir, f"fold_{fold-1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split summary
        summarize_fold(df, train_idx, "train")
        summarize_fold(df, val_idx, "val")
        summarize_fold(df, test_idx, "test")

        train_loader = DataLoader(dataset[np.asarray(train_idx, dtype=int)],
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[np.asarray(val_idx, dtype=int)],
                                batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[np.asarray(test_idx, dtype=int)],
                                 batch_size=args.batch_size, shuffle=False)

        # Model
        if args.model == "GIN":
            model = GINConvNet()
        elif args.model == "GAT":
            model = GATNet()
        elif args.model == "GCN":
            model = GCNNet()
        elif args.model == "GAT_GCN":
            model = GAT_GCN()
        else:
            raise ValueError(f"Unknown model: {args.model}")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()

        log_path = os.path.join(fold_dir, "log.csv")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("epoch,split,loss,AUROC,AUPRC,F1,Accuracy,Sensitivity,Specificity,Precision,MCC,time\n")

            best_val_auprc = -1.0
            patience_counter = 0
            best_model_path = os.path.join(fold_dir, "best.pt")

            for epoch in range(1, args.epochs + 1):
                t_start = time.time()

                train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics, val_y, val_score = validate(model, val_loader, criterion, device)

                duration = time.time() - t_start

                def log_line(split, m):
                    return (
                        f"{epoch},{split},{m['loss']:.6f},{m['AUROC']:.6f},{m['AUPRC']:.6f},"
                        f"{m['F1']:.6f},{m['Accuracy']:.6f},{m['Sensitivity']:.6f},{m['Specificity']:.6f},"
                        f"{m['Precision']:.6f},{m['MCC']:.6f},{duration:.2f}"
                    )

                log_file.write(log_line("train", train_metrics) + "\n")
                log_file.write(log_line("val", val_metrics) + "\n")
                log_file.flush()

                print(
                    f"\rEp {epoch:03d} | Val AUPRC: {val_metrics['AUPRC']:.4f} | Val Loss: {val_metrics['loss']:.4f}",
                    end=""
                )

                if val_metrics["AUPRC"] > best_val_auprc:
                    best_val_auprc = float(val_metrics["AUPRC"])
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f" -> Early stopping at epoch {epoch}")
                    break

        # Eval Test (use val-selected threshold)
        print("\nEvaluating on Test Set...")
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        model = model.to(device)

        _, val_y, val_score = validate(model, val_loader, criterion, device)
        best_thr = find_best_threshold(val_y, val_score)

        _, test_y, test_score = validate(model, test_loader, criterion, device)
        final_metrics = calculate_metrics(test_y, test_score, threshold=best_thr)
        final_metrics["threshold"] = float(best_thr)
        final_metrics["fold"] = int(fold)
        summary_metrics.append(final_metrics)

        pd.DataFrame([final_metrics]).to_csv(os.path.join(fold_dir, "result.csv"), index=False)
        print(f"Fold {fold} Done. AUPRC={final_metrics['AUPRC']:.4f} (thr={best_thr:.3f})")

    # Final report
    if len(summary_metrics) > 0:
        summ_df = pd.DataFrame(summary_metrics)
        summ_path = os.path.join(run_dir, "summary.csv")
        summ_df.to_csv(summ_path, index=False)

        print("\n" + "=" * 50)
        print("          FINAL REPORT")
        print("=" * 50)
        print(f"{'Metric':<15} | {'Mean ± Std':<25}")
        print("-" * 50)

        metrics_list = ["AUROC", "AUPRC", "F1", "Accuracy", "Sensitivity", "Specificity", "Precision", "MCC"]
        report_data = []

        for m in metrics_list:
            mean_val = float(summ_df[m].mean())
            std_val = float(summ_df[m].std())
            res_str = f"{mean_val:.4f} ± {std_val:.4f}"
            print(f"{m:<15} | {res_str:<25}")
            report_data.append({"Metric": m, "Mean": mean_val, "Std": std_val, "Display": res_str})

        print("=" * 50)
        final_path = os.path.join(run_dir, "final_report.csv")
        pd.DataFrame(report_data).to_csv(final_path, index=False)
        print(f"\n[INFO] Full report saved to: {final_path}")
    else:
        print("\n[WARN] No results generated (did training loop fail?)")


if __name__ == "__main__":
    main()
