# -*- coding: utf-8 -*-
"""
MolTrans CV runner with strict warm/cold splits.
Reads:  /root/lanyun-fs/{dataset}/{dataset}.csv
Writes: /root/lanyun-tmp/moltrans-runs/{dataset}_{split-mode}/...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    matthews_corrcoef,
)

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder


def set_seed(seed: int = 42):
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


def load_csv_any_schema(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    c_smiles = _find_col(df, ["smile", "smiles", "drug", "compound", "ligand"])
    c_prot = _find_col(df, ["seq", "sequence", "protein", "target", "target sequence"])
    c_y = _find_col(df, ["label", "y"])

    if c_smiles is None or c_prot is None or c_y is None:
        raise ValueError(
            f"CSV must contain columns like (smile/smiles) + (seq/protein) + (label/y). "
            f"Got columns={list(df.columns)}"
        )

    df = df[[c_smiles, c_prot, c_y]].rename(
        columns={c_smiles: "SMILES", c_prot: "Target Sequence", c_y: "Label"}
    )

    df = df.dropna().reset_index(drop=True)
    df["SMILES"] = df["SMILES"].astype(str)
    df["Target Sequence"] = df["Target Sequence"].astype(str)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"]).reset_index(drop=True)

    df["Label"] = df["Label"].astype(np.float32)

    return df


def compute_metrics(prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_bin = (y_true >= 0.5).astype(np.int64)
    pred = (prob >= thr).astype(np.int64)

    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_bin, pred))
    out["sen"] = float(recall_score(y_bin, pred, zero_division=0))
    out["f1"] = float(f1_score(y_bin, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_bin, pred))

    try:
        out["auroc"] = float(roc_auc_score(y_bin, prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_bin, prob))
    except Exception:
        out["auprc"] = float("nan")
    return out


def find_best_threshold(prob: np.ndarray, y_true: np.ndarray, grid=None) -> float:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_bin = (y_true >= 0.5).astype(np.int64)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 199)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_bin, (prob >= t).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return float(best_t)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(False)
    loss_f = nn.BCELoss(reduction="mean")

    loss_sum = 0.0
    n_sum = 0
    probs, labels = [], []

    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for (d, p, d_mask, p_mask, label) in pbar:
        score = model(
            d.long().to(device),
            p.long().to(device),
            d_mask.long().to(device),
            p_mask.long().to(device),
        )
        prob = torch.sigmoid(score).view(-1)
        y = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1)

        n = min(prob.numel(), y.numel())
        if n == 0:
            continue

        loss = loss_f(prob[:n], y[:n])
        bs = int(n)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs

        probs.append(prob[:n].detach().float().cpu().numpy())
        labels.append(y[:n].detach().cpu().numpy())

    pbar.close()
    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    y_all = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.float32)
    return loss_sum / max(1, n_sum), prob_all, y_all


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer, desc: str) -> float:
    model.train(True)
    loss_f = nn.BCELoss(reduction="mean")

    loss_sum = 0.0
    n_sum = 0
    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for (d, p, d_mask, p_mask, label) in pbar:
        optimizer.zero_grad(set_to_none=True)
        score = model(
            d.long().to(device),
            p.long().to(device),
            d_mask.long().to(device),
            p_mask.long().to(device),
        )
        prob = torch.sigmoid(score).view(-1)
        y = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1)

        n = min(prob.numel(), y.numel())
        if n == 0:
            continue

        loss = loss_f(prob[:n], y[:n])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = int(n)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs
        pbar.set_postfix_str(f"loss={float(loss.detach().cpu()):.4f}")

    pbar.close()
    return loss_sum / max(1, n_sum)


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
        drug_folds = _chunk_unique(drug_key, cv_folds, seed)
        prot_folds = _chunk_unique(prot_key, cv_folds, seed + 7)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(cv_folds):
            td = set(drug_folds[k])
            tp = set(prot_folds[k])
            te_mask = np.array([(d in td) and (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            pool_mask = np.array([(d not in td) and (p not in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            te = np.where(te_mask)[0]
            tr_pool = np.where(pool_mask)[0]
            if len(te) == 0 or len(tr_pool) == 0:
                raise RuntimeError("cold-both fold empty; reduce folds or change mode.")
            out.append((tr_pool, te))
        return out

    if mode in ("warm", "hot"):
        if len(np.unique(labels)) == 2:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            return [(tr, te) for tr, te in skf.split(all_idx, labels)]
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in kf.split(all_idx)]

    raise ValueError(f"Unknown split mode: {mode}")


def sample_val_indices(
    mode: str,
    pool_idx: np.ndarray,
    val_frac_in_pool: float,
    seed: int,
    drug_key: np.ndarray,
    prot_key: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
        rng = np.random.default_rng(seed)
        d_pool = np.unique(drug_key[pool_idx])
        p_pool = np.unique(prot_key[pool_idx])
        rng.shuffle(d_pool)
        rng.shuffle(p_pool)
        nd = max(1, int(round(val_frac_in_pool * len(d_pool))))
        np_ = max(1, int(round(val_frac_in_pool * len(p_pool))))
        val_d = set(d_pool[:nd])
        val_p = set(p_pool[:np_])

        d_sub = drug_key[pool_idx]
        p_sub = prot_key[pool_idx]
        va_mask = np.array([(d in val_d) and (p in val_p) for d, p in zip(d_sub, p_sub)], dtype=bool)
        tr_mask = np.array([(d not in val_d) and (p not in val_p) for d, p in zip(d_sub, p_sub)], dtype=bool)
        va_idx = pool_idx[va_mask]
        tr_idx = pool_idx[tr_mask]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            return sample_val_indices("cold-pair", pool_idx, val_frac_in_pool, seed, drug_key, prot_key, labels)
        return tr_idx, va_idx

    raise ValueError(f"Unknown mode: {mode}")


def summarize_split(name: str, idx: np.ndarray, drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray):
    idx = np.asarray(idx, dtype=int)
    d = drug_key[idx]
    p = prot_key[idx]
    y = labels[idx]
    msg = f"[{name}] n={len(idx)} | uniq_drug={len(np.unique(d))} | uniq_prot={len(np.unique(p))}"
    if len(np.unique(labels)) == 2:
        msg += f" | pos_ratio={float(np.mean(y)):.4f}"
    print(msg)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. davis / kiba / drugbank (folder name under data-root)")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-tmp/moltrans-runs")

    ap.add_argument(
        "--split-mode",
        choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
        default="cold-protein",
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--overall-val", type=float, default=0.10)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--resume", action="store_true", help="resume fold from fold_dir/last.pt if exists")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ds = args.dataset.strip()
    if ds.lower().endswith(".csv"):
        ds = ds[:-4]

    csv_path = Path(args.data_root) / ds / f"{ds}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df_all = load_csv_any_schema(csv_path)
    print(f"[CSV] {csv_path}")
    print(f"[ALL] loaded {len(df_all)} rows")

    drug_key = df_all["SMILES"].values.astype(object)
    prot_key = df_all["Target Sequence"].values.astype(object)
    labels = df_all["Label"].values.astype(np.float32)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer = make_outer_splits(split_mode, int(args.cv_folds), int(args.seed), drug_key, prot_key, labels)
    K = len(outer)
    overall_test = 1.0 / K
    val_frac_in_pool = float(args.overall_val) / (1.0 - overall_test)
    print(f"[SPLIT] train/val/test target = {1.0 - overall_test - args.overall_val:.2f}/{args.overall_val:.2f}/{overall_test:.2f} "
          f"| K={K} | val_frac_in_pool={val_frac_in_pool:.4f}")

    out_root = Path(args.out_root)
    run_dir = out_root / f"{ds}_{args.split_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {run_dir}")

    keys = ["auroc", "auprc", "f1", "acc", "sen", "mcc"]
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_pool_idx, test_idx) in enumerate(outer, start=1):
        fold_id = fold - 1
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        tr_idx, va_idx = sample_val_indices(
            split_mode,
            train_pool_idx,
            val_frac_in_pool,
            seed=args.seed + 100 + fold,
            drug_key=drug_key,
            prot_key=prot_key,
            labels=labels,
        )

        print("=" * 80)
        print(f"[Fold {fold}/{K}] train_pool={len(train_pool_idx)} test={len(test_idx)} mode={split_mode}")
        summarize_split("train", tr_idx, drug_key, prot_key, labels)
        summarize_split("val", va_idx, drug_key, prot_key, labels)
        summarize_split("test", test_idx, drug_key, prot_key, labels)

        df_tr = df_all.iloc[tr_idx].copy().reset_index(drop=True)
        df_va = df_all.iloc[va_idx].copy().reset_index(drop=True)
        df_te = df_all.iloc[test_idx].copy().reset_index(drop=True)

        dset_tr = BIN_Data_Encoder(np.arange(len(df_tr)), df_tr["Label"].values, df_tr)
        dset_va = BIN_Data_Encoder(np.arange(len(df_va)), df_va["Label"].values, df_va)
        dset_te = BIN_Data_Encoder(np.arange(len(df_te)), df_te["Label"].values, df_te)

        dl_train = DataLoader(
            dset_tr,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
        )
        dl_eval = dict(
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
        )
        dl_val = DataLoader(dset_va, **dl_eval)
        dl_test = DataLoader(dset_te, **dl_eval)

        config = BIN_config_DBPE()
        config["train_epoch"] = int(args.epochs)

        model = BIN_Interaction_Flat(**config)
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            model = nn.DataParallel(model, dim=0)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

        fold_dir = run_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_pt = fold_dir / "best.pt"
        last_pt = fold_dir / "last.pt"
        log_csv = fold_dir / "log.csv"
        result_csv = fold_dir / "result.csv"

        if not log_csv.exists():
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_f1", "val_acc", "val_sen", "val_mcc", "val_thr", "time_sec", "is_best"])

        start_ep = 1
        best_score = -1.0
        best_thr = 0.5
        best_epoch = 0
        no_improve = 0

        if args.resume and last_pt.exists():
            ck = torch.load(str(last_pt), map_location="cpu")
            state_dict = ck["state_dict"]
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state_dict, strict=False)
            optimizer.load_state_dict(ck["optimizer"])
            start_ep = int(ck.get("epoch", 0)) + 1
            best_score = float(ck.get("best_score", -1.0))
            best_thr = float(ck.get("best_thr", 0.5))
            best_epoch = int(ck.get("best_epoch", 0))
            no_improve = int(ck.get("no_improve", 0))
            print(f"[RESUME] start_ep={start_ep} best_score={best_score:.6f} best_epoch={best_epoch} no_improve={no_improve}")

        for ep in range(start_ep, int(args.epochs) + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(model, dl_train, device, optimizer, desc=f"{ds}/fold{fold} train ep{ep}")
            va_loss, prob_va, y_va = evaluate(model, dl_val, device, desc=f"{ds}/fold{fold} val ep{ep}")

            thr_now = find_best_threshold(prob_va, y_va)
            m = compute_metrics(prob_va, y_va, thr=thr_now)
            dt = time.time() - t0

            score = m["auprc"] if not math.isnan(m["auprc"]) else m["auroc"]
            is_best = bool(score > best_score + float(args.es_min_delta))

            if is_best:
                best_score = float(score)
                best_thr = float(thr_now)
                best_epoch = int(ep)
                no_improve = 0
                torch.save(
                    {
                        "epoch": ep,
                        "state_dict": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_thr": best_thr,
                        "best_epoch": best_epoch,
                        "metrics": m,
                    },
                    str(best_pt),
                )
            else:
                no_improve += 1

            torch.save(
                {
                    "epoch": ep,
                    "state_dict": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_thr": best_thr,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                },
                str(last_pt),
            )

            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        ep,
                        f"{tr_loss:.6f}",
                        f"{va_loss:.6f}",
                        f"{m['auroc']:.6f}",
                        f"{m['auprc']:.6f}",
                        f"{m['f1']:.6f}",
                        f"{m['acc']:.6f}",
                        f"{m['sen']:.6f}",
                        f"{m['mcc']:.6f}",
                        f"{thr_now:.6f}",
                        f"{dt:.2f}",
                        int(is_best),
                    ]
                )

            print(
                f"[{ds}/fold{fold}] ep{ep:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
                f"AUPRC {m['auprc']:.4f} | AUROC {m['auroc']:.4f} | F1 {m['f1']:.4f} | "
                f"ACC {m['acc']:.4f} | SEN {m['sen']:.4f} | MCC {m['mcc']:.4f} | thr {thr_now:.3f} | "
                f"bad={no_improve}/{args.patience}"
            )

            if int(args.patience) > 0 and no_improve >= int(args.patience):
                print(f"[EarlyStop] fold{fold}: no improvement for {args.patience} epochs. best_ep={best_epoch}")
                break

        ckpt_path = best_pt if best_pt.exists() else last_pt
        ck = torch.load(str(ckpt_path), map_location="cpu")
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ck["state_dict"], strict=False)

        te_loss, prob_te, y_te = evaluate(model, dl_test, device, desc=f"{ds}/fold{fold} test")
        te_m = compute_metrics(prob_te, y_te, thr=float(best_thr))
        te_m["thr"] = float(best_thr)
        te_m["fold"] = int(fold_id)

        fold_metrics.append(te_m)

        with open(result_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fold", fold_id])
            w.writerow(["best_epoch", best_epoch])
            w.writerow(["best_thr", best_thr])
            w.writerow(["test_loss", float(te_loss)])
            for k in keys:
                w.writerow([k, float(te_m.get(k, float("nan")))])

        print(
            f"[TEST/fold{fold}] thr={best_thr:.3f} | AUROC {te_m['auroc']:.4f} | AUPRC {te_m['auprc']:.4f} | "
            f"F1 {te_m['f1']:.4f} | ACC {te_m['acc']:.4f} | SEN {te_m['sen']:.4f} | MCC {te_m['mcc']:.4f} | loss {te_loss:.4f}"
        )

    mean = {k: float(np.nanmean([m.get(k, float("nan")) for m in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([m.get(k, float("nan")) for m in fold_metrics])) for k in keys}

    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys + ["thr"])
        for i, m in enumerate(fold_metrics):
            w.writerow([i] + [m.get(k, float("nan")) for k in keys] + [m.get("thr", float("nan"))])
        w.writerow(["mean"] + [mean[k] for k in keys] + [""])
        w.writerow(["std"] + [std[k] for k in keys] + [""])

    print("=" * 80)
    print(f"[Saved] {summary_csv}")
    print("[CV] " + " | ".join([f"{k.upper()} {mean[k]:.4f}±{std[k]:.4f}" for k in keys]))


if __name__ == "__main__":
    main()
