"""
train.py

5-fold CV training script for UGCA-DTI with:
- tqdm progress bars
- per-epoch logging to CSV
- best checkpoint by val AUPRC
- last checkpoint for resume
- early stopping by patience (#epochs without AUPRC improvement)

Output layout:
{out_root}/{DATASET}_{split_mode}/
  ├── summary.csv
  ├── fold1/
  │    ├── log.csv
  │    ├── result.csv
  │    ├── best.pt
  │    └── last.pt
  ├── fold2/
  │    └── ...
  ...
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='The verbose parameter is deprecated.*')

try:
    from .datamodule import DTIDataModule
except ImportError:
    try:
        from src.datamodule import DTIDataModule
    except ImportError:
        from datamodule import DTIDataModule
try:
    from .model import ModelConfig, UGCADTI
except ImportError:
    try:
        from src.model import ModelConfig, UGCADTI
    except ImportError:
        from model import ModelConfig, UGCADTI
try:
    from .splits import SplitMode, add_hash_ids, make_five_fold_splits, sanitize_label_series
except ImportError:
    try:
        from src.splits import SplitMode, add_hash_ids, make_five_fold_splits, sanitize_label_series
    except ImportError:
        from splits import SplitMode, add_hash_ids, make_five_fold_splits, sanitize_label_series


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total > 0 else float("nan")
    sen = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        mcc = float("nan")
    else:
        mcc = (tp * tn - fp * fn) / (denom ** 0.5)

    auroc = safe_auroc(y_true, y_prob)
    auprc = safe_auprc(y_true, y_prob)

    return {
        "AUPRC": auprc,
        "AUROC": auroc,
        "F1": float(f1),
        "ACC": float(acc),
        "SEN": float(sen),
        "MCC": float(mcc),
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
    }


@torch.no_grad()
def run_eval(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    for batch in loader:
        molclr = batch["molclr"].to(device)
        chemberta = batch["chemberta"].to(device)
        esm2 = batch["esm2"].to(device)
        pocket = batch["pocket"].to(device)
        y = batch["label"].to(device)

        logits = model(molclr, chemberta, esm2, pocket)
        loss = criterion(logits, y)
        losses.append(loss.item())

        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        lab = y.detach().cpu().numpy().reshape(-1)
        all_probs.append(prob)
        all_labels.append(lab)

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    metrics = compute_metrics(y_true, y_prob) if len(y_true) > 0 else {k: float("nan") for k in ["AUPRC","AUROC","F1","ACC","SEN","MCC"]}
    return float(np.mean(losses) if losses else float("nan")), metrics


def run_train_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
    desc: str = "",
) -> Tuple[float, Dict[str, float]]:
    model.train()
    losses = []
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        molclr = batch["molclr"].to(device)
        chemberta = batch["chemberta"].to(device)
        esm2 = batch["esm2"].to(device)
        pocket = batch["pocket"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(molclr, chemberta, esm2, pocket)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(loss.item())
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        lab = y.detach().cpu().numpy().reshape(-1)
        all_probs.append(prob)
        all_labels.append(lab)

        pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    metrics = compute_metrics(y_true, y_prob) if len(y_true) > 0 else {k: float("nan") for k in ["AUPRC","AUROC","F1","ACC","SEN","MCC"]}
    return float(np.mean(losses) if losses else float("nan")), metrics


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    return torch.load(str(path), map_location=map_location)


def append_log_row(log_path: Path, row: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def write_result_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["davis","kiba","drugbank","DAVIS","KIBA","DRUGBANK"])
    parser.add_argument("--split", type=str, required=True, choices=["warm","cold-drug","cold-protein","cold-both"])
    parser.add_argument("--out_root", type=str, default="/root/lanyun-tmp/ugca-runs")
    parser.add_argument("--root_dir", type=str, default="/root/lanyun-fs")
    parser.add_argument("--seed", type=int, default=42)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20, help="Early stop after N epochs without val AUPRC improvement.")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--allow_missing", action="store_true", help="Drop pairs with missing feature files instead of raising.")
    parser.add_argument("--resume", action="store_true", help="Resume each fold from fold*/last.pt if it exists.")

    # model (lightweight)
    parser.add_argument("--d_model", type=int, default=256, help="Projection dimension (smaller => lighter).")
    parser.add_argument("--ugca_layers", type=int, default=1, help="UGCA layers (1 recommended for lightweight).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pocket_use_edge", action="store_true", help="Include edge pooled features for pocket vector.")

    args = parser.parse_args()
    set_seed(args.seed)

    dataset = args.dataset.lower()
    split_mode: SplitMode = args.split  # type: ignore

    out_dir = Path(args.out_root) / f"{dataset.upper()}_{split_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataframe once (for generating splits)
    csv_path = Path(args.root_dir) / dataset / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df_all = pd.read_csv(csv_path)
    df_all = add_hash_ids(df_all)
    df_all["label"] = sanitize_label_series(df_all["label"])

    folds = make_five_fold_splits(df_all, mode=split_mode, n_splits=5, seed=args.seed, val_ratio_in_train=1/8)

    fold_results: List[Dict[str, float]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for fold_split in folds:
        fold = fold_split.fold
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[{dataset.upper()} | {split_mode}] Fold {fold}/5  ->  {fold_dir}")
        print("=" * 80)

        # Data
        dm = DTIDataModule(
            dataset=dataset,
            train_idx=fold_split.train_idx,
            val_idx=fold_split.val_idx,
            test_idx=fold_split.test_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            allow_missing=args.allow_missing,
            pocket_use_edge=args.pocket_use_edge,
            root_dir=args.root_dir,
        )
        dm.setup(verbose=True)

        train_loader = dm.get_dataloader("train", shuffle=True)
        val_loader = dm.get_dataloader("val", shuffle=False)
        test_loader = dm.get_dataloader("test", shuffle=False)

        dims = dm.input_dims
        assert dims is not None

        # Model
        cfg = ModelConfig(
            in_molclr=dims["molclr"],
            in_chemberta=dims["chemberta"],
            in_esm2=dims["esm2"],
            in_pocket=dims["pocket"],
            d_model=args.d_model,
            ugca_layers=args.ugca_layers,
            dropout=args.dropout,
        )
        model = UGCADTI(cfg).to(device)

        # Loss / optim
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3), verbose=False
        )

        # Resume
        best_path = fold_dir / "best.pt"
        last_path = fold_dir / "last.pt"
        log_path = fold_dir / "log.csv"
        result_path = fold_dir / "result.csv"

        start_epoch = 1
        best_val_auprc = -1.0
        best_epoch = 0
        bad_epochs = 0

        if args.resume and last_path.exists():
            ckpt = load_checkpoint(last_path, map_location=str(device))
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_auprc = float(ckpt.get("best_val_auprc", -1.0))
            best_epoch = int(ckpt.get("best_epoch", 0))
            bad_epochs = int(ckpt.get("bad_epochs", 0))

            print(f"[Resume] Loaded {last_path}. start_epoch={start_epoch}, best_val_auprc={best_val_auprc:.6f} (epoch {best_epoch})")

        # Train loop
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            lr = float(optimizer.param_groups[0]["lr"])

            train_loss, train_metrics = run_train_epoch(
                model=model,
                loader=train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                grad_clip=args.grad_clip if args.grad_clip > 0 else None,
                desc=f"Fold {fold} | Epoch {epoch}/{args.epochs} | train",
            )
            val_loss, val_metrics = run_eval(model, val_loader, device, criterion)

            val_auprc = val_metrics["AUPRC"]
            scheduler.step(val_auprc if not np.isnan(val_auprc) else -1.0)

            epoch_time = time.time() - t0

            # Check improvement (update best before logging)
            improved = (not np.isnan(val_auprc)) and (val_auprc > best_val_auprc + 1e-6)
            if improved:
                old = best_val_auprc
                best_val_auprc = float(val_auprc)
                best_epoch = epoch
                bad_epochs = 0
                print(f"✅ Val AUPRC improved: {old:.6f} -> {best_val_auprc:.6f}. Saving best.pt")
                save_checkpoint(
                    best_path,
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "cfg": asdict(cfg),
                        "best_val_auprc": best_val_auprc,
                        "args": vars(args),
                    },
                )
            else:
                bad_epochs += 1

            # Console summary (train + val metrics)
            def fmt_metrics(name: str, m: Dict[str, float]) -> str:
                return (
                    f"{name:<5} "
                    f"AUPRC {m['AUPRC']:>7.4f} | AUROC {m['AUROC']:>7.4f} | "
                    f"F1 {m['F1']:>7.4f} | ACC {m['ACC']:>7.4f} | "
                    f"SEN {m['SEN']:>7.4f} | MCC {m['MCC']:>7.4f}"
                )

            header = (
                f"[Fold {fold}][Epoch {epoch:03d}] "
                f"lr={lr:.2e}  time={epoch_time:>6.1f}s  "
                f"loss(train/val)={train_loss:.4f}/{val_loss:.4f}  "
                f"bestAUPRC={best_val_auprc:.4f}  bad={bad_epochs}/{args.patience}"
            )
            print(header)
            print(fmt_metrics("Train", train_metrics))
            print(fmt_metrics("Val", val_metrics))

            # Logging row
            row = {
                "epoch": epoch,
                "lr": lr,
                "time_sec": epoch_time,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items() if k in ["AUPRC","AUROC","F1","ACC","SEN","MCC"]},
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if k in ["AUPRC","AUROC","F1","ACC","SEN","MCC"]},
                "best_val_AUPRC": best_val_auprc,
                "best_epoch": best_epoch,
                "improved": int(improved),
                "bad_epochs": bad_epochs,
            }
            append_log_row(log_path, row)

            # Save last (for resume)
            save_checkpoint(
                last_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_auprc": best_val_auprc,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "cfg": asdict(cfg),
                    "args": vars(args),
                },
            )

            # Early stop check (after saving)
            if bad_epochs >= args.patience:
                print(f"⏹️ Early stopping: no val AUPRC improvement for {args.patience} epochs.")
                break

        # Test with best checkpoint
        if not best_path.exists():
            print("⚠️ best.pt not found (maybe all val AUPRC were NaN). Using last.pt for test.")
            best_ckpt_path = last_path
        else:
            best_ckpt_path = best_path

        ckpt = load_checkpoint(best_ckpt_path, map_location=str(device))
        model.load_state_dict(ckpt["model"], strict=True)

        test_loss, test_metrics = run_eval(model, test_loader, device, criterion)
        test_metrics = {k: float(v) for k, v in test_metrics.items() if k in ["AUPRC","AUROC","F1","ACC","SEN","MCC"]}
        test_metrics["loss"] = float(test_loss)
        test_metrics["best_val_AUPRC"] = float(best_val_auprc)
        test_metrics["best_epoch"] = int(best_epoch)

        write_result_csv(result_path, test_metrics)
        fold_results.append(test_metrics)

        print(f"[Fold {fold}] Test results saved to: {result_path}")
        print(test_metrics)

    # Summary
    summary_path = out_dir / "summary.csv"
    if len(fold_results) > 0:
        df_res = pd.DataFrame(fold_results)
        # compute mean/std for metrics
        metrics_cols = ["AUPRC","AUROC","F1","ACC","SEN","MCC","loss"]
        rows = []
        for m in metrics_cols:
            vals = df_res[m].astype(float).values
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append({"metric": m, "mean": mean, "std": std, "mean+-std": f"{mean:.6f}+-{std:.6f}"})
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(df_sum)
    else:
        print("No fold results collected; summary.csv not written.")


if __name__ == "__main__":
    main()
