# src/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from tqdm import tqdm

from src.datamodule import (
    DataModule, DMConfig, CacheDirs, CacheDims,
    read_smiles_protein_label,
)

from src.splits import make_outer_splits, sample_val_indices

# metrics
SKLEARN = True
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, recall_score, matthews_corrcoef
    )
except Exception:
    SKLEARN = False


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_metrics(prob: np.ndarray, lab: np.ndarray) -> Dict[str, float]:
    if prob is None or lab is None or prob.size == 0:
        return {k: float("nan") for k in ["auc", "auprc", "f1", "acc", "recall", "mcc"]}

    y_true = lab.astype(np.int32)
    y_score = prob.astype(np.float32)
    y_pred = (y_score >= 0.5).astype(np.int32)

    if not SKLEARN:
        return {
            "auc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "acc": float((y_pred == y_true).mean()),
            "recall": float("nan"),
            "mcc": float("nan"),
        }

    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["f1"] = float(f1_score(y_true, y_pred))
    except Exception:
        out["f1"] = float("nan")
    try:
        out["acc"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        out["acc"] = float("nan")
    try:
        out["recall"] = float(recall_score(y_true, y_pred))
    except Exception:
        out["recall"] = float("nan")
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = float("nan")
    return out


def _batch_to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        out = []
        for x in batch:
            out.append(x.to(device) if torch.is_tensor(x) else x)
        return tuple(out)
    return batch.to(device) if torch.is_tensor(batch) else batch


def _forward(model: nn.Module, batch):
    # batch: (P,Pm,D,Dm,C,pocket_list,y)
    P, Pm, D, Dm, C, pocket_list, y = batch
    logits = model(P, Pm, D, Dm, C, pocket_list)
    return logits, y


def build_model_from_src(dims: CacheDims, args) -> nn.Module:
    model_mod = importlib.import_module("src.model")

    if hasattr(model_mod, "build_model"):
        cfg = {
            "d_protein": int(dims.esm2),
            "d_molclr": int(dims.molclr),
            "d_chem": int(dims.chemberta),

            "d_model": int(args.d_model),
            "dropout": float(args.dropout),

            # compat keys
            "n_heads": int(args.n_heads),
            "nhead": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "nlayers": int(args.n_layers),

            "pocket_in_dim": int(args.pocket_in_dim),
            "pocket_hidden": int(args.pocket_hidden),
        }
        return model_mod.build_model(cfg)

    if args.model_class:
        for name, c in inspect.getmembers(model_mod, inspect.isclass):
            if name == args.model_class and issubclass(c, nn.Module):
                return c(dims)

    raise RuntimeError("找不到模型：请在 src/model.py 提供 build_model(cfg) 或 --model-class")


def _write_log_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr",
            "train_loss", "train_auc", "train_auprc", "train_f1", "train_acc", "train_recall", "train_mcc",
            "val_loss",   "val_auc",   "val_auprc",   "val_f1",   "val_acc",   "val_recall",   "val_mcc",
            "is_best_val_auprc",
            "time_sec",
        ])


def _append_log(path: Path, row: list):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def _save_kv_csv(path: Path, kv: Dict[str, float | int | str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in kv.items():
            w.writerow([k, v])


def _load_kv_csv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


def _save_resume_state(path: Path, epoch: int, optimizer: optim.Optimizer, scheduler,
                      best_val_auprc: float, bad_cnt: int, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "optimizer": optimizer.state_dict(),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "best_val_auprc": float(best_val_auprc),
        "bad_cnt": int(bad_cnt),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }, path)


def _try_resume(state_path: Path, last_path: Path, model: nn.Module,
                optimizer: optim.Optimizer, scheduler, device: torch.device, scaler=None):
    st = torch.load(state_path, map_location="cpu")
    sd = torch.load(last_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    optimizer.load_state_dict(st["optimizer"])
    _optimizer_to_device(optimizer, device)

    if scheduler is not None and st.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(st["scheduler"])
        except Exception:
            pass

    if scaler is not None and st.get("scaler") is not None:
        try:
            scaler.load_state_dict(st["scaler"])
        except Exception:
            pass

    start_epoch = int(st.get("epoch", 0)) + 1
    best_val_auprc = float(st.get("best_val_auprc", -1.0))
    bad_cnt = int(st.get("bad_cnt", 0))
    return start_epoch, best_val_auprc, bad_cnt


def train_epoch_collect_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    amp: str = "off",
    scaler=None,
    amp_dtype=None
) -> Tuple[float, Dict[str, float]]:
    model.train(True)
    crit = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    n = 0
    probs, labs = [], []

    pbar = tqdm(total=len(loader.dataset), ncols=120, unit="ex", leave=True)
    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if amp != "off":
            with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                                dtype=amp_dtype, enabled=True):
                logits, y = _forward(model, batch)
                loss = crit(logits, y)
        else:
            logits, y = _forward(model, batch)
            loss = crit(logits, y)

        if amp == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = int(y.shape[0])
        loss_sum += float(loss.detach().item()) * bs
        n += bs

        with torch.no_grad():
            probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
            labs.append(y.detach().float().cpu().numpy())

        pbar.update(bs)
        pbar.set_postfix(loss=f"{loss_sum/max(1,n):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    pbar.close()

    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab_all = np.concatenate(labs, axis=0) if labs else np.zeros((0,), dtype=np.float32)
    met = compute_metrics(prob_all, lab_all)
    return loss_sum / max(1, n), met


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
               amp: str = "off", amp_dtype=None) -> Tuple[float, Dict[str, float]]:
    model.train(False)
    crit = nn.BCEWithLogitsLoss()

    loss_sum = 0.0
    n = 0
    probs, labs = [], []

    for batch in loader:
        batch = _batch_to_device(batch, device)

        if amp != "off":
            with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                                dtype=amp_dtype, enabled=True):
                logits, y = _forward(model, batch)
                loss = crit(logits, y)
        else:
            logits, y = _forward(model, batch)
            loss = crit(logits, y)

        bs = int(y.shape[0])
        loss_sum += float(loss.detach().item()) * bs
        n += bs

        probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        labs.append(y.detach().float().cpu().numpy())

    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab_all = np.concatenate(labs, axis=0) if labs else np.zeros((0,), dtype=np.float32)
    met = compute_metrics(prob_all, lab_all)
    return loss_sum / max(1, n), met


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", type=str, default="/root/lanyun-tmp")
    ap.add_argument("--cache-root", type=str, default="/root/lanyun-tmp/cache")
    ap.add_argument("--out", type=str, default="/root/lanyun-tmp/ugca-runs")
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--split-mode",
                    choices=["warm", "hot", "cold-protein", "cold-drug", "cold-both", "cold-pair"],
                    default="warm")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--lr-sched", choices=["off", "cosine"], default="cosine")
    ap.add_argument("--eta-min", type=float, default=1e-6)
    ap.add_argument("--cosine-frac", type=float, default=0.5,
                    help="前 cosine_frac*epochs 衰减到 eta_min")

    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=1e-4)

    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    ap.add_argument("--model-class", default="")

    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # pocket hyperparams still configurable (but pocket is always used)
    ap.add_argument("--pocket-in-dim", type=int, default=21)
    ap.add_argument("--pocket-hidden", type=int, default=128)

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ds_lower = args.dataset.strip().lower()
    ds = {"bindingdb": "BindingDB", "davis": "DAVIS", "biosnap": "BioSNAP"}.get(ds_lower, args.dataset)

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root) if args.cache_root else (data_root / "cache")
    all_csv = data_root / ds / "all.csv"
    if not all_csv.exists():
        raise FileNotFoundError(f"未找到 {all_csv}")

    smiles, proteins, labels = read_smiles_protein_label(str(all_csv))
    N = len(labels)
    print(f"[ALL] loaded {N} rows from {all_csv}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # AMP
    amp = args.amp
    amp_dtype = None
    scaler = None
    if amp == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif amp == "bf16":
        amp_dtype = torch.bfloat16

    # cache dirs
    cache_dirs = CacheDirs(
        esm_dir=str(cache_root / "esm2" / ds),
        molclr_dir=str(cache_root / "molclr" / ds),
        chemberta_dir=str(cache_root / "chemberta" / ds),
        pocket_dir=str(cache_root / "pocket" / ds),
    )
    dims = CacheDims(esm2=1280, molclr=300, chemberta=384)

    drug_key = np.asarray(smiles, dtype=object)
    prot_key = np.asarray(proteins, dtype=object)
    y_arr = np.asarray(labels, dtype=np.float32)

    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer_splits = make_outer_splits(split_mode, args.cv_folds, args.seed, drug_key, prot_key, y_arr)

    # enforce 7:1:2 overall => val_frac_in_pool = 0.1 / (1 - 1/K)
    K = len(outer_splits)
    val_frac_in_pool = 0.10 / (1.0 - 1.0 / K)
    print(f"[SPLIT] target train/val/test = 0.70/0.10/0.20 | K={K} | val_frac_in_pool={val_frac_in_pool:.4f}")

    run_dir = Path(args.out) / f"{ds}_{args.split_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] {run_dir}")

    keys = ["auc", "auprc", "f1", "acc", "recall", "mcc"]
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        fold_id = fold - 1
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        fold_dir = run_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_pt = fold_dir / "best.pt"
        last_pt = fold_dir / "last.pt"
        log_csv = fold_dir / "log.csv"
        result_csv = fold_dir / "result.csv"
        state_pt = fold_dir / ".state.pt"

        print("=" * 80)
        print(f"[Fold {fold}/{K}] train_pool={len(train_idx)} test={len(test_idx)} mode={split_mode} pocket=ON")

        if args.resume and result_csv.exists():
            r = _load_kv_csv(result_csv)
            if all(k in r for k in keys):
                fold_metrics.append({k: float(r[k]) for k in keys})
                print(f"[Resume] fold_{fold_id} already finished (result.csv exists). skip.")
                continue

        tr_idx, va_idx = sample_val_indices(
            split_mode,
            train_idx,
            val_frac_in_pool,
            args.seed + fold,
            drug_key,
            prot_key,
            y_arr,
        )

        def slice_trip(ix: np.ndarray):
            ix = ix.tolist()
            return [smiles[i] for i in ix], [proteins[i] for i in ix], [labels[i] for i in ix]

        tr = slice_trip(tr_idx)
        va = slice_trip(va_idx)
        te = slice_trip(test_idx)

        print(f"[COUNT] train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)} "
              f"(ratio≈{len(tr_idx)/N:.3f}:{len(va_idx)/N:.3f}:{len(test_idx)/N:.3f})")

        dm = DataModule(
            DMConfig(
                train_data=tr, val_data=va, test_data=te,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=False,
                persistent_workers=(args.workers > 0),
                prefetch_factor=2,
                drop_last=False,
            ),
            cache_dirs=cache_dirs,
            verbose=False,
        )
        s = dm.summary()
        print(f"[CACHE] esm_dir_used={s['esm_dir_used']} | idx_size={s['index_size']} | hit(train/val/test)={s['hit_rate']}")
        print(f"[UNIQ] train={s['uniq']['train']} | val={s['uniq']['val']} | test={s['uniq']['test']}")

        train_loader = dm.train_loader()
        val_loader = dm.val_loader()
        test_loader = dm.test_loader()

        model = build_model_from_src(dims, args).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = None
        tmax = max(1, int(args.epochs * max(0.05, min(args.cosine_frac, 1.0))))
        if args.lr_sched == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=args.eta_min)
            print(f"[LR] cosine: T_max={tmax}, eta_min={args.eta_min}")

        _write_log_header(log_csv)

        best_val_auprc = -1.0
        bad_cnt = 0
        start_epoch = 1

        if args.resume and state_pt.exists() and last_pt.exists():
            start_epoch, best_val_auprc, bad_cnt = _try_resume(
                state_pt, last_pt, model, optimizer, scheduler, device, scaler=scaler
            )
            print(f"[Resume] fold_{fold_id}: start_epoch={start_epoch}, best_val_auprc={best_val_auprc:.6f}, bad_cnt={bad_cnt}")

        for ep in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_met = train_epoch_collect_metrics(
                model, train_loader, device, optimizer, amp=amp, scaler=scaler, amp_dtype=amp_dtype
            )
            va_loss, va_met = eval_epoch(model, val_loader, device, amp=amp, amp_dtype=amp_dtype)

            cur = va_met.get("auprc", float("nan"))
            improved = (cur > best_val_auprc + args.es_min_delta) if np.isfinite(cur) else False
            if improved:
                prev = best_val_auprc
                best_val_auprc = float(cur)
                bad_cnt = 0
                torch.save(model.state_dict(), best_pt)
                print(f">>> [BEST] val AUPRC improved: {prev:.6f} -> {best_val_auprc:.6f} | saved best.pt")
            else:
                bad_cnt += 1

            torch.save(model.state_dict(), last_pt)
            _save_resume_state(state_pt, ep, optimizer, scheduler, best_val_auprc, bad_cnt, scaler=scaler)

            lr_now = float(optimizer.param_groups[0]["lr"])
            dt = time.time() - t0

            print(f"[Epoch {ep:03d}] train_auprc={tr_met['auprc']:.4f} val_auprc={va_met['auprc']:.4f} "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} lr={lr_now:.2e} time={dt:.1f}s")

            _append_log(log_csv, [
                ep, lr_now,
                float(tr_loss), float(tr_met["auc"]), float(tr_met["auprc"]), float(tr_met["f1"]),
                float(tr_met["acc"]), float(tr_met["recall"]), float(tr_met["mcc"]),
                float(va_loss), float(va_met["auc"]), float(va_met["auprc"]), float(va_met["f1"]),
                float(va_met["acc"]), float(va_met["recall"]), float(va_met["mcc"]),
                int(improved),
                float(dt),
            ])

            if scheduler is not None:
                if ep <= tmax:
                    scheduler.step()
                else:
                    for pg in optimizer.param_groups:
                        pg["lr"] = args.eta_min

            if args.early_stop > 0 and bad_cnt >= args.early_stop:
                print(f"[EarlyStop] no improve for {bad_cnt} epochs. stop.")
                break

        # test with best.pt if exists
        used = "best.pt" if best_pt.exists() else "last.pt"
        ckpt = best_pt if best_pt.exists() else last_pt
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

        te_loss, te_met = eval_epoch(model, test_loader, device, amp=amp, amp_dtype=amp_dtype)
        fold_metrics.append(te_met)

        print(f"[TEST] used={used} loss={te_loss:.4f} auc={te_met['auc']:.4f} auprc={te_met['auprc']:.4f} f1={te_met['f1']:.4f}")

        _save_kv_csv(result_csv, {
            "fold": fold_id,
            "used_ckpt": used,
            "test_loss": float(te_loss),
            "auc": float(te_met["auc"]),
            "auprc": float(te_met["auprc"]),
            "f1": float(te_met["f1"]),
            "acc": float(te_met["acc"]),
            "recall": float(te_met["recall"]),
            "mcc": float(te_met["mcc"]),
            "best_val_auprc": float(best_val_auprc),
            "pocket": "ON",
        })

    # summary
    mean = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys}
    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for i, m in enumerate(fold_metrics):
            w.writerow([i] + [m[k] for k in keys])
        w.writerow(["mean"] + [mean[k] for k in keys])
        w.writerow(["std"] + [std[k] for k in keys])

    print("=" * 80)
    print(f"[Saved] {summary_csv}")
    for k in keys:
        print(f"  {k}: {mean[k]:.4f} ± {std[k]:.4f}")


if __name__ == "__main__":
    main()
