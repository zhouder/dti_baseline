# -*- coding: utf-8 -*-
"""
MolTrans cross-validation training script (extended)
- 新增 split 模式: warm/hot, cold-both, cold-pair（原有 cold-protein / cold-drug 保留）
- 断点续训: 每折保存 best.pt / last.pt，--resume 时从 last.pt 续跑
- 其它保持与基线一致：AUPRC 监控 + 早停 + 验证集阈值搜索 + tqdm
"""
from __future__ import annotations
import os, math, csv, argparse, copy
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, matthews_corrcoef
)

# ===== MolTrans components =====
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder
import stream as _moltrans_stream  # for safe monkey-patch

# ---------- safety patch: 防止 __getitem__ 索引越界 ----------
_orig_getitem = _moltrans_stream.BIN_Data_Encoder.__getitem__
def _safe_getitem(self, index):
    try:
        idx = int(index)
    except Exception:
        idx = int(np.asarray(index).item())
    if hasattr(self, "list_IDs") and len(self.list_IDs) > 0:
        mapped = int(self.list_IDs[idx % len(self.list_IDs)])
    else:
        mapped = idx
    if mapped >= len(self.df):
        mapped = mapped % max(1, len(self.df))
    return _orig_getitem(self, mapped)
_moltrans_stream.BIN_Data_Encoder.__getitem__ = _safe_getitem
# -----------------------------------------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def compute_metrics(prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred = (prob >= thr).astype(np.int64)
    out["acc"] = float(accuracy_score(y_true, pred))
    out["sen"] = float(recall_score(y_true, pred))
    out["f1"]  = float(f1_score(y_true, pred))
    out["mcc"] = float(matthews_corrcoef(y_true, pred))
    try:    out["auroc"] = float(roc_auc_score(y_true, prob))
    except Exception: out["auroc"] = float("nan")
    try:    out["auprc"] = float(average_precision_score(y_true, prob))
    except Exception: out["auprc"] = float("nan")
    return out

def find_best_threshold(prob: np.ndarray, y_true: np.ndarray, grid=None) -> float:
    if grid is None: grid = np.linspace(0.01, 0.99, 199)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        try: f1 = f1_score(y_true, (prob >= t).astype(np.int64))
        except Exception: f1 = float("nan")
        if not np.isnan(f1) and f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return float(best_t)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "val"):
    model.train(False)
    loss_f = nn.BCELoss()
    tot_loss = 0.0
    probs, labels = [], []
    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for (d, p, d_mask, p_mask, label) in pbar:
        score  = model(d.long().to(device), p.long().to(device),
                       d_mask.long().to(device), p_mask.long().to(device))
        logits = torch.sigmoid(score).view(-1)
        y = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1)
        n = min(logits.numel(), y.numel())
        loss = loss_f(logits[:n], y[:n])
        tot_loss += float(loss.detach().cpu())
        probs.append(logits[:n].detach().float().cpu().numpy())
        labels.append(y[:n].detach().cpu().numpy())
    pbar.close()
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab  = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.float32)
    return tot_loss / max(1, len(loader)), prob, lab

def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    optimizer: optim.Optimizer, desc: str) -> float:
    model.train(True)
    loss_f = nn.BCELoss()
    tot_loss = 0.0
    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for (d, p, d_mask, p_mask, label) in pbar:
        optimizer.zero_grad(set_to_none=True)
        score  = model(d.long().to(device), p.long().to(device),
                       d_mask.long().to(device), p_mask.long().to(device))
        logits = torch.sigmoid(score).view(-1)
        y = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1)
        n = min(logits.numel(), y.numel())
        loss = loss_f(logits[:n], y[:n])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot_loss += float(loss.detach().cpu())
        pbar.set_postfix_str(f"loss={float(loss):.4f}")
    pbar.close()
    return tot_loss / max(1, len(loader))

# ------------ split utilities ------------
def _chunk_groups(groups: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed); rng.shuffle(uniq)
    return [uniq[k::n_splits] for k in range(n_splits)]

def make_outer_splits(mode: str, cv_folds: int, seed: int,
                      drug_key: np.ndarray, prot_key: np.ndarray,
                      labels: np.ndarray) -> List[np.ndarray]:
    N = len(labels); all_idx = np.arange(N)
    if mode == "cold-protein":
        gkf = GroupKFold(n_splits=cv_folds); return [te for _, te in gkf.split(all_idx, groups=prot_key)]
    if mode == "cold-drug":
        gkf = GroupKFold(n_splits=cv_folds); return [te for _, te in gkf.split(all_idx, groups=drug_key)]
    if mode == "cold-pair":
        pair = np.array([f"{d}||{p}" for d,p in zip(drug_key, prot_key)], dtype=object)
        gkf = GroupKFold(n_splits=cv_folds); return [te for _, te in gkf.split(all_idx, groups=pair)]
    if mode in ("warm","hot"):
        if len(np.unique(labels)) == 2:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            return [te for _, te in skf.split(all_idx, labels)]
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        return [te for _, te in kf.split(all_idx)]
    if mode == "cold-both":
        drug_folds = _chunk_groups(drug_key, cv_folds, seed)
        prot_folds = _chunk_groups(prot_key, cv_folds, seed+7)
        splits: List[np.ndarray] = []
        for k in range(cv_folds):
            td = set(drug_folds[k]); tp = set(prot_folds[k])
            test_mask = np.array([(d in td) and (p in tp) for d,p in zip(drug_key, prot_key)], dtype=bool)
            te_idx = np.where(test_mask)[0]
            if len(te_idx) < max(1, int(0.05 * N / cv_folds)):
                # 样本太少则退化到实体并集（更稳），避免空折
                test_mask = np.array([(d in td) or (p in tp) for d,p in zip(drug_key, prot_key)], dtype=bool)
                te_idx = np.where(test_mask)[0]
            splits.append(te_idx)
        return splits
    raise ValueError(f"Unknown split mode: {mode}")

def sample_val_indices(mode: str, pool_idx: np.ndarray, val_frac_in_pool: float, seed: int,
                       drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pool_idx = np.asarray(pool_idx)
    if mode in ("warm","hot"):
        if len(np.unique(labels)) == 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_in_pool, random_state=seed)
            tr_sub, va_sub = next(sss.split(pool_idx, labels[pool_idx]))
            return pool_idx[tr_sub], pool_idx[va_sub]
        n_va = max(1, int(round(val_frac_in_pool * len(pool_idx))))
        rng = np.random.default_rng(seed); rng.shuffle(pool_idx)
        return pool_idx[n_va:], pool_idx[:n_va]
    if mode == "cold-protein":
        g = prot_key[pool_idx]; uniq = np.unique(g)
        rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val]); mask = np.array([x in val_g for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]
    if mode == "cold-drug":
        g = drug_key[pool_idx]; uniq = np.unique(g)
        rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val]); mask = np.array([x in val_g for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]
    if mode == "cold-pair":
        pair = np.array([f"{d}||{p}" for d,p in zip(drug_key, prot_key)], dtype=object)[pool_idx]
        uniq = np.unique(pair); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        val_g = set(uniq[:n_val]); mask = np.array([x in val_g for x in pair], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]
    if mode == "cold-both":
        rng = np.random.default_rng(seed)
        d_pool = np.unique(drug_key[pool_idx]); rng.shuffle(d_pool)
        p_pool = np.unique(prot_key[pool_idx]); rng.shuffle(p_pool)
        nd = max(1, int(round(val_frac_in_pool * len(d_pool))))
        np_ = max(1, int(round(val_frac_in_pool * len(p_pool))))
        val_d = set(d_pool[:nd]); val_p = set(p_pool[:np_])
        va_mask = np.array([(d in val_d) and (p in val_p) for d,p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)
        va_idx  = pool_idx[va_mask]
        tr_mask = np.array([(d not in val_d) and (p not in val_p) for d,p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)
        tr_idx  = pool_idx[tr_mask]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            return sample_val_indices("cold-pair", pool_idx, val_frac_in_pool, seed, drug_key, prot_key, labels)
        return tr_idx, va_idx
    raise ValueError(f"unknown mode {mode}")
# -----------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="DAVIS / BindingDB / BioSNAP")
    ap.add_argument("--split-mode",
        choices=["warm","hot","cold-protein","cold-drug","cold-both","cold-pair"],
        default="cold-protein")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--overall-train", type=float, default=0.7)
    ap.add_argument("--overall-val", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out-dir", default="/root/lanyun-tmp/moltrans-runs", help="Optional output root")
    ap.add_argument("--suffix", default="", help="Suffix in output dir name")

    ap.add_argument("--early-stop", type=int, default=10, help="patience for early stopping (0=disabled)")
    ap.add_argument("--es-min-delta", type=float, default=0.0, help="minimum improvement to reset patience")

    # 断点续训
    ap.add_argument("--resume", action="store_true", help="resume this fold from fold_dir/last.pt if exists")
    ap.add_argument("--start-fold", type=int, default=0,
                    help="指定从第几折开始（1-based）。0 表示自动检测/从折1开始")
    return ap.parse_args()

def summarize_split(name: str, idx: np.ndarray,
                    drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray):
    idx = np.asarray(idx)
    d = drug_key[idx]
    p = prot_key[idx]
    y = labels[idx]

    n = len(idx)
    n_drug = len(np.unique(d))
    n_prot = len(np.unique(p))
    n_pair = len(np.unique(np.array([f"{dd}||{pp}" for dd, pp in zip(d, p)], dtype=object)))

    # 仅当是二分类时给出正例比例（可选）
    pos_ratio = None
    if len(np.unique(labels)) == 2:
        pos_ratio = float(np.mean(y))

    msg = (f"[{name}] samples={n} | uniq_drug={n_drug} | uniq_target={n_prot} | uniq_pair={n_pair}")
    if pos_ratio is not None:
        msg += f" | pos_ratio={pos_ratio:.4f}"
    print(msg)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try: print("GPU:", torch.cuda.get_device_name(0))
        except Exception: pass

    # ---- Load CSV ----
    ds_lower = args.dataset.lower()
    ds_cap   = {"bindingdb":"BindingDB", "davis":"DAVIS", "biosnap":"BioSNAP"}.get(ds_lower, args.dataset)
    data_root = Path("/root/lanyun-tmp")
    all_csv   = data_root / ds_cap / "all.csv"
    if not all_csv.exists():
        raise FileNotFoundError(f"not found: {all_csv}")
    df_all = pd.read_csv(all_csv)

    # unify columns
    rename_map = {}
    for k in list(df_all.columns):
        k_low = k.lower()
        if k_low == "smiles": rename_map[k] = "SMILES"
        if k_low in ("protein", "sequence", "target", "target sequence"): rename_map[k] = "Target Sequence"
        if k_low in ("label","y"): rename_map[k] = "Label"
    df_all = df_all.rename(columns=rename_map)
    if not {"SMILES","Target Sequence","Label"} <= set(df_all.columns):
        raise ValueError("CSV must contain columns: smiles, protein, label (any case)")
    df_all = df_all[["SMILES","Target Sequence","Label"]].dropna().reset_index(drop=True)

    N = len(df_all)
    print(f"[ALL] loaded: {N} rows from {all_csv}")

    prot_key = df_all["Target Sequence"].values
    drug_key = df_all["SMILES"].values
    labels   = df_all["Label"].values.astype(np.float32)

    # ---- Outer splits ----
    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer_splits = make_outer_splits(
        split_mode, args.cv_folds if args.cv_folds>1 else 5, args.seed,
        drug_key=np.asarray(drug_key), prot_key=np.asarray(prot_key), labels=np.asarray(labels)
    )
    overall_test    = 1.0 / len(outer_splits)
    val_frac_in_pool= args.overall_val / (1.0 - overall_test)
    print(f"[SPLIT] 模式={split_mode} | 折数={len(outer_splits)} | 目标总体比例 train:val:test = "
          f"{args.overall_train}:{args.overall_val}:{overall_test:.1f}")
    print("[SPLIT] 实施策略：每折 test=1/K；在其余 (1-1/K) 中按模式抽取 val，其余为 train；冷模式均按组不交叠")

    # ---- Output dir ----
    split_tag = split_mode
    base = args.out_dir if args.out_dir else f"/root/lanyun-tmp/moltrans-runs/{ds_cap}-{split_tag}"
    if args.suffix: base += (args.suffix if args.suffix.startswith("-") else f"-{args.suffix}")
    out_dir = Path(base); out_dir.mkdir(parents=True, exist_ok=True)
    print("[OUT]", out_dir)

    test_metrics_all: List[Dict[str, float]] = []
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    # ---- 依据 start-fold / resume 自动确定从哪一折开始 ----
    ep_total = int(BIN_config_DBPE().get("train_epoch", args.epochs or 13))

    start_fold = 1
    if args.start_fold and args.start_fold > 0:
        # 用户强制指定从第几折开始
        start_fold = max(1, min(len(outer_splits), int(args.start_fold)))
    elif args.resume:
        # 自动扫描：优先跳过带 done.flag 的折；否则找第一个有 last.pt 的折；都没有就从第一个未开始的折
        for k in range(1, len(outer_splits) + 1):
            fdir = out_dir / f"fold{k}"
            if (fdir / "done.flag").exists():
                start_fold = k + 1
                continue
            if (fdir / "last.pt").exists():
                start_fold = k
                break
            start_fold = k
            break

    if start_fold > len(outer_splits):
        print("[RESUME] 检测到所有折都已完成，退出。")
        import sys;

        sys.exit(0)

    # 用切片让外层循环从 start_fold 开始
    for fold_id, te_idx in enumerate(outer_splits[start_fold - 1:], start_fold):
        te_idx = np.asarray(te_idx)
        # 双冷：把含测试实体的样本从候选池剔除
        if split_mode == "cold-both":
            td = set(drug_key[te_idx]); tp = set(prot_key[te_idx])
            pool_mask = np.array([(d not in td) and (p not in tp) for d,p in zip(drug_key, prot_key)], dtype=bool)
            pool_idx = np.where(pool_mask)[0]
        else:
            pool_idx = np.setdiff1d(np.arange(N), te_idx)
        tr_idx, va_idx = sample_val_indices(split_mode, pool_idx, val_frac_in_pool, seed=args.seed + fold_id,
                                            drug_key=np.asarray(drug_key), prot_key=np.asarray(prot_key),
                                            labels=np.asarray(labels))

        # 重叠检查（冷模式应为0）
        if split_mode.startswith("cold"):
            set_tr_p, set_va_p, set_te_p = set(prot_key[tr_idx]), set(prot_key[va_idx]), set(prot_key[te_idx])
            set_tr_d, set_va_d, set_te_d = set(drug_key[tr_idx]), set(drug_key[va_idx]), set(drug_key[te_idx])
            print(f"[fold{fold_id}] overlap P: tr∩va={len(set_tr_p & set_va_p)}, tr∩te={len(set_tr_p & set_te_p)}, va∩te={len(set_va_p & set_te_p)} | "
                  f"D: tr∩va={len(set_tr_d & set_va_d)}, tr∩te={len(set_tr_d & set_te_d)}, va∩te={len(set_va_d & set_te_d)}")

        df_tr = df_all.iloc[tr_idx].copy().reset_index(drop=True)
        df_va = df_all.iloc[va_idx].copy().reset_index(drop=True)
        df_te = df_all.iloc[te_idx].copy().reset_index(drop=True)
        print(f"\n===== fold{fold_id} split summary =====")
        summarize_split("train", tr_idx, drug_key=np.asarray(drug_key), prot_key=np.asarray(prot_key),
                        labels=np.asarray(labels))
        summarize_split("val  ", va_idx, drug_key=np.asarray(drug_key), prot_key=np.asarray(prot_key),
                        labels=np.asarray(labels))
        summarize_split("test ", te_idx, drug_key=np.asarray(drug_key), prot_key=np.asarray(prot_key),
                        labels=np.asarray(labels))

        # （可选）再把三者的“实体重叠”也打印出来，warm 下也能看
        tr_d, va_d, te_d = set(drug_key[tr_idx]), set(drug_key[va_idx]), set(drug_key[te_idx])
        tr_p, va_p, te_p = set(prot_key[tr_idx]), set(prot_key[va_idx]), set(prot_key[te_idx])
        print(f"[overlap drugs]  tr∩va={len(tr_d & va_d)} | tr∩te={len(tr_d & te_d)} | va∩te={len(va_d & te_d)}")
        print(f"[overlap targets] tr∩va={len(tr_p & va_p)} | tr∩te={len(tr_p & te_p)} | va∩te={len(va_p & te_p)}")
        print("======================================\n")

        params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.workers, 'drop_last': False}
        dset_tr = BIN_Data_Encoder(np.arange(len(df_tr)), df_tr['Label'].values, df_tr)
        dset_va = BIN_Data_Encoder(np.arange(len(df_va)), df_va['Label'].values, df_va)
        dset_te = BIN_Data_Encoder(np.arange(len(df_te)), df_te['Label'].values, df_te)
        loader_tr = DataLoader(dset_tr, **params)
        loader_va = DataLoader(dset_va, **params)
        loader_te = DataLoader(dset_te, **params)

        # model & optimizer
        config = BIN_config_DBPE()
        if args.epochs is not None and args.epochs > 0:
            config["train_epoch"] = args.epochs
        model = BIN_Interaction_Flat(**config)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, dim=0)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        fold_dir = out_dir / f"fold{fold_id}"; fold_dir.mkdir(parents=True, exist_ok=True)
        csv_path = fold_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","thr"])

        # ---------- resume ----------
        start_ep = 1
        last_ckpt = fold_dir / "last.pt"
        if args.resume and last_ckpt.exists():
            ck = torch.load(str(last_ckpt), map_location="cpu")
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            start_ep = int(ck.get("epoch", 0)) + 1
            print(f"[RESUME] fold{fold_id} from epoch {start_ep}")

        best_score = -1.0; best_state = None; best_thr = 0.5
        best_metrics = {}; no_improve = 0; best_epoch = 0
        ep_total = int(config["train_epoch"])

        for ep in range(start_ep, ep_total + 1):
            tr_loss = train_one_epoch(model, loader_tr, device, optimizer, desc=f"{ds_cap}/fold{fold_id} train ep{ep}/{ep_total}")
            va_loss, prob_va, y_va = evaluate(model, loader_va, device, desc=f"{ds_cap}/fold{fold_id} val ep{ep}/{ep_total}")
            thr_now = find_best_threshold(prob_va, y_va)
            m = compute_metrics(prob_va, y_va, thr=thr_now)

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{va_loss:.6f}",
                            f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                            f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}", f"{thr_now:.6f}"])

            print(f"[{ds_cap}/fold{fold_id}] ep{ep:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | "
                  f"AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | F1 {m['f1']:.4f} | "
                  f"ACC {m['acc']:.4f} | SEN {m['sen']:.4f} | MCC {m['mcc']:.4f} | thr {thr_now:.3f}")

            score = m["auprc"] if not math.isnan(m["auprc"]) else m["auroc"]
            if score > best_score + args.es_min_delta:
                best_score = score
                best_state = copy.deepcopy((model.module if isinstance(model, nn.DataParallel) else model).state_dict())
                best_thr = float(thr_now); best_metrics = dict(m); best_epoch = ep; no_improve = 0
                torch.save({"epoch": ep, "state_dict": best_state,
                            "optimizer": optimizer.state_dict(),
                            "metrics": m}, str(fold_dir / "best.pt"))
            else:
                no_improve += 1

            # save last every epoch (for resume)
            torch.save({"epoch": ep,
                        "state_dict": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                        "optimizer": optimizer.state_dict()}, str(last_ckpt))

            if args.early_stop and no_improve >= args.early_stop:
                print(f"[EARLY-STOP] fold{fold_id}: no improvement for {args.early_stop} epochs (best at ep{best_epoch})")
                break

        if best_state is not None:
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(best_state)
        te_loss, prob_te, y_te = evaluate(model, loader_te, device, desc=f"{ds_cap}/fold{fold_id} test")
        te_m = compute_metrics(prob_te, y_te, thr=best_thr)
        print(f"[TEST/fold{fold_id}] thr={best_thr:.3f} | AUROC {te_m['auroc']:.4f} | AUPRC {te_m['auprc']:.4f} | "
              f"F1 {te_m['f1']:.4f} | ACC {te_m['acc']:.4f} | SEN {te_m['sen']:.4f} | MCC {te_m['mcc']:.4f} | loss {te_loss:.4f}")

        with open(fold_dir / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["VAL_BEST_EPOCH", best_epoch]); w.writerow(["VAL_BEST_THR", f"{best_thr:.6f}"])
            for k,v in best_metrics.items(): w.writerow([f"VAL_{k.upper()}", f"{v:.6f}"])
            for k,v in te_m.items():         w.writerow([f"TEST_{k.upper()}", f"{v:.6f}"])
        with open(fold_dir / "done.flag", "w", encoding="utf-8") as f:
            f.write(f"done best_epoch={best_epoch}\n")
        test_metrics_all.append(te_m)

    # CV summary
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    mean = {k: float(np.mean([m[k] for m in test_metrics_all])) for k in keys}
    std  = {k: float(np.std ([m[k] for m in test_metrics_all])) for k in keys}
    with open(out_dir / "cv_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","mean","std"])
        for k in keys: w.writerow([k.upper()],); f.seek(0,2)  # keep header
    with open(out_dir / "cv_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","mean","std"])
        for k in keys: w.writerow([k.upper(), f"{mean[k]:.6f}", f"{std[k]:.6f}"])


    print("[CV] " + " | ".join([f"{k.upper()} {mean[k]:.4f}±{std[k]:.4f}" for k in keys]))
