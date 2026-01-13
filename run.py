# -*- coding: utf-8 -*-
"""run.py

DrugBAN runner for CSV datasets under /root/lanyun-fs:
  - davis.csv / drugbank.csv / kiba.csv
Columns (case-insensitive, order-independent):
  - uid, cid, smile, seq, label

This script keeps DrugBAN model and modalities unchanged.
It implements strict warm/cold splits and offline feature caching (Plan B).

Cache is stored under: {out_root}/cache/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

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

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# DrugBAN local modules
from configs import get_cfg_defaults
from dataloader import DTIDataset
import dataloader as _drugban_dataloader
from models import DrugBAN


def _set_mp_safe():
    import torch.multiprocessing as mp
    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low2col = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low2col:
            return low2col[c.lower()]
    return None


def load_dataset_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    c_uid = _find_col(df, ["uid"])
    c_cid = _find_col(df, ["cid"])
    c_smiles = _find_col(df, ["smile", "smiles"])
    c_seq = _find_col(df, ["seq", "sequence", "protein"])
    c_y = _find_col(df, ["label", "y"])

    missing = []
    if c_smiles is None:
        missing.append("smile/smiles")
    if c_seq is None:
        missing.append("seq/sequence/protein")
    if c_y is None:
        missing.append("label/y")
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Current columns={list(df.columns)}")

    keep_cols = [c_smiles, c_seq, c_y]
    rename = {c_smiles: "SMILES", c_seq: "Protein", c_y: "Y"}

    if c_uid is not None:
        keep_cols.append(c_uid)
        rename[c_uid] = "UID"
    if c_cid is not None:
        keep_cols.append(c_cid)
        rename[c_cid] = "CID"

    df = df[keep_cols].rename(columns=rename)
    df = df.dropna().reset_index(drop=True)

    df["SMILES"] = df["SMILES"].astype(str)
    df["Protein"] = df["Protein"].astype(str)
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).reset_index(drop=True)

    if "UID" in df.columns:
        df["UID"] = df["UID"].astype(str)
    if "CID" in df.columns:
        df["CID"] = df["CID"].astype(str)

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


def _sanitize_graph_inplace(g):
    for k in list(g.ndata.keys()):
        if k != "h":
            g.ndata.pop(k)
    for k in list(g.edata.keys()):
        g.edata.pop(k)
    return g


def _maybe_patch_dataset_truncation(truncate: bool, max_nodes: int):
    if not truncate:
        return

    try:
        import dgl  # noqa
        import torch  # noqa
    except Exception:
        return

    if getattr(DTIDataset, "_patched_truncate_large", False):
        return

    def _getitem(self, index):
        import dgl
        import torch

        index = self.list_IDs[index]
        smi = self.df.iloc[index]["SMILES"]
        g = self.fc(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

        max_n = int(getattr(self, "max_drug_nodes", max_nodes))
        n_nodes = int(g.num_nodes())
        if n_nodes > max_n:
            keep = list(range(max_n))
            g = dgl.node_subgraph(g, keep)
            for k in list(g.ndata.keys()):
                if str(k).startswith("_"):
                    g.ndata.pop(k)
            for k in list(g.edata.keys()):
                if str(k).startswith("_"):
                    g.edata.pop(k)

        node_feats = g.ndata.pop("h")
        n = int(node_feats.shape[0])
        num_virtual = max_n - n
        if num_virtual < 0:
            num_virtual = 0

        virtual_bit = torch.zeros((n, 1), dtype=node_feats.dtype)
        node_feats = torch.cat((node_feats, virtual_bit), dim=1)
        g.ndata["h"] = node_feats

        if num_virtual > 0:
            v_feat = torch.cat(
                (
                    torch.zeros((num_virtual, 74), dtype=node_feats.dtype),
                    torch.ones((num_virtual, 1), dtype=node_feats.dtype),
                ),
                dim=1,
            )
            g.add_nodes(num_virtual, {"h": v_feat})

        try:
            g = g.add_self_loop()
        except Exception:
            g = dgl.add_self_loop(g)

        _sanitize_graph_inplace(g)

        seq = self.df.iloc[index]["Protein"]
        p = _drugban_dataloader.integer_label_protein(seq)
        y = self.df.iloc[index]["Y"]
        return g, p, y

    DTIDataset.__getitem__ = _getitem
    DTIDataset._patched_truncate_large = True


def graph_collate_func_safe(batch):
    import dgl

    gs, ps, ys = zip(*batch)
    gs2 = []
    for g in gs:
        _sanitize_graph_inplace(g)
        gs2.append(g)
    bg = dgl.batch(gs2)

    if torch.is_tensor(ps[0]):
        p = torch.stack([pp.long() for pp in ps], dim=0)
    else:
        p = torch.tensor(np.asarray(ps), dtype=torch.long)

    y = torch.tensor(ys, dtype=torch.float32).view(-1, 1)
    return bg, p, y


def compute_metrics(prob: np.ndarray, lab: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    lab = np.asarray(lab, dtype=np.float32).reshape(-1)
    y_true = lab.astype(np.int32)
    y_pred = (prob >= thr).astype(np.int32)

    out: Dict[str, float] = {}
    out["acc"] = float((y_pred == y_true).mean())

    if (not _HAS_SK) or (len(np.unique(y_true)) < 2):
        out.update({"auc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                    "recall": float("nan"), "precision": float("nan"), "mcc": float("nan")})
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
                raise RuntimeError(f"cold-both fold{k+1}: test or train_pool is empty.")
            out.append((tr_pool, te))
        return out

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


def _cache_tag(ds_name: str, max_nodes: int, truncate: bool) -> str:
    return f"{ds_name}_max{int(max_nodes)}_prot1200_trunc{int(bool(truncate))}"


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class CachedDTIView(Dataset):
    def __init__(self, indices: np.ndarray, cache_dir: Path):
        self.indices = np.asarray(indices, dtype=int)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        import dgl
        idx = int(self.indices[i])
        f = self.cache_dir / f"{idx}.bin"
        graphs, label_dict = dgl.load_graphs(str(f))
        g = graphs[0]
        _sanitize_graph_inplace(g)
        p = label_dict["p"][0].long()
        y = float(label_dict["y"][0].item())
        return g, p, y


def build_cache_if_needed(
    df_all: pd.DataFrame,
    cache_dir: Path,
    max_drug_nodes: int,
    truncate_large_mols: bool,
    refresh: bool = False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"

    if refresh and cache_dir.exists():
        for p in cache_dir.glob("*.bin"):
            try:
                p.unlink()
            except Exception:
                pass

    N = len(df_all)
    existing = sum(1 for _ in cache_dir.glob("*.bin"))
    if existing >= N and meta_path.exists() and (not refresh):
        print(f"[CACHE] hit: {cache_dir} (files={existing}/{N})")
        return

    print(f"[CACHE] build -> {cache_dir} (existing={existing}/{N})")

    base = DTIDataset(np.arange(N), df_all, max_drug_nodes=int(max_drug_nodes))

    import dgl

    def _build_one(idx: int) -> Tuple[int, Optional[str]]:
        out_f = cache_dir / f"{idx}.bin"
        if out_f.exists():
            return idx, None
        try:
            g, p, y = base[idx]
            _sanitize_graph_inplace(g)
            p_t = torch.as_tensor(p, dtype=torch.long).view(1, -1)
            y_t = torch.as_tensor([float(y)], dtype=torch.float32).view(1, 1)
            dgl.save_graphs(str(out_f), [g], {"p": p_t, "y": y_t})
            return idx, None
        except Exception as e:
            return idx, repr(e)

    bad: List[Tuple[int, str]] = []
    it = range(N)
    if tqdm is not None:
        it = tqdm(it, total=N, ncols=120, desc="[CACHE]", leave=True)

    for idx in it:
        _i, err = _build_one(int(idx))
        if err is not None:
            bad.append((_i, err))

    meta = {
        "N": int(N),
        "max_drug_nodes": int(max_drug_nodes),
        "truncate_large_mols": bool(truncate_large_mols),
        "torch": getattr(torch, "__version__", ""),
        "torch_cuda": getattr(torch.version, "cuda", None),
    }
    try:
        import dgl  # noqa
        meta["dgl"] = getattr(dgl, "__version__", "")
    except Exception:
        meta["dgl"] = ""
    _write_json(meta_path, meta)

    if bad:
        bad_path = cache_dir / "bad_items.csv"
        with open(bad_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "error"])
            for i, e in bad:
                w.writerow([i, e])
        print(f"[CACHE] WARN: {len(bad)} items failed; see {bad_path}")
    else:
        print("[CACHE] done.")


def filter_indices_by_cache(indices: np.ndarray, cache_dir: Path) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    ok = []
    for i in indices:
        if (Path(cache_dir) / f"{int(i)}.bin").exists():
            ok.append(int(i))
    return np.asarray(ok, dtype=int)


@torch.no_grad()
def eval_epoch(model: DrugBAN, loader: DataLoader, device: torch.device, crit: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(False)
    loss_sum = 0.0
    n = 0
    probs, labs = [], []

    it = loader
    if tqdm is not None:
        it = tqdm(loader, total=len(loader), ncols=120, leave=False, desc="eval")

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
    loss_sum = 0.0
    n = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, total=len(loader), ncols=120, leave=False, desc="train")

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
    ap.add_argument("--dataset", required=True, help="davis / drugbank / kiba (or provide --csv)")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--csv", type=str, default="", help="Optional explicit csv path (overrides --dataset)")
    ap.add_argument("--out", type=str, default="/root/lanyun-tmp/drugban-runs")
    ap.add_argument("--cfg", type=str, default="configs/DrugBAN.yaml")

    ap.add_argument("--split-mode",
                    choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
                    default="warm")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)

    ap.add_argument("--max-drug-nodes", type=int, default=290)
    ap.add_argument("--truncate-large-mols", action="store_true")

    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--label-thr", type=float, default=None)
    ap.add_argument("--label-thr-op", choices=["ge", "le"], default="ge")

    ap.add_argument("--pos-weight", choices=["auto", "none"], default="auto")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=1e-4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--suppress-warnings", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*TensorDispatcher: dlopen failed.*")
        warnings.filterwarnings("ignore", message=".*You are using `torch.load`.*")

    set_seed(args.seed)
    _set_mp_safe()
    _maybe_patch_dataset_truncation(bool(args.truncate_large_mols), int(args.max_drug_nodes))

    if args.csv:
        csv_path = Path(args.csv)
        ds_name = csv_path.stem
    else:
        ds_name = args.dataset.strip().lower()
        csv_path = Path(args.data_root) / f"{ds_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df_all = load_dataset_csv(csv_path)
    print(f"[ALL] loaded {len(df_all)} rows from {csv_path}")

    if args.label_thr is not None:
        thr = float(args.label_thr)
        if args.label_thr_op == "ge":
            df_all["Y"] = (df_all["Y"].astype(float) >= thr).astype(np.float32)
        else:
            df_all["Y"] = (df_all["Y"].astype(float) <= thr).astype(np.float32)
        print(f"[LABEL] binarized by {args.label_thr_op} {thr}")

    labels = df_all["Y"].values.astype(np.float32)

    # Prefer CID/UID as entity keys for cold splits when available
    if "CID" in df_all.columns:
        drug_key = df_all["CID"].values.astype(object)
        drug_key_name = "CID"
    else:
        drug_key = df_all["SMILES"].values.astype(object)
        drug_key_name = "SMILES"

    if "UID" in df_all.columns:
        prot_key = df_all["UID"].values.astype(object)
        prot_key_name = "UID"
    else:
        prot_key = df_all["Protein"].values.astype(object)
        prot_key_name = "Protein"

    print(f"[KEY] drug_key={drug_key_name} | prot_key={prot_key_name}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.SOLVER.MAX_EPOCH = int(args.epochs)
    cfg.SOLVER.BATCH_SIZE = int(args.batch_size)
    cfg.SOLVER.NUM_WORKERS = int(args.workers)
    cfg.SOLVER.LR = float(args.lr)
    cfg.SOLVER.SEED = int(args.seed)
    cfg.DA.USE = False

    out_root = Path(args.out)
    run_dir = out_root / f"{ds_name}_{args.split_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_root = out_root / "cache"
    cache_dir = cache_root / _cache_tag(ds_name, int(args.max_drug_nodes), bool(args.truncate_large_mols))

    build_cache_if_needed(
        df_all=df_all,
        cache_dir=cache_dir,
        max_drug_nodes=int(args.max_drug_nodes),
        truncate_large_mols=bool(args.truncate_large_mols),
        refresh=bool(args.refresh_cache),
    )

    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer_splits = make_outer_splits(split_mode, args.cv_folds, args.seed, drug_key, prot_key, labels)
    K = len(outer_splits)
    val_frac_in_pool = 0.10 / (1.0 - 1.0 / K)

    print(f"[SPLIT] target train/val/test = 0.70/0.10/0.20 | K={K} | val_frac_in_pool={val_frac_in_pool:.4f}")
    print(f"[OUT] {run_dir}")
    print(f"[CACHE] {cache_dir}")

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

        tr_idx, va_idx = sample_val_indices(
            split_mode,
            train_pool_idx,
            val_frac_in_pool,
            args.seed + 100 + fold,
            drug_key,
            prot_key,
            labels,
        )

        tr_idx = filter_indices_by_cache(tr_idx, cache_dir)
        va_idx = filter_indices_by_cache(va_idx, cache_dir)
        te_idx = filter_indices_by_cache(test_idx, cache_dir)

        print(f"[COUNT] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
        summarize_split("train", tr_idx, drug_key, prot_key, labels)
        summarize_split("val", va_idx, drug_key, prot_key, labels)
        summarize_split("test", te_idx, drug_key, prot_key, labels)

        train_set = CachedDTIView(tr_idx, cache_dir)
        val_set = CachedDTIView(va_idx, cache_dir)
        test_set = CachedDTIView(te_idx, cache_dir)

        dl_train = dict(
            batch_size=int(cfg.SOLVER.BATCH_SIZE),
            num_workers=int(cfg.SOLVER.NUM_WORKERS),
            collate_fn=graph_collate_func_safe,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(cfg.SOLVER.NUM_WORKERS) > 0),
        )
        dl_eval = dict(
            batch_size=int(cfg.SOLVER.BATCH_SIZE),
            num_workers=int(cfg.SOLVER.NUM_WORKERS),
            collate_fn=graph_collate_func_safe,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(cfg.SOLVER.NUM_WORKERS) > 0),
        )
        if int(cfg.SOLVER.NUM_WORKERS) > 0:
            dl_train["multiprocessing_context"] = "spawn"
            dl_eval["multiprocessing_context"] = "spawn"

        train_loader = DataLoader(train_set, shuffle=True, **dl_train)
        val_loader = DataLoader(val_set, shuffle=False, **dl_eval)
        test_loader = DataLoader(test_set, shuffle=False, **dl_eval)

        model = DrugBAN(**cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.SOLVER.LR))

        if args.pos_weight == "auto":
            y_tr = labels[tr_idx]
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

        if not log_csv.exists():
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "lr",
                    "train_loss", "val_loss",
                    "val_auc", "val_auprc", "val_thr",
                    "val_f1", "val_acc", "val_recall", "val_precision", "val_mcc",
                    "is_best_val_auprc", "time_sec",
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
            improved = np.isfinite(cur) and (cur > best_val_auprc + args.es_min_delta)
            if improved:
                prev = best_val_auprc
                best_val_auprc = cur
                bad_cnt = 0
                torch.save({"model": model.state_dict()}, best_pt)
                print(f">>> [BEST] val AUPRC: {prev:.6f} -> {best_val_auprc:.6f} | saved best.pt")
            else:
                bad_cnt += 1

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
                    float(tr_loss), float(va_loss),
                    float(va_met["auc"]), float(va_met["auprc"]), float(thr_now),
                    float(va_met["f1"]), float(va_met["acc"]), float(va_met["recall"]),
                    float(va_met["precision"]), float(va_met["mcc"]),
                    int(improved), float(dt),
                ])

            if args.patience > 0 and bad_cnt >= args.patience:
                print(f"[EarlyStop] no improve for {bad_cnt} epochs. stop.")
                break

        used = "best.pt" if best_pt.exists() else "last.pt"
        ckpt = best_pt if best_pt.exists() else last_pt
        ck = torch.load(ckpt, map_location="cpu")
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
        else:
            model.load_state_dict(ck, strict=False)

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
