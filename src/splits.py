# src/splits.py
# -*- coding: utf-8 -*-
"""
Splitting utilities for DTI.

Used by train.py:
1) make_outer_splits(split_mode, n_splits, seed, drug_key, prot_key, y)
2) sample_val_indices(split_mode, train_idx, val_frac, seed, drug_key, prot_key, y)

split_mode:
  - warm / hot      : random KFold / StratifiedKFold (binary)
  - cold-drug       : train/test drug sets disjoint
  - cold-protein    : train/test protein sets disjoint
  - cold-pair       : train/test (drug,protein) pairs disjoint
  - cold-both       : strict both-cold:
                      train/test drug disjoint AND protein disjoint
                      implemented by test = union(drug_in_fold OR prot_in_fold)
"""

from __future__ import annotations
from typing import List, Tuple, Sequence, Any
import numpy as np


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed) & 0x7FFFFFFF)


def _to_np(a) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)


def _normalize_drug_key(k: Any) -> str:
    return "" if k is None else str(k).strip()


def _normalize_prot_key(k: Any) -> str:
    return "" if k is None else str(k).strip().upper()


def _normalize_keys(drug_key: Sequence, prot_key: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    d = _to_np(drug_key).astype(object, copy=False)
    p = _to_np(prot_key).astype(object, copy=False)
    d = np.asarray([_normalize_drug_key(x) for x in d], dtype=object)
    p = np.asarray([_normalize_prot_key(x) for x in p], dtype=object)
    return d, p


def _is_binary_classification(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    uniq = np.unique(y)
    if uniq.size != 2:
        return False
    return np.all(np.isin(uniq, [0, 1]))


def _pair_keys(drug_key: np.ndarray, prot_key: np.ndarray) -> np.ndarray:
    return (drug_key.astype(str) + "||" + prot_key.astype(str)).astype(object)


def _group_counts(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g, c = np.unique(keys, return_counts=True)
    return g.astype(object), c.astype(np.int64)


def _assign_groups_to_folds_by_size(groups: np.ndarray, counts: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    """
    Greedy bin packing by sample counts (more balanced than random group split).
    """
    r = _rng(seed)
    jitter = r.rand(len(groups))
    order = np.lexsort((jitter, -counts))  # big first
    g_sorted = groups[order]
    c_sorted = counts[order]

    bins: List[List[Any]] = [[] for _ in range(n_splits)]
    bin_sz = np.zeros((n_splits,), dtype=np.int64)

    for g, c in zip(g_sorted.tolist(), c_sorted.tolist()):
        k = int(np.argmin(bin_sz))
        bins[k].append(g)
        bin_sz[k] += int(c)

    return [np.asarray(b, dtype=object) for b in bins]


def _validate_disjoint(a: np.ndarray, b: np.ndarray, msg: str):
    if np.intersect1d(np.unique(a), np.unique(b)).size != 0:
        raise ValueError(msg)


def _kfold_indices(n: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    r = _rng(seed)
    idx = r.permutation(n)
    folds = np.array_split(idx, n_splits)
    out = []
    all_idx = np.arange(n, dtype=np.int64)
    for k in range(n_splits):
        te = folds[k].astype(np.int64)
        tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
        out.append((tr, te))
    return out


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    r = _rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    r.shuffle(idx0)
    r.shuffle(idx1)
    folds0 = np.array_split(idx0, n_splits)
    folds1 = np.array_split(idx1, n_splits)

    out = []
    all_idx = np.arange(len(y), dtype=np.int64)
    for k in range(n_splits):
        te = np.concatenate([folds0[k], folds1[k]], axis=0)
        r.shuffle(te)
        te = te.astype(np.int64)
        tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
        out.append((tr, te))
    return out


def make_outer_splits(
    split_mode: str,
    n_splits: int,
    seed: int,
    drug_key: Sequence,
    prot_key: Sequence,
    y: Sequence,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    split_mode = str(split_mode).strip().lower()
    y = _to_np(y).astype(np.float32, copy=False)
    n = len(y)

    d_raw = _to_np(drug_key)
    p_raw = _to_np(prot_key)
    assert len(d_raw) == n and len(p_raw) == n, "drug_key/prot_key/y 长度必须一致"

    d, p = _normalize_keys(d_raw, p_raw)
    all_idx = np.arange(n, dtype=np.int64)

    # warm/hot
    if split_mode in ("warm", "hot"):
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
            if _is_binary_classification(y):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in skf.split(np.zeros(n), y)]
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in kf.split(np.zeros(n))]
        except Exception:
            if _is_binary_classification(y):
                return _stratified_kfold_indices(y, n_splits, seed)
            return _kfold_indices(n, n_splits, seed)

    # cold-drug
    if split_mode == "cold-drug":
        groups, counts = _group_counts(d)
        folds = _assign_groups_to_folds_by_size(groups, counts, n_splits, seed)
        out = []
        for k in range(n_splits):
            te = np.where(np.isin(d, folds[k]))[0].astype(np.int64)
            tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
            _validate_disjoint(d[tr], d[te], "cold-drug violated: drug overlap between train/test")
            out.append((tr, te))
        return out

    # cold-protein
    if split_mode == "cold-protein":
        groups, counts = _group_counts(p)
        folds = _assign_groups_to_folds_by_size(groups, counts, n_splits, seed)
        out = []
        for k in range(n_splits):
            te = np.where(np.isin(p, folds[k]))[0].astype(np.int64)
            tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
            _validate_disjoint(p[tr], p[te], "cold-protein violated: protein overlap between train/test")
            out.append((tr, te))
        return out

    # cold-pair
    if split_mode == "cold-pair":
        pair = _pair_keys(d, p)
        groups, counts = _group_counts(pair)
        folds = _assign_groups_to_folds_by_size(groups, counts, n_splits, seed)
        out = []
        for k in range(n_splits):
            te = np.where(np.isin(pair, folds[k]))[0].astype(np.int64)
            tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
            _validate_disjoint(pair[tr], pair[te], "cold-pair violated: pair overlap between train/test")
            out.append((tr, te))
        return out

    # cold-both (strict): test = union(drug in fold OR protein in fold)
    if split_mode == "cold-both":
        d_groups, d_counts = _group_counts(d)
        p_groups, p_counts = _group_counts(p)

        d_folds = _assign_groups_to_folds_by_size(d_groups, d_counts, n_splits, seed)
        p_folds = _assign_groups_to_folds_by_size(p_groups, p_counts, n_splits, seed + 9973)

        out = []
        for k in range(n_splits):
            mask_d = np.isin(d, d_folds[k])
            mask_p = np.isin(p, p_folds[k])
            te = np.where(mask_d | mask_p)[0].astype(np.int64)
            tr = np.setdiff1d(all_idx, te, assume_unique=False).astype(np.int64)
            _validate_disjoint(d[tr], d[te], "cold-both violated: drug overlap between train/test")
            _validate_disjoint(p[tr], p[te], "cold-both violated: protein overlap between train/test")
            out.append((tr, te))
        return out

    raise ValueError(f"Unknown split_mode: {split_mode}")


def sample_val_indices(
    split_mode: str,
    train_idx: np.ndarray,
    val_frac: float,
    seed: int,
    drug_key: Sequence,
    prot_key: Sequence,
    y: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split outer-train indices into (train_sub, val) according to split_mode.
    For group-based splits, val is selected by groups to approximate val_frac by sample count.
    """
    split_mode = str(split_mode).strip().lower()
    train_idx = _to_np(train_idx).astype(np.int64)
    y = _to_np(y).astype(np.float32, copy=False)

    if train_idx.size == 0:
        return train_idx, train_idx

    d_raw = _to_np(drug_key)
    p_raw = _to_np(prot_key)
    d, p = _normalize_keys(d_raw, p_raw)

    n_tr = train_idx.size
    val_frac = float(val_frac)
    val_target = max(1, int(round(val_frac * n_tr)))
    r = _rng(seed)

    # warm/hot
    if split_mode in ("warm", "hot"):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            if _is_binary_classification(y):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                idx_in = train_idx
                y_in = y[idx_in]
                tr_sub_in, va_in = next(sss.split(np.zeros_like(y_in), y_in))
                return idx_in[tr_sub_in].astype(np.int64), idx_in[va_in].astype(np.int64)
        except Exception:
            pass
        idx = train_idx.copy()
        r.shuffle(idx)
        va = idx[:val_target]
        tr_sub = idx[val_target:]
        return tr_sub.astype(np.int64), va.astype(np.int64)

    def _pick_groups_by_count(keys_local: np.ndarray) -> np.ndarray:
        groups, counts = np.unique(keys_local, return_counts=True)
        counts = counts.astype(np.int64)

        jitter = r.rand(len(groups))
        order = np.lexsort((jitter, -counts))  # big first
        g_sorted = groups[order]
        c_sorted = counts[order]

        picked = []
        acc = 0
        for g, c in zip(g_sorted.tolist(), c_sorted.tolist()):
            if acc >= val_target and picked:
                break
            if len(picked) >= len(groups) - 1:
                break
            picked.append(g)
            acc += int(c)

        picked = np.asarray(picked, dtype=object)
        return np.isin(keys_local, picked)

    if split_mode == "cold-drug":
        keys_local = d[train_idx]
        m = _pick_groups_by_count(keys_local)
        va = train_idx[m]
        tr_sub = train_idx[~m]
        _validate_disjoint(d[tr_sub], d[va], "cold-drug (val) violated: drug overlap between train_sub/val")
        return tr_sub.astype(np.int64), va.astype(np.int64)

    if split_mode == "cold-protein":
        keys_local = p[train_idx]
        m = _pick_groups_by_count(keys_local)
        va = train_idx[m]
        tr_sub = train_idx[~m]
        _validate_disjoint(p[tr_sub], p[va], "cold-protein (val) violated: protein overlap between train_sub/val")
        return tr_sub.astype(np.int64), va.astype(np.int64)

    if split_mode == "cold-pair":
        pair = _pair_keys(d, p)
        keys_local = pair[train_idx]
        m = _pick_groups_by_count(keys_local)
        va = train_idx[m]
        tr_sub = train_idx[~m]
        _validate_disjoint(pair[tr_sub], pair[va], "cold-pair (val) violated: pair overlap between train_sub/val")
        return tr_sub.astype(np.int64), va.astype(np.int64)

    # cold-both val: strict union(drug in val_drugs OR prot in val_prots)
    if split_mode == "cold-both":
        local_d = d[train_idx]
        local_p = p[train_idx]

        u_d, inv_d = np.unique(local_d, return_inverse=True)
        cnt_d = np.bincount(inv_d).astype(np.int64)
        u_p, inv_p = np.unique(local_p, return_inverse=True)
        cnt_p = np.bincount(inv_p).astype(np.int64)

        # if too few groups, fallback to random (cannot guarantee strict both-cold inside train pool)
        if u_d.size < 2 or u_p.size < 2:
            idx = train_idx.copy()
            r.shuffle(idx)
            va = idx[:val_target]
            tr_sub = idx[val_target:]
            return tr_sub.astype(np.int64), va.astype(np.int64)

        # group -> positions (local positions)
        order_d = np.argsort(inv_d)
        start_d = np.cumsum(cnt_d) - cnt_d
        end_d = np.cumsum(cnt_d)
        pos_d = [order_d[start_d[i]:end_d[i]] for i in range(u_d.size)]

        order_p = np.argsort(inv_p)
        start_p = np.cumsum(cnt_p) - cnt_p
        end_p = np.cumsum(cnt_p)
        pos_p = [order_p[start_p[i]:end_p[i]] for i in range(u_p.size)]

        cand = []
        for i in range(u_d.size):
            cand.append(("drug", i, int(cnt_d[i])))
        for i in range(u_p.size):
            cand.append(("prot", i, int(cnt_p[i])))

        jitter = r.rand(len(cand))
        sizes_neg = np.asarray([-c[2] for c in cand], dtype=np.int64)
        ord2 = np.lexsort((jitter, sizes_neg))
        cand = [cand[i] for i in ord2.tolist()]

        sel = np.zeros((train_idx.size,), dtype=bool)
        covered = 0
        picked_d = set()
        picked_p = set()

        for typ, gi, _sz in cand:
            if covered >= val_target:
                break

            if typ == "drug":
                if len(picked_d) >= u_d.size - 1:
                    continue
                pos = pos_d[gi]
                picked_d.add(gi)
            else:
                if len(picked_p) >= u_p.size - 1:
                    continue
                pos = pos_p[gi]
                picked_p.add(gi)

            new_cnt = int((~sel[pos]).sum())
            if new_cnt == 0:
                continue
            sel[pos] = True
            covered += new_cnt

        va = train_idx[sel]
        tr_sub = train_idx[~sel]

        _validate_disjoint(d[tr_sub], d[va], "cold-both (val) violated: drug overlap between train_sub/val")
        _validate_disjoint(p[tr_sub], p[va], "cold-both (val) violated: protein overlap between train_sub/val")
        return tr_sub.astype(np.int64), va.astype(np.int64)

    raise ValueError(f"Unknown split_mode: {split_mode}")
