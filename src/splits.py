"""
splits.py

DTI 5-fold splits supporting:
- warm
- cold-drug
- cold-protein
- cold-both

The split protocol follows:
- 5-fold CV: 4 folds train, 1 fold test
- validation split: randomly sample 1/8 from the training set
  => overall ratio train:val:test = 7:1:2 (approximately for cold settings)

Hash ID rules:
- did = sha1(smiles)[:24]
- pid = sha1(seq)[:24]

These rules match your feature files naming convention.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SplitMode = Literal["warm", "cold-drug", "cold-protein", "cold-both"]


def sha1_24(text: str) -> str:
    """Return sha1(text)[:24] as lowercase hex string."""
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def add_hash_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns:
      - did from smiles
      - pid from seq
    """
    if "smiles" not in df.columns or "seq" not in df.columns:
        raise ValueError("CSV must contain columns: smiles, seq")
    df = df.copy()
    df["did"] = df["smiles"].astype(str).map(sha1_24)
    df["pid"] = df["seq"].astype(str).map(sha1_24)
    return df


def sanitize_label_series(label: pd.Series) -> pd.Series:
    """
    Ensure binary {0,1} labels.

    - If labels already only contain {0,1}, keep them.
    - Otherwise, binarize with threshold 0.5 (common when labels are floats 0/1).
    """
    y = pd.to_numeric(label, errors="coerce").fillna(0.0)
    uniq = sorted(y.unique().tolist())
    if set(uniq).issubset({0, 1}) and len(uniq) <= 2:
        return y.astype(int)
    return (y.astype(float) >= 0.5).astype(int)


@dataclass
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}


def _split_entities_to_folds(entities: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    """Shuffle unique entities and split into n_splits folds."""
    rng = np.random.RandomState(seed)
    ent = np.array(entities, dtype=object)
    rng.shuffle(ent)
    folds = np.array_split(ent, n_splits)
    return folds


def _sample_val_from_train(train_idx: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx2, val_idx) by sampling from train_idx."""
    rng = np.random.RandomState(seed)
    train_idx = np.array(train_idx, dtype=int)
    rng.shuffle(train_idx)
    n_val = max(1, int(round(len(train_idx) * val_ratio)))
    val_idx = train_idx[:n_val]
    train_idx2 = train_idx[n_val:]
    return train_idx2, val_idx


def make_five_fold_splits(
    df: pd.DataFrame,
    mode: SplitMode,
    n_splits: int = 5,
    seed: int = 42,
    val_ratio_in_train: float = 1.0 / 8.0,
) -> List[FoldSplit]:
    """
    Create 5 folds.

    Returns a list of FoldSplit (fold id starts from 1).
    """
    if "did" not in df.columns or "pid" not in df.columns:
        raise ValueError("DataFrame must include did & pid. Call add_hash_ids() first.")

    df = df.reset_index(drop=True)
    n = len(df)
    all_idx = np.arange(n, dtype=int)

    splits: List[FoldSplit] = []

    if mode == "warm":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (trainval_idx, test_idx) in enumerate(kf.split(all_idx), start=1):
            train_idx, val_idx = _sample_val_from_train(trainval_idx, val_ratio_in_train, seed + fold)
            splits.append(FoldSplit(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=np.array(test_idx)))
        return splits

    if mode in ("cold-drug", "cold-protein"):
        group_col = "did" if mode == "cold-drug" else "pid"
        uniq = df[group_col].unique()
        folds = _split_entities_to_folds(uniq, n_splits=n_splits, seed=seed)

        for fold in range(1, n_splits + 1):
            test_groups = set(folds[fold - 1].tolist())
            is_test = df[group_col].isin(test_groups).values
            test_idx = all_idx[is_test]
            trainval_idx = all_idx[~is_test]
            train_idx, val_idx = _sample_val_from_train(trainval_idx, val_ratio_in_train, seed + fold)
            splits.append(FoldSplit(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
        return splits

    if mode == "cold-both":
        # Strict: test uses (did in D_fold) AND (pid in P_fold).
        # Train removes any pair containing those test dids or pids => no leakage.
        did_folds = _split_entities_to_folds(df["did"].unique(), n_splits=n_splits, seed=seed)
        pid_folds = _split_entities_to_folds(df["pid"].unique(), n_splits=n_splits, seed=seed + 999)

        for fold in range(1, n_splits + 1):
            test_dids = set(did_folds[fold - 1].tolist())
            test_pids = set(pid_folds[fold - 1].tolist())

            is_test = df["did"].isin(test_dids).values & df["pid"].isin(test_pids).values
            is_forbidden_for_train = df["did"].isin(test_dids).values | df["pid"].isin(test_pids).values

            test_idx = all_idx[is_test]
            trainval_idx = all_idx[~is_forbidden_for_train]

            train_idx, val_idx = _sample_val_from_train(trainval_idx, val_ratio_in_train, seed + fold)
            splits.append(FoldSplit(fold=fold, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
        return splits

    raise ValueError(f"Unknown split mode: {mode}")
