"""
datamodule.py

Lightweight data loading for UGCA-DTI:

- reads `{dataset}.csv`
- computes did/pid hash IDs
- loads 4 offline encodings:
    molclr/{did}.npy
    chemberta/{did}.npy
    esm2/{pid}.npz
    pocket_graph/{pid}.npz

Directory layout and naming rules follow your feature document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .splits import add_hash_ids, sanitize_label_series
except ImportError:
    try:
        from src.splits import add_hash_ids, sanitize_label_series
    except ImportError:
        from splits import add_hash_ids, sanitize_label_series


# ----------------------------
# Feature loading helpers
# ----------------------------

ESM2_KEY_PRIORITY = ("emb", "repr", "x", "feat")


def _np_to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float16 or x.dtype == np.float32:
        return x.astype(np.float32, copy=False)
    return x.astype(np.float32)


def load_npy_vector(path: Path) -> np.ndarray:
    """Load .npy feature and return a fixed-length vector.

    Some feature generators save token/atom-level matrices (L, d). To keep the
    whole model lightweight and batchable, we apply mean pooling over the first
    dimension whenever the loaded array is not 1-D.
    """
    x = np.load(str(path), allow_pickle=False)
    x = np.asarray(x)

    # If (L, d) or (..., d): pool over token/atom dimension(s)
    if x.ndim >= 2:
        # treat the last dim as feature dim
        x = x.reshape(-1, x.shape[-1]).mean(axis=0)

    if x.ndim != 1:
        x = x.reshape(-1)
    return _np_to_float32(x)


def load_esm2_vector(path: Path) -> np.ndarray:
    npz = np.load(str(path), allow_pickle=False)
    key = None
    for k in ESM2_KEY_PRIORITY:
        if k in npz.files:
            key = k
            break
    if key is None:
        if len(npz.files) == 0:
            raise ValueError(f"Empty ESM2 npz: {path}")
        key = npz.files[0]
    x = npz[key]
    x = np.asarray(x)
    x = _np_to_float32(x)
    if x.ndim == 2:
        # (L, 1280) -> mean pool
        x = x.mean(axis=0)
    elif x.ndim != 1:
        x = x.reshape(-1)
    return x


def pocket_npz_to_vector(path: Path, use_edge: bool = True) -> np.ndarray:
    """
    Convert pocket_graph GVP feature .npz to a single global vector.

    Required keys (per your spec):
      node_s: (N, ds)
      node_v: (N, dv, 3)
      edge_index: (2, E)
      edge_s: (E, des)
      edge_v: (E, dev, 3)
      res_idx: (N,)

    We keep it lightweight by using simple statistics pooling:
      - node: mean(node_s) + mean(norm(node_v))
      - edge: mean(edge_s) + mean(norm(edge_v)) (optional)

    This avoids running a deep GVP-GNN while still using geometry signals.
    """
    npz = np.load(str(path), allow_pickle=False)

    # Node features
    node_s = np.asarray(npz["node_s"])
    node_v = np.asarray(npz["node_v"])
    node_s = _np_to_float32(node_s)
    node_v = _np_to_float32(node_v)

    if node_v.ndim != 3 or node_v.shape[-1] != 3:
        raise ValueError(f"node_v must have shape (N, dv, 3). Got {node_v.shape} in {path}")

    node_v_norm = np.linalg.norm(node_v, axis=-1)  # (N, dv)
    node_feat = np.concatenate([node_s, node_v_norm], axis=-1)  # (N, ds+dv)
    node_mean = node_feat.mean(axis=0)  # (ds+dv,)

    if not use_edge:
        return _np_to_float32(node_mean)

    # Edge features (optional)
    edge_s = np.asarray(npz["edge_s"])
    edge_v = np.asarray(npz["edge_v"])
    edge_s = _np_to_float32(edge_s)
    edge_v = _np_to_float32(edge_v)
    if edge_v.ndim != 3 or edge_v.shape[-1] != 3:
        raise ValueError(f"edge_v must have shape (E, dev, 3). Got {edge_v.shape} in {path}")

    edge_v_norm = np.linalg.norm(edge_v, axis=-1)  # (E, dev)
    edge_feat = np.concatenate([edge_s, edge_v_norm], axis=-1)  # (E, des+dev)
    edge_mean = edge_feat.mean(axis=0)  # (des+dev,)

    vec = np.concatenate([node_mean, edge_mean], axis=-1)
    return _np_to_float32(vec)


# ----------------------------
# Dataset
# ----------------------------

class DTIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_dirs: Dict[str, Path],
        pocket_use_edge: bool = True,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_dirs = feature_dirs
        self.pocket_use_edge = pocket_use_edge

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        did = row["did"]
        pid = row["pid"]

        molclr = load_npy_vector(self.feature_dirs["molclr"] / f"{did}.npy")
        chemberta = load_npy_vector(self.feature_dirs["chemberta"] / f"{did}.npy")
        esm2 = load_esm2_vector(self.feature_dirs["esm2"] / f"{pid}.npz")
        pocket = pocket_npz_to_vector(self.feature_dirs["pocket_graph"] / f"{pid}.npz", use_edge=self.pocket_use_edge)

        y = float(row["label"])

        return {
            "molclr": torch.from_numpy(molclr),
            "chemberta": torch.from_numpy(chemberta),
            "esm2": torch.from_numpy(esm2),
            "pocket": torch.from_numpy(pocket),
            "label": torch.tensor(y, dtype=torch.float32),
        }


def dti_collate_fn(batch):
    # batch: list[dict]
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "label":
            out[k] = torch.stack([b[k] for b in batch], dim=0).view(-1, 1)
        else:
            shapes = [tuple(b[k].shape) for b in batch]
            if len(set(shapes)) != 1:
                raise RuntimeError(f"Collate shape mismatch for '{k}': {shapes}. "
                                   "This usually means the stored feature is token/atom-level without pooling.")
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ----------------------------
# DataModule
# ----------------------------

@dataclass
class DataPaths:
    dataset: str
    root_dir: Path = Path("/root/lanyun-fs")

    @property
    def dataset_dir(self) -> Path:
        return self.root_dir / self.dataset.lower()

    @property
    def csv_path(self) -> Path:
        return self.dataset_dir / f"{self.dataset.lower()}.csv"

    @property
    def feature_dirs(self) -> Dict[str, Path]:
        return {
            "molclr": self.dataset_dir / "molclr",
            "chemberta": self.dataset_dir / "chemberta",
            "esm2": self.dataset_dir / "esm2",
            "pocket_graph": self.dataset_dir / "pocket_graph",
        }


class DTIDataModule:
    def __init__(
        self,
        dataset: str,
        train_idx,
        val_idx,
        test_idx,
        batch_size: int = 256,
        num_workers: int = 4,
        allow_missing: bool = False,
        pocket_use_edge: bool = True,
        root_dir: str = "/root/lanyun-fs",
    ) -> None:
        self.dataset = dataset
        self.paths = DataPaths(dataset=dataset, root_dir=Path(root_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_missing = allow_missing
        self.pocket_use_edge = pocket_use_edge

        self.train_idx = np.asarray(train_idx, dtype=int)
        self.val_idx = np.asarray(val_idx, dtype=int)
        self.test_idx = np.asarray(test_idx, dtype=int)

        self.df_all: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        self._id_sets = {}

        self.input_dims: Optional[Dict[str, int]] = None

    def load_dataframe(self) -> pd.DataFrame:
        if not self.paths.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.paths.csv_path}")
        df = pd.read_csv(self.paths.csv_path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column for classification.")
        df = add_hash_ids(df)
        df["label"] = sanitize_label_series(df["label"])
        return df

    def _build_feature_id_sets(self) -> None:
        """Scan feature directories to build {modality: set(ids)} for fast hit-check."""
        dirs = self.paths.feature_dirs
        id_sets = {}
        for mod, d in dirs.items():
            if not d.exists():
                raise FileNotFoundError(f"Feature directory not found: {d}")
            if mod in ("molclr", "chemberta"):
                exts = (".npy",)
            else:
                exts = (".npz",)
            ids = set()
            for fn in os.listdir(d):
                if fn.endswith(exts):
                    ids.add(Path(fn).stem)
            id_sets[mod] = ids
        self._id_sets = id_sets

    def _feature_hit_stats(self, df_split: pd.DataFrame) -> Dict[str, int]:
        """Return missing counts for each modality in this split (pair-level)."""
        miss = {}
        dids = df_split["did"].astype(str)
        pids = df_split["pid"].astype(str)

        miss["molclr"] = int((~dids.isin(self._id_sets["molclr"])).sum())
        miss["chemberta"] = int((~dids.isin(self._id_sets["chemberta"])).sum())
        miss["esm2"] = int((~pids.isin(self._id_sets["esm2"])).sum())
        miss["pocket_graph"] = int((~pids.isin(self._id_sets["pocket_graph"])).sum())
        return miss

    def _filter_valid_pairs(self, df_split: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        """Filter out rows where any modality feature file is missing."""
        dids = df_split["did"].astype(str)
        pids = df_split["pid"].astype(str)
        ok = (
            dids.isin(self._id_sets["molclr"])
            & dids.isin(self._id_sets["chemberta"])
            & pids.isin(self._id_sets["esm2"])
            & pids.isin(self._id_sets["pocket_graph"])
        )
        miss = self._feature_hit_stats(df_split)
        dropped = int((~ok).sum())
        return df_split[ok].reset_index(drop=True), miss, dropped

    def setup(self, verbose: bool = True) -> None:
        self.df_all = self.load_dataframe()
        if len(self.df_all) == 0:
            raise ValueError("Empty dataset CSV.")

        self._build_feature_id_sets()

        df_train = self.df_all.iloc[self.train_idx].copy()
        df_val = self.df_all.iloc[self.val_idx].copy()
        df_test = self.df_all.iloc[self.test_idx].copy()

        # Check / filter missing features
        df_train_f, miss_train, drop_train = self._filter_valid_pairs(df_train)
        df_val_f, miss_val, drop_val = self._filter_valid_pairs(df_val)
        df_test_f, miss_test, drop_test = self._filter_valid_pairs(df_test)


        if verbose:
            total_pairs = len(df_train) + len(df_val) + len(df_test)
            after_pairs = len(df_train_f) + len(df_val_f) + len(df_test_f)

            # overall (pair-level) hit rates
            miss_total = {
                "molclr": miss_train["molclr"] + miss_val["molclr"] + miss_test["molclr"],
                "chemberta": miss_train["chemberta"] + miss_val["chemberta"] + miss_test["chemberta"],
                "esm2": miss_train["esm2"] + miss_val["esm2"] + miss_test["esm2"],
                "pocket_graph": miss_train["pocket_graph"] + miss_val["pocket_graph"] + miss_test["pocket_graph"],
            }
            # pairs missing ANY modality
            dropped_total = drop_train + drop_val + drop_test

            def _rate(miss_cnt: int) -> float:
                return 0.0 if total_pairs == 0 else (1.0 - miss_cnt / total_pairs) * 100.0

            print(f"Split sizes (pairs): train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
            print(f"Feature hit-rate (overall): "
                  f"molclr={_rate(miss_total['molclr']):.2f}%, "
                  f"chemberta={_rate(miss_total['chemberta']):.2f}%, "
                  f"esm2={_rate(miss_total['esm2']):.2f}%, "
                  f"pocket={_rate(miss_total['pocket_graph']):.2f}%  "
                  f"(dropped_any={dropped_total}/{total_pairs})")
            print(f"All four encodings hit for this fold?  {dropped_total == 0}")

        if (drop_train + drop_val + drop_test) > 0 and not self.allow_missing:
            raise RuntimeError(
                "Some feature files are missing. "
                "Re-run with --allow_missing to drop those pairs, or fix the feature directories."
            )

        self.df_train, self.df_val, self.df_test = df_train_f, df_val_f, df_test_f

        # Infer input dims from one training example
        if len(self.df_train) == 0:
            raise RuntimeError("No training pairs left after feature checking.")
        sample = DTIDataset(self.df_train.iloc[:1], self.paths.feature_dirs, pocket_use_edge=self.pocket_use_edge)[0]
        self.input_dims = {
            "molclr": int(sample["molclr"].numel()),
            "chemberta": int(sample["chemberta"].numel()),
            "esm2": int(sample["esm2"].numel()),
            "pocket": int(sample["pocket"].numel()),
        }

    def get_dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be in {'train','val','test'}")
        if self.df_train is None:
            raise RuntimeError("Call setup() first.")
        df = {"train": self.df_train, "val": self.df_val, "test": self.df_test}[split]
        dataset = DTIDataset(df, self.paths.feature_dirs, pocket_use_edge=self.pocket_use_edge)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dti_collate_fn,
            drop_last=False,
        )
