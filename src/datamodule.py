# src/datamodule.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import csv
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------- configs ----------------
@dataclass
class DMConfig:
    train_data: Tuple[List[str], List[str], List[float]]
    val_data:   Tuple[List[str], List[str], List[float]]
    test_data:  Tuple[List[str], List[str], List[float]]

    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False


@dataclass
class CacheDirs:
    esm_dir: str
    molclr_dir: str
    chemberta_dir: str
    pocket_dir: str


@dataclass
class CacheDims:
    esm2: int
    molclr: int
    chemberta: int


# ---------------- basic utils ----------------
def _sha1_24(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def _index_npz_dir(dir_path: Path) -> Dict[str, Path]:
    tbl: Dict[str, Path] = {}
    if not dir_path.exists():
        return tbl
    for p in dir_path.rglob("*.npz"):
        tbl[p.stem] = p
    return tbl


def _npz_load(p: Path) -> Dict[str, np.ndarray]:
    with np.load(str(p), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _npz_load_first_key(p: Optional[Path]) -> Optional[np.ndarray]:
    if p is None or (not p.exists()):
        return None
    with np.load(str(p), allow_pickle=False) as z:
        k = list(z.files)[0]
        return z[k]


def _as_2d(x: Optional[np.ndarray]) -> np.ndarray:
    if x is None:
        return np.zeros((0, 1), dtype=np.float32)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return np.asarray(x, dtype=np.float32, order="C")


def read_smiles_protein_label(path: str) -> Tuple[List[str], List[str], List[float]]:
    """
    Read CSV with columns:
      smiles, protein, label
    If header doesn't match, fallback to first 3 columns.
    """
    smiles, proteins, labels = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        si = pi = li = 0
        if header:
            heads = [c.strip().lower() for c in header]
            if all(x in heads for x in ("smiles", "protein", "label")):
                si, pi, li = heads.index("smiles"), heads.index("protein"), heads.index("label")
            else:
                si, pi, li = 0, 1, 2
                # header might be a data row
                if len(header) >= 3:
                    try:
                        smiles.append(header[si]); proteins.append(header[pi]); labels.append(float(header[li]))
                    except Exception:
                        pass

        for row in reader:
            if len(row) <= max(si, pi, li):
                continue
            try:
                smiles.append(row[si])
                proteins.append(row[pi])
                labels.append(float(row[li]))
            except Exception:
                continue
    return smiles, proteins, labels


# backward compatible name used by older train.py
_read_smiles_protein_label = read_smiles_protein_label


_AMINO = set("ACDEFGHIKLMNPQRSTVWY")
def _prot_variants(s: str) -> List[str]:
    raw = s or ""
    v = [raw, raw.strip(), "".join(raw.split()), "".join(raw.split()).upper()]
    only_aa = "".join(ch for ch in raw if ch.upper() in _AMINO)
    if only_aa:
        v += [only_aa, only_aa.upper()]
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _smiles_variants(s: str) -> List[str]:
    raw = s or ""
    v = [raw, raw.strip(), "".join(raw.split())]
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def _lookup(tbl: Dict[str, Path], keys: List[str]) -> Optional[Path]:
    for k in keys:
        p = tbl.get(_sha1_24(k))
        if p is not None:
            return p
    return None


# ---------------- pocket helpers ----------------
def _factorize_to_int(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1)
    mp: Dict[str, int] = {}
    out = np.empty((flat.shape[0],), dtype=np.int64)
    for i, x in enumerate(flat):
        s = str(x)
        if s not in mp:
            mp[s] = len(mp)
        out[i] = mp[s]
    return out.reshape(arr.shape)


def _pocket_to_tensors(npz: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Convert pocket npz to a dict of tensors.
    Keep only fields the model may need; skip strings except chain_id/res_idx (factorized).
    """
    keep = ["node_scalar_feat", "coords", "edge_index", "edge_scalar_feat", "res_idx", "chain_id"]
    out: Dict[str, torch.Tensor] = {}

    for k in keep:
        if k not in npz:
            continue

        arr = np.asarray(npz[k])

        # strings / objects
        if arr.dtype.kind in ("U", "S", "O"):
            if k in ("chain_id", "res_idx"):
                arr_i = _factorize_to_int(arr)
                out[k] = torch.as_tensor(arr_i, dtype=torch.long)
            continue

        if k in ("edge_index", "res_idx", "chain_id"):
            out[k] = torch.as_tensor(arr.astype(np.int64, copy=False), dtype=torch.long)
        else:
            out[k] = torch.as_tensor(arr.astype(np.float32, copy=False), dtype=torch.float32)

    # ensure always present
    out.setdefault("node_scalar_feat", torch.zeros((0, 21), dtype=torch.float32))
    out.setdefault("coords", torch.zeros((0, 3), dtype=torch.float32))
    out.setdefault("edge_index", torch.zeros((2, 0), dtype=torch.long))
    out.setdefault("edge_scalar_feat", torch.zeros((0, 1), dtype=torch.float32))
    out.setdefault("res_idx", torch.zeros((0,), dtype=torch.long))
    out.setdefault("chain_id", torch.zeros((0,), dtype=torch.long))
    return out


# ---------------- dataset ----------------
class DTIDataset(Dataset):
    """
    Sequence-only (vec removed). Pocket is always used.

    return per item:
      (P_seq, D_seq, C_vec, pocket_dict, y)
    """
    def __init__(
        self,
        data: Tuple[List[str], List[str], List[float]],
        esm_tbl: Dict[str, Path],
        molclr_tbl: Dict[str, Path],
        chem_tbl: Dict[str, Path],
        pocket_tbl: Dict[str, Path],
    ):
        self.smiles, self.proteins, self.labels = data
        self.esm_tbl = esm_tbl
        self.molclr_tbl = molclr_tbl
        self.chem_tbl = chem_tbl
        self.pocket_tbl = pocket_tbl

        # small caches reduce IO
        self._cache_esm: Dict[str, np.ndarray] = {}
        self._cache_mol: Dict[str, np.ndarray] = {}
        self._cache_chem: Dict[str, np.ndarray] = {}
        self._cache_pocket: Dict[str, Dict[str, torch.Tensor]] = {}

    def __len__(self):
        return len(self.labels)

    def _seq_cached(self, p: Optional[Path], cache: dict) -> np.ndarray:
        if p is None:
            return np.zeros((0, 1), dtype=np.float32)
        k = str(p)
        v = cache.get(k)
        if v is None:
            v = _as_2d(_npz_load_first_key(p))
            cache[k] = v
        return v

    def _pocket_cached(self, p: Optional[Path]) -> Dict[str, torch.Tensor]:
        if p is None:
            return _pocket_to_tensors({})
        k = str(p)
        d = self._cache_pocket.get(k)
        if d is None:
            d = _pocket_to_tensors(_npz_load(p))
            self._cache_pocket[k] = d
        return d

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        pro = self.proteins[idx]
        y = float(self.labels[idx])

        p_path = _lookup(self.esm_tbl, _prot_variants(pro))
        d1_path = _lookup(self.molclr_tbl, _smiles_variants(smi))
        c_path = _lookup(self.chem_tbl, _smiles_variants(smi))
        pk_path = _lookup(self.pocket_tbl, _prot_variants(pro))

        P = self._seq_cached(p_path, self._cache_esm)
        D = self._seq_cached(d1_path, self._cache_mol)
        C = self._seq_cached(c_path, self._cache_chem)
        if C.ndim == 2:
            C = C.mean(axis=0)  # (d_chem,)

        pocket = self._pocket_cached(pk_path)
        return P, D, C.astype(np.float32, copy=True), pocket, np.float32(y)


# ---------------- collate ----------------
def _pad_and_mask(batch_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = [x.shape[0] for x in batch_list]
    d = max([x.shape[1] for x in batch_list] + [1])
    tmax = max(lens + [1])
    B = len(batch_list)
    out = np.zeros((B, tmax, d), dtype=np.float32)
    mask = np.zeros((B, tmax), dtype=bool)
    for i, x in enumerate(batch_list):
        L = x.shape[0]
        out[i, :L, :x.shape[1]] = x
        mask[i, :L] = True
    return torch.tensor(out, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def collate_fn(batch):
    # (P, D, C, pocket, y)
    P_list = [x[0] for x in batch]
    D_list = [x[1] for x in batch]
    C_list = [x[2] for x in batch]
    pocket_list = [x[3] for x in batch]
    y_list = [x[4] for x in batch]

    P, Pm = _pad_and_mask(P_list)
    D, Dm = _pad_and_mask(D_list)
    C = torch.tensor(np.stack(C_list, axis=0), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_list, dtype=np.float32), dtype=torch.float32)
    return P, Pm, D, Dm, C, pocket_list, y


# ---------------- datamodule ----------------
class DataModule:
    def __init__(self, cfg: DMConfig, cache_dirs: CacheDirs, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose

        # esm/esm2 fallback
        cand = [Path(cache_dirs.esm_dir)]
        if "/esm2/" in cache_dirs.esm_dir:
            cand.append(Path(cache_dirs.esm_dir.replace("/esm2/", "/esm/")))
        elif "/esm/" in cache_dirs.esm_dir:
            cand.append(Path(cache_dirs.esm_dir.replace("/esm/", "/esm2/")))

        picked = None
        esm_tbl = {}
        for c in cand:
            tbl = _index_npz_dir(c)
            if tbl:
                picked, esm_tbl = c, tbl
                break
        if picked is None:
            picked = cand[0]
            esm_tbl = _index_npz_dir(picked)

        self.esm_dir_used = str(picked)
        self.esm_tbl = esm_tbl
        self.molclr_tbl = _index_npz_dir(Path(cache_dirs.molclr_dir))
        self.chem_tbl = _index_npz_dir(Path(cache_dirs.chemberta_dir))
        self.pocket_tbl = _index_npz_dir(Path(cache_dirs.pocket_dir))

        self._summary = self._build_summary()

    def _hit_rate(self, data: Tuple[List[str], List[str], List[float]], sample_cap: int = 2000) -> str:
        smi, pro, _ = data
        n = len(smi)
        if n == 0:
            return "NA"
        idxs = list(range(n))
        if n > sample_cap:
            random.seed(42)
            idxs = random.sample(idxs, sample_cap)

        hit_all = 0
        for i in idxs:
            p = _lookup(self.esm_tbl, _prot_variants(pro[i]))
            d = _lookup(self.molclr_tbl, _smiles_variants(smi[i]))
            c = _lookup(self.chem_tbl, _smiles_variants(smi[i]))
            pk = _lookup(self.pocket_tbl, _prot_variants(pro[i]))
            ok = (p is not None) and (d is not None) and (c is not None) and (pk is not None)
            hit_all += int(ok)

        return f"{hit_all/len(idxs)*100:.1f}%"

    def _uniq_counts(self, data: Tuple[List[str], List[str], List[float]]) -> Tuple[int, int]:
        smi, pro, _ = data
        return len(set(smi)), len(set(pro))

    def _build_summary(self) -> Dict[str, Any]:
        return {
            "esm_dir_used": self.esm_dir_used,
            "index_size": {
                "esm": len(self.esm_tbl),
                "molclr": len(self.molclr_tbl),
                "chemberta": len(self.chem_tbl),
                "pocket": len(self.pocket_tbl),
            },
            "hit_rate": {
                "train": self._hit_rate(self.cfg.train_data),
                "val": self._hit_rate(self.cfg.val_data),
                "test": self._hit_rate(self.cfg.test_data),
            },
            "uniq": {
                "train": self._uniq_counts(self.cfg.train_data),
                "val": self._uniq_counts(self.cfg.val_data),
                "test": self._uniq_counts(self.cfg.test_data),
            },
        }

    def summary(self) -> Dict[str, Any]:
        return self._summary

    def _make_loader(self, data, shuffle: bool) -> DataLoader:
        ds = DTIDataset(
            data=data,
            esm_tbl=self.esm_tbl,
            molclr_tbl=self.molclr_tbl,
            chem_tbl=self.chem_tbl,
            pocket_tbl=self.pocket_tbl,
        )

        kwargs = dict(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=(self.cfg.drop_last if shuffle else False),
            collate_fn=collate_fn,
        )

        # DataLoader: num_workers=0 cannot use persistent_workers/prefetch_factor
        if self.cfg.num_workers > 0:
            kwargs["persistent_workers"] = bool(self.cfg.persistent_workers)
            kwargs["prefetch_factor"] = int(self.cfg.prefetch_factor)

        return DataLoader(**kwargs)

    def train_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.train_data, shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.val_data, shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.test_data, shuffle=False)
