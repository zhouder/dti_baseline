# -*- coding: utf-8 -*-
"""MRBDTA (2022) reproduction runner with MolTrans-style CV splits + binary classification.

What this script does
---------------------
* Reads pairwise DTI CSV:  /root/lanyun-fs/{dataset}/{dataset}.csv
* Auto-detects columns (SMILES, protein sequence, label).
* Creates MolTrans-style outer CV splits with strict warm/cold modes:
    - warm (random stratified)
    - cold-drug (grouped by SMILES)
    - cold-protein (grouped by protein sequence)
  (Also supports cold-pair, cold-both for convenience.)
* Trains MRBDTA architecture (Trans-block w/ skip connections) but swaps the
  regression objective for binary classification using BCEWithLogitsLoss.
* Logs per-epoch metrics like your MolTrans runner:
    auroc/auprc/f1/acc/sen/mcc + threshold chosen on val
* Early stopping on AUPRC (fallback AUROC) with --patience.

Notes on "repro"
----------------
The original MRBDTA codebase is written as a single script (MRBDTA.py) and
targets affinity regression. This runner keeps the architecture and its
hyperparameter defaults (e.g., davis: 300 epochs; kiba: 600 epochs; batch 32;
accumulation steps per their README) but:
  1) makes it importable/usable with your CSV pairs;
  2) replaces MSELoss with BCEWithLogitsLoss;
  3) removes hard-coded .cuda() calls inside LayerNorm, so device handling is safe.

Author: generated for zhou der
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
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm


# =========================
# Utils (MolTrans-compatible)
# =========================


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    low2col = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low2col:
            return low2col[c.lower()]
    return None


def load_csv_any_schema(csv_path: Path) -> pd.DataFrame:
    """Load a CSV and normalize schema to [SMILES, Target Sequence, Label]."""
    df = pd.read_csv(csv_path)

    c_smiles = _find_col(df, ["smile", "smiles", "drug", "compound", "ligand"])
    c_prot = _find_col(df, ["seq", "sequence", "protein", "target", "target sequence"])
    c_y = _find_col(df, ["label", "y"])

    if c_smiles is None or c_prot is None or c_y is None:
        raise ValueError(
            "CSV must contain columns like (smile/smiles) + (seq/protein) + (label/y). "
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


# =========================
# MRBDTA tokenizer (DataHelper-like, but robust)
# =========================


TARGET_VOCAB = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
TARGET_VOCAB_SIZE = 25


DRUG_VOCAB = {
    "#": 1,
    "%": 2,
    ")": 3,
    "(": 4,
    "+": 5,
    "-": 6,
    ".": 7,
    "1": 8,
    "0": 9,
    "3": 10,
    "2": 11,
    "5": 12,
    "4": 13,
    "7": 14,
    "6": 15,
    "9": 16,
    "8": 17,
    "=": 18,
    "A": 19,
    "C": 20,
    "B": 21,
    "E": 22,
    "D": 23,
    "G": 24,
    "F": 25,
    "I": 26,
    "H": 27,
    "K": 28,
    "M": 29,
    "L": 30,
    "O": 31,
    "N": 32,
    "P": 33,
    "S": 34,
    "R": 35,
    "U": 36,
    "T": 37,
    "W": 38,
    "V": 39,
    "Y": 40,
    "[": 41,
    "Z": 42,
    "]": 43,
    "_": 44,
    "a": 45,
    "c": 46,
    "b": 47,
    "e": 48,
    "d": 49,
    "g": 50,
    "f": 51,
    "i": 52,
    "h": 53,
    "m": 54,
    "l": 55,
    "o": 56,
    "n": 57,
    "s": 58,
    "r": 59,
    "u": 60,
    "t": 61,
    "y": 62,
}
DRUG_VOCAB_SIZE = 62


def _encode_string_to_ids(s: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    """Character-level encoding. Unknown chars -> 0 (PAD). Returns (ids, n_unk)."""
    ids: List[int] = []
    n_unk = 0
    for ch in s[:max_len]:
        v = vocab.get(ch, 0)
        if v == 0:
            n_unk += 1
        ids.append(v)
    return ids, n_unk


def build_feature_cache(
    df_all: pd.DataFrame,
    drug_maxlen: int,
    prot_maxlen: int,
    cache_path: Path,
) -> Dict[str, torch.Tensor]:
    """Encode all rows once. Stores padded int tensors (N, Ld)/(N, Lp) + labels."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        obj = torch.load(str(cache_path), map_location="cpu")
        if (
            obj.get("drug_maxlen") == drug_maxlen
            and obj.get("prot_maxlen") == prot_maxlen
            and int(obj.get("n", -1)) == len(df_all)
        ):
            print(f"[Preprocess] loaded cache: {cache_path}")
            return obj

    print("[Preprocess] building feature cache once (char vocab + fixed pad)...")
    smiles = df_all["SMILES"].tolist()
    prots = df_all["Target Sequence"].tolist()
    y = df_all["Label"].astype(np.float32).to_numpy()

    # encode unique to speed up
    smi_uniq = sorted(set(smiles))
    prot_uniq = sorted(set(prots))
    smi_map: Dict[str, List[int]] = {}
    prot_map: Dict[str, List[int]] = {}
    smi_unk = 0
    prot_unk = 0

    for s in tqdm(smi_uniq, desc="[Preprocess] SMILES", ncols=110):
        ids, n_unk = _encode_string_to_ids(s, DRUG_VOCAB, drug_maxlen)
        smi_map[s] = ids
        smi_unk += n_unk

    for p in tqdm(prot_uniq, desc="[Preprocess] PROT", ncols=110):
        ids, n_unk = _encode_string_to_ids(p, TARGET_VOCAB, prot_maxlen)
        prot_map[p] = ids
        prot_unk += n_unk

    N = len(df_all)
    drug_ids = torch.zeros((N, drug_maxlen), dtype=torch.long)
    prot_ids = torch.zeros((N, prot_maxlen), dtype=torch.long)

    for i, (s, p) in enumerate(zip(smiles, prots)):
        ds = smi_map[s]
        ps = prot_map[p]
        drug_ids[i, : len(ds)] = torch.as_tensor(ds, dtype=torch.long)
        prot_ids[i, : len(ps)] = torch.as_tensor(ps, dtype=torch.long)

    obj = {
        "drug": drug_ids,
        "prot": prot_ids,
        "y": torch.as_tensor(y, dtype=torch.float32),
        "drug_maxlen": drug_maxlen,
        "prot_maxlen": prot_maxlen,
        "n": N,
        "smi_unk_chars": int(smi_unk),
        "prot_unk_chars": int(prot_unk),
        "n_unique_smiles": int(len(smi_uniq)),
        "n_unique_prot": int(len(prot_uniq)),
    }
    torch.save(obj, str(cache_path))
    print(
        f"[Preprocess] saved cache: {cache_path} | samples={N} | "
        f"uniq_smiles={obj['n_unique_smiles']} uniq_prot={obj['n_unique_prot']} | "
        f"unk_chars(smiles)={obj['smi_unk_chars']} unk_chars(prot)={obj['prot_unk_chars']}"
    )
    return obj


# =========================
# MRBDTA model (same structure, but device-safe LayerNorm + batch-safe sums)
# =========================


# Transformer Parameters (same as original MRBDTA.py)
d_model = 128
d_ff = 512
d_k = d_v = 32
n_layers = 1
n_heads = 4


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    # seq_q=seq_k: [B, L]
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [B, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [B, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor):
        # Q: [B, H, Lq, d_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores = scores.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_Q: torch.Tensor, input_K: torch.Tensor, input_V: torch.Tensor, attn_mask: torch.Tensor):
        batch_size, seq_len, model_len = input_Q.size()
        residual_2d = input_Q.reshape(batch_size * seq_len, model_len)
        residual = self.fc0(residual_2d).reshape(batch_size, seq_len, model_len)

        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.ln(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs: torch.Tensor, enc_self_attn_mask: torch.Tensor):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size + 1, d_model)  # +1 keeps PAD=0 safe
        self.pos_emb = PositionalEncoding(d_model)
        self.stream0 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream1 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.stream2 = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs: torch.Tensor):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        stream0 = enc_outputs

        enc_self_attns0, enc_self_attns1, enc_self_attns2 = [], [], []
        for layer in self.stream0:
            stream0, enc_self_attn0 = layer(stream0, enc_self_attn_mask)
            enc_self_attns0.append(enc_self_attn0)

        stream1 = stream0 + enc_outputs
        stream2 = stream0 + enc_outputs
        for layer in self.stream1:
            stream1, enc_self_attn1 = layer(stream1, enc_self_attn_mask)
            enc_self_attns1.append(enc_self_attn1)

        for layer in self.stream2:
            stream2, enc_self_attn2 = layer(stream2, enc_self_attn_mask)
            enc_self_attns2.append(enc_self_attn2)

        return torch.cat((stream1, stream2), 2), enc_self_attns0, enc_self_attns1, enc_self_attns2


class MRBDTA_Transformer(nn.Module):
    """Same head dims as original MRBDTA. Output is a single logit."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.encoderD = Encoder(DRUG_VOCAB_SIZE)
        self.encoderT = Encoder(TARGET_VOCAB_SIZE)

        self.fc0 = nn.Sequential(
            nn.Linear(4 * d_model, 16 * d_model, bias=False),
            nn.LayerNorm(16 * d_model),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * d_model, 4 * d_model, bias=False),
            nn.LayerNorm(4 * d_model),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(4 * d_model, 1, bias=False)

    def forward(self, input_drugs: torch.Tensor, input_tars: torch.Tensor):
        enc_drugs, attnD0, attnD1, attnD2 = self.encoderD(input_drugs)
        enc_tars, attnT0, attnT1, attnT2 = self.encoderT(input_tars)

        # batch-safe aggregation (no squeeze)
        enc_drugs_2d = torch.sum(enc_drugs, dim=1)  # [B, 2*d_model]
        enc_tars_2d = torch.sum(enc_tars, dim=1)  # [B, 2*d_model]
        fc = torch.cat((enc_drugs_2d, enc_tars_2d), dim=1)  # [B, 4*d_model]

        x = self.fc0(fc)
        x = self.fc1(x)
        logit = self.fc2(x).view(-1)  # [B]
        return logit, attnD0, attnT0, attnD1, attnT1, attnD2, attnT2


# =========================
# Dataset / loaders
# =========================


class PairDataset(Data.Dataset):
    def __init__(self, drug_ids: torch.Tensor, prot_ids: torch.Tensor, y: torch.Tensor, indices: np.ndarray):
        self.drug = drug_ids
        self.prot = prot_ids
        self.y = y
        self.idx = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        return self.drug[j], self.prot[j], self.y[j]


def collate_fixed(batch, pad=0):
    drugs, prots, y = list(zip(*batch))
    drugs = torch.stack(drugs, dim=0)
    prots = torch.stack(prots, dim=0)
    y = torch.stack(y, dim=0).float().view(-1)
    return drugs, prots, y


# =========================
# Train / eval
# =========================


@torch.no_grad()
def evaluate(model: nn.Module, loader: Data.DataLoader, device: torch.device, desc: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(False)
    loss_f = nn.BCEWithLogitsLoss(reduction="mean")
    loss_sum = 0.0
    n_sum = 0
    probs, labels = [], []

    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for drugs, prots, y in pbar:
        drugs = drugs.to(device)
        prots = prots.to(device)
        y = y.to(device)

        logits, *_ = model(drugs, prots)
        logits = logits.view(-1)
        y = y.view(-1)
        n = min(logits.numel(), y.numel())
        if n == 0:
            continue

        loss = loss_f(logits[:n], y[:n])
        bs = int(n)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs

        prob = torch.sigmoid(logits[:n])
        probs.append(prob.detach().float().cpu().numpy())
        labels.append(y[:n].detach().float().cpu().numpy())

    pbar.close()
    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    y_all = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.float32)
    return loss_sum / max(1, n_sum), prob_all, y_all


def train_one_epoch(
    model: nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    accumulation_steps: int,
    grad_clip: float,
    desc: str,
) -> float:
    model.train(True)
    loss_f = nn.BCEWithLogitsLoss(reduction="mean")
    loss_sum = 0.0
    n_sum = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for step, (drugs, prots, y) in enumerate(pbar, start=1):
        drugs = drugs.to(device)
        prots = prots.to(device)
        y = y.to(device).view(-1)

        logits, *_ = model(drugs, prots)
        logits = logits.view(-1)
        n = min(logits.numel(), y.numel())
        if n == 0:
            continue

        loss = loss_f(logits[:n], y[:n])
        (loss / max(1, accumulation_steps)).backward()

        if step % max(1, accumulation_steps) == 0:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = int(n)
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs
        pbar.set_postfix_str(f"loss={float(loss.detach().cpu()):.4f}")

    # flush remainder
    if len(loader) % max(1, accumulation_steps) != 0:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    pbar.close()
    return loss_sum / max(1, n_sum)


# =========================
# CLI / main
# =========================


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. davis / kiba / drugbank (folder name under data-root)")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-tmp/mrbdta-cls-runs")
    ap.add_argument("--cache-root", type=str, default="/root/lanyun-tmp/mrbdta-cls-cache")

    ap.add_argument(
        "--split-mode",
        choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
        default="warm",
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--overall-val", type=float, default=0.10)

    # Repro-ish defaults (from MRBDTA README):
    #  - kiba: EPOCHS, batch_size, accumulation_steps = 600, 32, 32
    #  - davis: EPOCHS, batch_size, accumulation_steps = 300, 32, 8
    ap.add_argument("--epochs", type=int, default=0, help="0 means use MRBDTA's dataset-specific default")
    ap.add_argument("--batch-size", type=int, default=0, help="0 means use MRBDTA's default (32)")
    ap.add_argument("--accumulation-steps", type=int, default=0, help="0 means use MRBDTA's dataset default")

    ap.add_argument("--drug-maxlen", type=int, default=0, help="0 means dataset default (davis=85,kiba=100,drugbank=100)")
    ap.add_argument("--prot-maxlen", type=int, default=0, help="0 means dataset default (davis=1200,kiba=1000,drugbank=1000)")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume fold from fold_dir/last.pt if exists")
    return ap.parse_args()


def _defaults_by_dataset(ds: str) -> Dict[str, int]:
    ds = ds.lower()
    if ds == "davis":
        return {"epochs": 300, "batch": 32, "accum": 8, "drug_maxlen": 85, "prot_maxlen": 1200}
    if ds == "kiba":
        return {"epochs": 600, "batch": 32, "accum": 32, "drug_maxlen": 100, "prot_maxlen": 1000}
    # For drugbank (classification), no official MRBDTA setting. Use kiba-like lengths + davis-like accum.
    return {"epochs": 300, "batch": 32, "accum": 8, "drug_maxlen": 100, "prot_maxlen": 1000}


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    args = parse_args()
    set_seed(int(args.seed))

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

    dflt = _defaults_by_dataset(ds)
    epochs = int(args.epochs) if int(args.epochs) > 0 else int(dflt["epochs"])
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(dflt["batch"])
    accumulation_steps = int(args.accumulation_steps) if int(args.accumulation_steps) > 0 else int(dflt["accum"])
    drug_maxlen = int(args.drug_maxlen) if int(args.drug_maxlen) > 0 else int(dflt["drug_maxlen"])
    prot_maxlen = int(args.prot_maxlen) if int(args.prot_maxlen) > 0 else int(dflt["prot_maxlen"])

    # Build/preload cache once for the whole dataset.
    cache_path = Path(args.cache_root) / f"{ds}_Ld{drug_maxlen}_Lp{prot_maxlen}.pt"
    cache = build_feature_cache(df_all, drug_maxlen, prot_maxlen, cache_path)
    drug_ids: torch.Tensor = cache["drug"]
    prot_ids: torch.Tensor = cache["prot"]
    y_all: torch.Tensor = cache["y"]

    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer = make_outer_splits(split_mode, int(args.cv_folds), int(args.seed), drug_key, prot_key, labels)
    K = len(outer)
    overall_test = 1.0 / K
    val_frac_in_pool = float(args.overall_val) / (1.0 - overall_test)
    print(
        f"[SPLIT] train/val/test target = {1.0 - overall_test - args.overall_val:.2f}/{args.overall_val:.2f}/{overall_test:.2f} "
        f"| K={K} | val_frac_in_pool={val_frac_in_pool:.4f}"
    )

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
            seed=int(args.seed) + 100 + fold,
            drug_key=drug_key,
            prot_key=prot_key,
            labels=labels,
        )

        print("=" * 80)
        print(f"[Fold {fold}/{K}] train_pool={len(train_pool_idx)} test={len(test_idx)} mode={split_mode}")
        summarize_split("train", tr_idx, drug_key, prot_key, labels)
        summarize_split("val", va_idx, drug_key, prot_key, labels)
        summarize_split("test", test_idx, drug_key, prot_key, labels)

        dset_tr = PairDataset(drug_ids, prot_ids, y_all, tr_idx)
        dset_va = PairDataset(drug_ids, prot_ids, y_all, va_idx)
        dset_te = PairDataset(drug_ids, prot_ids, y_all, test_idx)

        dl_train = Data.DataLoader(
            dset_tr,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
            collate_fn=collate_fixed,
        )
        dl_eval = dict(
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
            collate_fn=collate_fixed,
        )
        dl_val = Data.DataLoader(dset_va, **dl_eval)
        dl_test = Data.DataLoader(dset_te, **dl_eval)

        model = MRBDTA_Transformer(dropout=float(args.dropout)).to(device)
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            model = nn.DataParallel(model, dim=0)

        n_params = _count_params(model.module if isinstance(model, nn.DataParallel) else model)
        print(
            f"[Model] MRBDTA_Transformer | params={n_params:,} | epochs={epochs} | bs={batch_size} | "
            f"accum={accumulation_steps} | lr={args.lr} | Ld={drug_maxlen} Lp={prot_maxlen}"
        )

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
                w.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_auroc",
                        "val_auprc",
                        "val_f1",
                        "val_acc",
                        "val_sen",
                        "val_mcc",
                        "val_thr",
                        "time_sec",
                        "is_best",
                    ]
                )

        start_ep = 1
        best_score = -1.0
        best_thr = 0.5
        best_epoch = 0
        no_improve = 0

        if args.resume and last_pt.exists():
            ck = torch.load(str(last_pt), map_location="cpu")
            state_dict = ck["state_dict"]
            (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state_dict, strict=True)
            optimizer.load_state_dict(ck["optimizer"])
            start_ep = int(ck.get("epoch", 0)) + 1
            best_score = float(ck.get("best_score", -1.0))
            best_thr = float(ck.get("best_thr", 0.5))
            best_epoch = int(ck.get("best_epoch", 0))
            no_improve = int(ck.get("no_improve", 0))
            print(
                f"[RESUME] start_ep={start_ep} best_score={best_score:.6f} best_epoch={best_epoch} no_improve={no_improve}"
            )

        for ep in range(start_ep, epochs + 1):
            t0 = time.time()
            tr_loss = train_one_epoch(
                model,
                dl_train,
                device,
                optimizer,
                accumulation_steps=accumulation_steps,
                grad_clip=float(args.grad_clip),
                desc=f"{ds}/fold{fold} train ep{ep}",
            )
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
                        "args": vars(args),
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
                    "args": vars(args),
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
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(ck["state_dict"], strict=True)

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
