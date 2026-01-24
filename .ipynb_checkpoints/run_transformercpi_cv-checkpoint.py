# -*- coding: utf-8 -*-
"""Run TransformerCPI on a CSV with CV + warm/cold splits.

- No gensim needed (protein 3-mer embedding is learned end-to-end).
- RDKit is used to compute 34-d atom features + adjacency matrices.

CSV must have columns for: SMILES, Target Sequence, Label (0/1).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# RDKit for compound features
from rdkit import Chem


# -----------------------------
# Utils
# -----------------------------

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

    # binarize
    y = df["Label"].values
    uniq = np.unique(y)
    if len(uniq) > 2:
        raise ValueError(
            f"TransformerCPI in this script is for binary classification. "
            f"Found {len(uniq)} unique labels (e.g. {uniq[:10]})."
        )
    df["Label"] = (df["Label"].astype(np.float32) >= 0.5).astype(np.int64)
    return df


# -----------------------------
# Splits (same logic as your MolTrans runner)
# -----------------------------

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
            # fallback
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


# -----------------------------
# RDKit compound features (same recipe as repo)
# -----------------------------

NUM_ATOM_FEAT = 34


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H: bool = False, use_chirality: bool = True):
    symbol = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        "other",
    ]

    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbol)
        + one_of_k_encoding_unk(atom.GetDegree(), degree)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType)
        + [atom.GetIsAromatic()]
    )  # 26

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 31

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [
                atom.HasProp("_ChiralityPossible")
            ]
        except Exception:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]

    return results  # 34


def mol_features(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError("SMILES cannot be parsed")

    atom_feat = np.zeros((mol.GetNumAtoms(), NUM_ATOM_FEAT), dtype=np.float32)
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = np.asarray(atom_features(atom), dtype=np.float32)

    adjacency = Chem.GetAdjacencyMatrix(mol)
    adj = np.asarray(adjacency, dtype=np.float32)
    return atom_feat, adj


# -----------------------------
# Protein k-mer vocab
# -----------------------------


def seq_to_kmers(seq: str, k: int = 3) -> List[str]:
    seq = str(seq)
    n = len(seq)
    if n < k:
        return []
    return [seq[i : i + k] for i in range(n - k + 1)]


def build_kmer_vocab(seqs: List[str], k: int = 3, min_freq: int = 1) -> Dict[str, int]:
    c = Counter()
    for s in seqs:
        c.update(seq_to_kmers(s, k=k))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in c.items():
        if freq >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def kmers_to_ids(kmers: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("<UNK>", 1)
    return [vocab.get(km, unk) for km in kmers]


# -----------------------------
# Dataset + collate
# -----------------------------


class CPIDataset(Dataset):
    def __init__(
        self,
        smiles: List[str],
        seqs: List[str],
        labels: List[int],
        mol_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
        vocab: Dict[str, int],
        kmer: int = 3,
        max_prot_len: int | None = None,
        max_atoms: int | None = None,
    ):
        self.smiles = list(smiles)
        self.seqs = list(seqs)
        self.labels = list(map(int, labels))
        self.mol_cache = mol_cache
        self.vocab = vocab
        self.kmer = int(kmer)
        self.max_prot_len = max_prot_len
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        seq = self.seqs[idx]
        y = int(self.labels[idx])

        atom_feat, adj = self.mol_cache[smi]
        if self.max_atoms is not None and atom_feat.shape[0] > self.max_atoms:
            # truncate (keeps first max_atoms)
            atom_feat = atom_feat[: self.max_atoms]
            adj = adj[: self.max_atoms, : self.max_atoms]

        km = seq_to_kmers(seq, k=self.kmer)
        if self.max_prot_len is not None and len(km) > self.max_prot_len:
            km = km[: self.max_prot_len]
        prot_ids = kmers_to_ids(km, self.vocab)

        return (
            torch.tensor(atom_feat, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(prot_ids, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def collate_batch(batch, pad_idx: int = 0):
    atoms, adjs, prot_ids, labels = zip(*batch)
    N = len(batch)

    atom_lens = [a.shape[0] for a in atoms]
    prot_lens = [p.shape[0] for p in prot_ids]
    max_atoms = max(atom_lens) if atom_lens else 0
    max_prot = max(prot_lens) if prot_lens else 0

    atoms_new = torch.zeros((N, max_atoms, NUM_ATOM_FEAT), dtype=torch.float32)
    adjs_new = torch.zeros((N, max_atoms, max_atoms), dtype=torch.float32)
    prot_new = torch.full((N, max_prot), pad_idx, dtype=torch.long)
    labels_new = torch.stack(labels).long()

    for i in range(N):
        a_len = atom_lens[i]
        p_len = prot_lens[i]
        atoms_new[i, :a_len, :] = atoms[i]

        # add self-loop like the original pack()
        adj = adjs[i]
        adj = adj + torch.eye(a_len, dtype=torch.float32)
        adjs_new[i, :a_len, :a_len] = adj

        if p_len > 0:
            prot_new[i, :p_len] = prot_ids[i]

    atom_num = torch.tensor(atom_lens, dtype=torch.long)
    prot_num = torch.tensor(prot_lens, dtype=torch.long)

    return atoms_new, adjs_new, prot_new, labels_new, atom_num, prot_num


# -----------------------------
# Model (from repo, with a protein embedding layer)
# -----------------------------


class SelfAttention(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    """Protein feature extraction (Conv + GLU + residual)."""

    def __init__(
        self,
        protein_dim: int,
        hid_dim: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        # protein: [batch, prot_len, protein_dim]
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    """Compound feature extraction + cross-attention to protein."""

    def __init__(
        self,
        atom_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device

        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.do_1 = nn.Dropout(0.2)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ft(trg)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # original code: use atom-wise norm as attention weights
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        # weighted sum over atoms (vectorized)
        sum_vec = torch.sum(trg * norm.unsqueeze(-1), dim=1)

        x = self.do_1(F.relu(self.fc_1(sum_vec)))
        x = self.fc_2(x)
        return x


class TransformerCPI(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        protein_dim: int = 100,
        atom_dim: int = 34,
        hid_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 8,
        pf_dim: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 9,
        padding_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        self.prot_emb = nn.Embedding(vocab_size, protein_dim, padding_idx=padding_idx)
        self.encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, self.device)
        self.decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, self.device)

        self.weight_1 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self._init_weight()

    def _init_weight(self):
        stdv = 1.0 / math.sqrt(self.weight_1.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)

    def gcn(self, x, adj):
        support = torch.matmul(x, self.weight_1)
        out = torch.bmm(adj, support)
        support = torch.matmul(out, self.weight_2)
        out = torch.bmm(adj, support)
        return out

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = atom_num.shape[0]
        compound_mask = torch.zeros((N, compound_max_len), device=self.device)
        protein_mask = torch.zeros((N, protein_max_len), device=self.device)
        for i in range(N):
            compound_mask[i, : int(atom_num[i])] = 1
            protein_mask[i, : int(protein_num[i])] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def forward(self, atoms, adjs, prot_ids, atom_num, prot_num):
        # atoms: [B, A, 34], adjs: [B, A, A], prot_ids: [B, P]
        compound_max_len = atoms.shape[1]
        protein_max_len = prot_ids.shape[1]

        compound_mask, protein_mask = self.make_masks(atom_num, prot_num, compound_max_len, protein_max_len)

        atoms = self.gcn(atoms, adjs)

        prot = self.prot_emb(prot_ids)  # [B, P, 100]
        enc = self.encoder(prot)
        logits = self.decoder(atoms, enc, compound_mask, protein_mask)
        return logits


# -----------------------------
# Metrics
# -----------------------------


def compute_metrics(prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = (prob >= thr).astype(np.int64)

    out: Dict[str, float] = {}
    out["acc"] = float(accuracy_score(y_true, pred))
    out["sen"] = float(recall_score(y_true, pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, pred))

    try:
        out["auroc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["auprc"] = float("nan")

    return out


def find_best_threshold(prob: np.ndarray, y_true: np.ndarray, grid=None) -> float:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 199)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (prob >= t).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return float(best_t)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str):
    model.eval()
    loss_f = nn.CrossEntropyLoss(reduction="mean")

    loss_sum = 0.0
    n_sum = 0
    probs, labels = [], []

    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for batch in pbar:
        atoms, adjs, prot_ids, y, atom_num, prot_num = batch
        atoms = atoms.to(device, non_blocking=True)
        adjs = adjs.to(device, non_blocking=True)
        prot_ids = prot_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        atom_num = atom_num.to(device, non_blocking=True)
        prot_num = prot_num.to(device, non_blocking=True)
        logits = model(atoms, adjs, prot_ids, atom_num, prot_num)
        loss = loss_f(logits, y)

        prob = torch.softmax(logits, dim=1)[:, 1]
        bs = int(y.shape[0])

        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs

        probs.append(prob.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

        pbar.set_postfix_str(f"loss={float(loss.detach().cpu()):.4f}")

    pbar.close()
    prob_all = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    y_all = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    return loss_sum / max(1, n_sum), prob_all, y_all


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer, desc: str):
    model.train()
    loss_f = nn.CrossEntropyLoss(reduction="mean")

    loss_sum = 0.0
    n_sum = 0

    pbar = tqdm(loader, total=len(loader), ncols=120, leave=False, desc=f"[{desc}]")
    for batch in pbar:
        atoms, adjs, prot_ids, y, atom_num, prot_num = batch
        atoms = atoms.to(device, non_blocking=True)
        adjs = adjs.to(device, non_blocking=True)
        prot_ids = prot_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        atom_num = atom_num.to(device, non_blocking=True)
        prot_num = prot_num.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(atoms, adjs, prot_ids, atom_num, prot_num)
        loss = loss_f(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = int(y.shape[0])
        loss_sum += float(loss.detach().cpu()) * bs
        n_sum += bs
        pbar.set_postfix_str(f"loss={float(loss.detach().cpu()):.4f}")

    pbar.close()
    return loss_sum / max(1, n_sum)


# -----------------------------
# Main
# -----------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name (folder under data-root)")
    ap.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    ap.add_argument("--out-root", type=str, default="/root/lanyun-tmp/transformercpi-runs")

    ap.add_argument(
        "--split-mode",
        choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
        default="cold-protein",
    )
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--overall-val", type=float, default=0.10)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--es-min-delta", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)

    # model hparams
    ap.add_argument("--hid-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--pf-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kernel-size", type=int, default=9)

    # protein kmers
    ap.add_argument("--kmer", type=int, default=3)
    ap.add_argument("--min-kmer-freq", type=int, default=1)
    ap.add_argument("--max-prot-len", type=int, default=None)
    ap.add_argument("--max-atoms", type=int, default=None)

    ap.add_argument("--resume", action="store_true", help="resume fold from fold_dir/last.pt if exists")
    return ap.parse_args()


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
    labels = df_all["Label"].values.astype(np.int64)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

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

    # ---------- precompute molecule cache (safe across folds) ----------
    cache_path = run_dir / "mol_cache.pkl"
    mol_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if cache_path.exists():
        import pickle

        with open(cache_path, "rb") as f:
            mol_cache = pickle.load(f)
        print(f"[CACHE] loaded mol_cache with {len(mol_cache)} SMILES")
    else:
        print("[CACHE] building mol_cache (RDKit featurization)...")
        uniq = list(dict.fromkeys(df_all["SMILES"].tolist()))
        bad = 0
        for smi in tqdm(uniq, ncols=120):
            try:
                mol_cache[smi] = mol_features(smi)
            except Exception:
                bad += 1
        if bad > 0:
            print(f"[CACHE] warning: {bad} SMILES failed parsing and will be dropped")

        # drop bad rows
        if bad > 0:
            ok_mask = np.array([s in mol_cache for s in df_all["SMILES"].tolist()], dtype=bool)
            df_all = df_all.iloc[ok_mask].reset_index(drop=True)
            drug_key = df_all["SMILES"].values.astype(object)
            prot_key = df_all["Target Sequence"].values.astype(object)
            labels = df_all["Label"].values.astype(np.int64)

        import pickle

        with open(cache_path, "wb") as f:
            pickle.dump(mol_cache, f)
        print(f"[CACHE] saved mol_cache to {cache_path} (n={len(mol_cache)})")

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

        df_tr = df_all.iloc[tr_idx].copy().reset_index(drop=True)
        df_va = df_all.iloc[va_idx].copy().reset_index(drop=True)
        df_te = df_all.iloc[test_idx].copy().reset_index(drop=True)

        # build vocab from TRAIN proteins only (strict)
        vocab = build_kmer_vocab(
            df_tr["Target Sequence"].tolist(),
            k=int(args.kmer),
            min_freq=int(args.min_kmer_freq),
        )
        pad_idx = vocab["<PAD>"]

        dset_tr = CPIDataset(
            df_tr["SMILES"].tolist(),
            df_tr["Target Sequence"].tolist(),
            df_tr["Label"].tolist(),
            mol_cache=mol_cache,
            vocab=vocab,
            kmer=int(args.kmer),
            max_prot_len=args.max_prot_len,
            max_atoms=args.max_atoms,
        )
        dset_va = CPIDataset(
            df_va["SMILES"].tolist(),
            df_va["Target Sequence"].tolist(),
            df_va["Label"].tolist(),
            mol_cache=mol_cache,
            vocab=vocab,
            kmer=int(args.kmer),
            max_prot_len=args.max_prot_len,
            max_atoms=args.max_atoms,
        )
        dset_te = CPIDataset(
            df_te["SMILES"].tolist(),
            df_te["Target Sequence"].tolist(),
            df_te["Label"].tolist(),
            mol_cache=mol_cache,
            vocab=vocab,
            kmer=int(args.kmer),
            max_prot_len=args.max_prot_len,
            max_atoms=args.max_atoms,
        )

        def _collate(b):
            return collate_batch(b, pad_idx=pad_idx)

        dl_train = DataLoader(
            dset_tr,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=False,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
        )
        dl_eval = dict(
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            drop_last=False,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.workers) > 0),
        )
        dl_val = DataLoader(dset_va, **dl_eval)
        dl_test = DataLoader(dset_te, **dl_eval)

        model = TransformerCPI(
            vocab_size=len(vocab),
            protein_dim=100,
            atom_dim=34,
            hid_dim=int(args.hid_dim),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
            pf_dim=int(args.pf_dim),
            dropout=float(args.dropout),
            kernel_size=int(args.kernel_size),
            padding_idx=pad_idx,
            device=device,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

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
            model.load_state_dict(ck["state_dict"], strict=False)
            optimizer.load_state_dict(ck["optimizer"])
            start_ep = int(ck.get("epoch", 0)) + 1
            best_score = float(ck.get("best_score", -1.0))
            best_thr = float(ck.get("best_thr", 0.5))
            best_epoch = int(ck.get("best_epoch", 0))
            no_improve = int(ck.get("no_improve", 0))
            print(
                f"[RESUME] start_ep={start_ep} best_score={best_score:.6f} best_epoch={best_epoch} no_improve={no_improve}"
            )

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
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_thr": best_thr,
                        "best_epoch": best_epoch,
                        "metrics": m,
                        "vocab_size": len(vocab),
                        "pad_idx": pad_idx,
                    },
                    str(best_pt),
                )
            else:
                no_improve += 1

            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_thr": best_thr,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "vocab_size": len(vocab),
                    "pad_idx": pad_idx,
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
        model.load_state_dict(ck["state_dict"], strict=False)

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
