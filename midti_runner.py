# -*- coding: utf-8 -*-
"""
MIDTI One-File Runner (Enhanced Version)
Features:
1.  Complete Metrics: AUROC, AUPRC, F1, ACC, SEN, MCC
2.  Flexible Splits: warm, cold-drug, cold-protein, cold-pair, cold-both
3.  Robustness: Dtype fixes, Cache handling, Device consistency
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# SECTION 1: Model Architectures
# ==============================================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        # STRICT TYPE CASTING
        if adj.dtype != torch.float32:
            adj = adj.float()
        if input.dtype != torch.float32:
            input = input.float()
            
        # STRICT DEVICE CHECK
        if adj.device != self.weight.device:
            adj = adj.to(self.weight.device)
        if input.device != self.weight.device:
            input = input.to(self.weight.device)

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class ReGCN_Flexible(nn.Module):
    """Pretraining Model: Multi-view GCN autoencoder"""
    def __init__(self, num_nodes, num_views, feature_dim=512):
        super(ReGCN_Flexible, self).__init__()
        self.num_nodes = num_nodes
        self.num_views = num_views
        
        self.gcn1_list = nn.ModuleList([
            GraphConvolution(num_nodes, feature_dim) for _ in range(num_views)
        ])
        self.gcn2_list = nn.ModuleList([
            GraphConvolution(feature_dim, feature_dim) for _ in range(num_views)
        ])
        
        self.att_fc = nn.Sequential(
            nn.Linear(num_views, num_views * 2),
            nn.ReLU(),
            nn.Linear(num_views * 2, num_views),
            nn.Sigmoid()
        )

    def forward(self, adj_matrices):
        # FORCE FLOAT32 for all inputs
        adj_matrices = [adj.float() for adj in adj_matrices]
        
        device = adj_matrices[0].device
        x_ident = torch.eye(self.num_nodes).float().to(device)
        
        embs = []
        for i in range(self.num_views):
            h = F.relu(self.gcn1_list[i](adj_matrices[i], x_ident))
            h = F.relu(self.gcn2_list[i](adj_matrices[i], h))
            embs.append(h)
        
        stack = torch.stack(embs, dim=0)
        view_sig = stack.mean(dim=(1, 2)).unsqueeze(0)
        att = self.att_fc(view_sig).view(self.num_views, 1, 1)
        fused = (stack * att).sum(dim=0)
        recon = torch.mm(fused, fused.t())
        return fused, recon

class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()
        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.nhead = n_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = int(hid_dim / n_heads)

    def forward(self, v, k, q):
        B = q.size(0)
        v = self.linear_v(v).view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.linear_k(k).view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        q = self.linear_q(q).view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.hid_dim)
        return self.linear_merge(out)

class Deep_inter_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(Deep_inter_att, self).__init__()
        self.sda = MHAtt(dim, nhead, dropout)
        self.sta = MHAtt(dim, nhead, dropout)
        self.dta = MHAtt(dim, nhead, dropout)
        self.tda = MHAtt(dim, nhead, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, d, p):
        d = self.norm(d + self.sda(d, d, d))
        p = self.norm(p + self.sta(p, p, p))
        d_new = self.norm(d + self.dta(p, p, d))
        p_new = self.norm(p + self.tda(d, d, p))
        return d_new, p_new

class MIDTI(nn.Module):
    def __init__(self, n_d, n_p, dim=512, layers=2):
        super(MIDTI, self).__init__()
        self.gcn_d = nn.ModuleList([GraphConvolution(dim, dim) for _ in range(2)])
        self.gcn_p = nn.ModuleList([GraphConvolution(dim, dim) for _ in range(2)])
        self.gcn_bi = nn.ModuleList([GraphConvolution(dim, dim) for _ in range(2)])
        self.n_d = n_d
        self.att_layers = nn.ModuleList([Deep_inter_att(dim, 4, 0.1) for _ in range(layers)])
        self.fc_d = nn.Linear((layers + 1) * dim, dim)
        self.fc_p = nn.Linear((layers + 1) * dim, dim)
        self.classifier = nn.Sequential(
            nn.Linear(2*dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, d_idx, p_idx, dd_adj, pp_adj, bi_adj, d_feat, p_feat):
        # FORCE FLOAT32
        dd_adj = dd_adj.float()
        pp_adj = pp_adj.float()
        bi_adj = bi_adj.float()
        d_feat = d_feat.float()
        p_feat = p_feat.float()
        
        d_views = [d_feat]
        curr = d_feat
        for gcn in self.gcn_d:
            curr = F.relu(gcn(dd_adj, curr))
            d_views.append(curr)
            
        p_views = [p_feat]
        curr = p_feat
        for gcn in self.gcn_p:
            curr = F.relu(gcn(pp_adj, curr))
            p_views.append(curr)
            
        x_all = torch.cat([d_feat, p_feat], dim=0)
        curr = x_all
        bi_views_d, bi_views_p = [], []
        for gcn in self.gcn_bi:
            curr = F.relu(gcn(bi_adj, curr))
            bi_views_d.append(curr[:self.n_d])
            bi_views_p.append(curr[self.n_d:])
            
        d_stack = torch.stack(d_views + bi_views_d, dim=1)[d_idx] 
        p_stack = torch.stack(p_views + bi_views_p, dim=1)[p_idx]
        
        d_hist, p_hist = d_stack, p_stack
        curr_d, curr_p = d_stack, p_stack
        
        for layer in self.att_layers:
            curr_d, curr_p = layer(curr_d, curr_p)
            d_hist = torch.cat([d_hist, curr_d], dim=-1)
            p_hist = torch.cat([p_hist, curr_p], dim=-1)
            
        d_vec = self.fc_d(d_hist).mean(dim=1)
        p_vec = self.fc_p(p_hist).mean(dim=1)
        
        feat = torch.cat([d_vec, p_vec], dim=1)
        return self.classifier(feat)

# ==============================================================================
# SECTION 2: Utils & Splitting
# ==============================================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

# ==============================================================================
# SECTION 3: Data Pipeline
# ==============================================================================

class MIDTIDataset(Dataset):
    def __init__(self, idx_list, df, d_map, p_map, labels):
        self.idx_list = idx_list
        self.df = df
        self.d_map = d_map
        self.p_map = p_map
        self.labels = labels

    def __len__(self): return len(self.idx_list)

    def __getitem__(self, i):
        real_idx = self.idx_list[i]
        row = self.df.iloc[real_idx]
        try:
            d_idx = self.d_map[str(row["SMILES"])]
            p_idx = self.p_map[str(row["Target Sequence"])]
        except KeyError:
            d_idx = 0
            p_idx = 0
        y = self.labels[real_idx]
        return d_idx, p_idx, y

def normalize_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def build_features(items, k_list, name):
    print(f"[Feature] Generating {name} similarity views...", flush=True)
    views = []
    vec = CountVectorizer(min_df=1, max_features=512, binary=True)
    
    for k in k_list:
        try:
            texts = [" ".join([str(s)[i:i+k] for i in range(len(str(s))-k+1)]) for s in items]
            if not texts: raise ValueError("Empty texts")
            X = vec.fit_transform(texts)
            sim = cosine_similarity(X).astype(np.float32) # Explicit float32
            views.append(normalize_adj(sim))
        except Exception as e:
            print(f"Error k={k}: {e}", flush=True)
            sim = np.eye(len(items), dtype=np.float32)
            views.append(normalize_adj(sim))
    return views

@torch.no_grad()
def evaluate_model(model, loader, device, desc, dd_adj, pp_adj, bi_adj, d_feat, p_feat):
    model.eval()
    loss_f = nn.BCEWithLogitsLoss()
    
    loss_sum = 0.0
    n_sum = 0
    probs, labels = [], []
    
    # Optional: use tqdm if verbose
    # pbar = tqdm(loader, ncols=100, leave=False, desc=desc)
    
    for d, p, y in loader:
        d = d.to(device)
        p = p.to(device)
        y = y.to(device, dtype=torch.float32).view(-1, 1)
        
        score = model(d, p, dd_adj, pp_adj, bi_adj, d_feat, p_feat)
        loss = loss_f(score, y)
        prob = torch.sigmoid(score)
        
        n = y.numel()
        loss_sum += loss.item() * n
        n_sum += n
        
        probs.append(prob.cpu().numpy())
        labels.append(y.cpu().numpy())
        
    prob_all = np.concatenate(probs) if probs else np.zeros((0,))
    y_all = np.concatenate(labels) if labels else np.zeros((0,))
    
    return loss_sum / max(1, n_sum), prob_all, y_all

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=str, default="/root/lanyun-fs")
    parser.add_argument("--out-root", type=str, default="/root/lanyun-tmp/midti-run")
    
    parser.add_argument(
        "--split-mode",
        choices=["warm", "hot", "cold-protein", "cold-drug", "cold-pair", "cold-both"],
        default="cold-protein",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--overall-val", type=float, default=0.10)
    
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--es-min-delta", type=float, default=0.0)
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[Init] Device: {args.device}", flush=True)
    device = torch.device(args.device)
    try: _ = torch.tensor([1.0]).to(device)
    except: 
        print("[Warn] CUDA failed, falling back to CPU", flush=True)
        device = torch.device("cpu")

    set_seed(args.seed)

    # --- Load CSV ---
    csv_path = Path(args.data_root) / args.dataset / f"{args.dataset}.csv"
    if not csv_path.exists():
        print(f"[Error] Not found: {csv_path}", flush=True)
        return

    print(f"[Load] Reading {csv_path}...", flush=True)
    df_all = pd.read_csv(csv_path)
    
    # Normalize Columns
    cols = df_all.columns
    c_s = next((c for c in cols if 'smile' in c.lower()), None)
    c_p = next((c for c in cols if 'seq' in c.lower()), None)
    c_y = next((c for c in cols if 'label' in c.lower() or 'y' == c.lower()), None)
    
    if not all([c_s, c_p, c_y]):
        print(f"[Error] Columns missing. Found: {cols}", flush=True)
        return
        
    df_all = df_all.rename(columns={c_s: 'SMILES', c_p: 'Target Sequence', c_y: 'Label'})
    df_all['SMILES'] = df_all['SMILES'].astype(str)
    df_all['Target Sequence'] = df_all['Target Sequence'].astype(str)
    df_all = df_all.dropna().reset_index(drop=True)
    
    drug_key = df_all['SMILES'].values
    prot_key = df_all['Target Sequence'].values
    labels = df_all['Label'].values.astype(np.float32)
    
    # Unique Maps
    drugs = df_all['SMILES'].unique()
    prots = df_all['Target Sequence'].unique()
    d_map = {str(d): i for i, d in enumerate(drugs)}
    p_map = {str(p): i for i, p in enumerate(prots)}
    
    print(f"[Data] Drugs: {len(drugs)}, Proteins: {len(prots)}, Labels: {len(df_all)}", flush=True)

    # --- Pretraining / Feature Loading ---
    # We do this ONCE for the whole dataset (Transductive Setting usually assumes full feature availability)
    # Note: For strict inductive setting, one might re-build features per fold, but MIDTI is GCN based
    # and usually operates on the full graph structure.
    
    out_dir = Path(args.out_root) / f"{args.dataset}_{args.split_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache path specific to dataset (independent of split mode)
    cache_dir = Path(args.out_root) / f"{args.dataset}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "features.pt"
    
    need_pretrain = True
    if cache_file.exists():
        print("[Cache] Loading features...", flush=True)
        try:
            cache = torch.load(cache_file, map_location=device)
            d_feat, p_feat, dd_adj, pp_adj = cache['d'], cache['p'], cache['dd'], cache['pp']
            # FORCE FLOAT32 LOAD
            d_feat = d_feat.float()
            p_feat = p_feat.float()
            dd_adj = dd_adj.float()
            pp_adj = pp_adj.float()
            
            if d_feat.size(0) == len(drugs) and p_feat.size(0) == len(prots):
                need_pretrain = False
            else:
                print("[Cache] Dimension mismatch, re-running...", flush=True)
        except Exception as e:
            print(f"[Cache] Load failed ({e}), re-running...", flush=True)

    if need_pretrain:
        d_views_np = build_features(drugs, [2,3], "Drug")
        p_views_np = build_features(prots, [3,4], "Protein")
        
        d_views = [torch.from_numpy(v.astype(np.float32)).to(device) for v in d_views_np]
        p_views = [torch.from_numpy(v.astype(np.float32)).to(device) for v in p_views_np]
        
        print("[Pretrain] Training ReGCN...", flush=True)
        regcn_d = ReGCN_Flexible(len(drugs), len(d_views)).to(device)
        opt_d = optim.Adam(regcn_d.parameters(), lr=1e-3)
        for i in range(20): 
            feat, recon = regcn_d(d_views)
            loss = sum([F.mse_loss(recon, v) for v in d_views])
            opt_d.zero_grad(); loss.backward(); opt_d.step()
            
        d_feat, _ = regcn_d(d_views)
        d_feat = d_feat.detach().float()
        dd_adj = d_views[0].float()

        regcn_p = ReGCN_Flexible(len(prots), len(p_views)).to(device)
        opt_p = optim.Adam(regcn_p.parameters(), lr=1e-3)
        for i in range(20):
            feat, recon = regcn_p(p_views)
            loss = sum([F.mse_loss(recon, v) for v in p_views])
            opt_p.zero_grad(); loss.backward(); opt_p.step()
            
        p_feat, _ = regcn_p(p_views)
        p_feat = p_feat.detach().float()
        pp_adj = p_views[0].float()
        
        torch.save({'d':d_feat, 'p':p_feat, 'dd':dd_adj, 'pp':pp_adj}, cache_file)

    # --- Build Global Interaction Graph (Transductive Base) ---
    # For GCN, we typically need the full graph structure. 
    # To prevent leakage, we should strictly mask edges in the 'forward' pass that belong to test set.
    # However, standard MIDTI implementation builds the graph once. 
    # Here we build the base graph structure.
    print("[Graph] Building Base Interaction Graph...", flush=True)
    dp_adj_full = torch.zeros(len(drugs), len(prots), dtype=torch.float32).to(device)
    
    # We will fill this adjacency matrix dynamically inside the Fold Loop to ensure strict inductive split if needed,
    # or just use training edges.
    
    # --- Split Logic ---
    split_mode = "warm" if args.split_mode == "hot" else args.split_mode
    outer = make_outer_splits(split_mode, int(args.cv_folds), int(args.seed), drug_key, prot_key, labels)
    K = len(outer)
    overall_test = 1.0 / K
    val_frac_in_pool = float(args.overall_val) / (1.0 - overall_test)
    
    print(f"[SPLIT] {split_mode} | K={K} | val_frac_in_pool={val_frac_in_pool:.4f}")
    
    keys = ["auroc", "auprc", "f1", "acc", "sen", "mcc"]
    fold_metrics: List[Dict[str, float]] = []
    
    for fold, (train_pool_idx, test_idx) in enumerate(outer, start=1):
        fold_id = fold - 1
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        
        # Sub-split train into train/val
        tr_idx, va_idx = sample_val_indices(
            split_mode,
            train_pool_idx,
            val_frac_in_pool,
            seed=args.seed + 100 + fold,
            drug_key=drug_key,
            prot_key=prot_key,
            labels=labels,
        )
        
        print("=" * 80)
        print(f"[Fold {fold}/{K}] train={len(tr_idx)} val={len(va_idx)} test={len(test_idx)}")
        
        # --- Construct Fold-Specific Graph (Masking Test Edges) ---
        # Only use training edges for the Interaction Graph part of GCN
        # This prevents label leakage in transductive setting
        dp_adj = torch.zeros(len(drugs), len(prots), dtype=torch.float32).to(device)
        
        # Get Train Positives
        tr_pos_mask = (labels[tr_idx] == 1)
        tr_pos_idx = tr_idx[tr_pos_mask]
        
        tr_d_indices = [d_map[str(s)] for s in drug_key[tr_pos_idx]]
        tr_p_indices = [p_map[str(s)] for s in prot_key[tr_pos_idx]]
        
        if len(tr_d_indices) > 0:
            t_d = torch.tensor(tr_d_indices, dtype=torch.long, device=device)
            t_p = torch.tensor(tr_p_indices, dtype=torch.long, device=device)
            dp_adj.index_put_((t_d, t_p), torch.tensor(1.0, device=device))
            
        # Build Bipartite Adjacency
        top = torch.cat([torch.zeros(len(drugs), len(drugs), dtype=torch.float32).to(device), dp_adj], dim=1)
        bot = torch.cat([dp_adj.t(), torch.zeros(len(prots), len(prots), dtype=torch.float32).to(device)], dim=1)
        bi_adj = torch.cat([top, bot], dim=0)
        
        # Norm
        deg = bi_adj.sum(1)
        deg[deg==0] = 1
        d_inv = torch.diag(torch.pow(deg, -0.5))
        bi_adj_norm = torch.mm(d_inv, torch.mm(bi_adj, d_inv))
        
        # --- Datasets ---
        # Note: We pass full df but use indices to select rows
        tr_ds = MIDTIDataset(tr_idx, df_all, d_map, p_map, labels)
        va_ds = MIDTIDataset(va_idx, df_all, d_map, p_map, labels)
        te_ds = MIDTIDataset(test_idx, df_all, d_map, p_map, labels)
        
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)
        
        # --- Model & Opt ---
        model = MIDTI(len(drugs), len(prots)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # --- Training Loop ---
        fold_dir = out_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_pt = fold_dir / "best.pt"
        log_csv = fold_dir / "log.csv"
        
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "tr_loss", "va_loss", "va_auc", "va_f1", "va_thr"])
            
        best_score = -1.0
        best_thr = 0.5
        best_epoch = 0
        no_improve = 0
        
        for ep in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0
            n_tr = 0
            try:
                for d, p, y in tr_loader:
                    d, p = d.to(device), p.to(device)
                    y = y.to(device, dtype=torch.float32).view(-1, 1)
                    
                    optimizer.zero_grad()
                    out = model(d, p, dd_adj, pp_adj, bi_adj_norm, d_feat, p_feat)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item() * y.numel()
                    n_tr += y.numel()
            except Exception as e:
                print(f"[Error] Training crashed: {e}")
                traceback.print_exc()
                return

            tr_loss /= max(1, n_tr)
            
            # Validation
            va_loss, prob_va, y_va = evaluate_model(model, va_loader, device, "Val", dd_adj, pp_adj, bi_adj_norm, d_feat, p_feat)
            
            thr_now = find_best_threshold(prob_va, y_va)
            m_va = compute_metrics(prob_va, y_va, thr=thr_now)
            
            score = m_va["auprc"] if not math.isnan(m_va["auprc"]) else m_va["auroc"]
            
            # Logging
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([ep, tr_loss, va_loss, m_va['auroc'], m_va['f1'], thr_now])
            
            print(f"Ep {ep}: TrL {tr_loss:.4f} | VaL {va_loss:.4f} | AUC {m_va['auroc']:.4f} | AUPR {m_va['auprc']:.4f} | F1 {m_va['f1']:.4f}")
            
            if score > best_score + args.es_min_delta:
                best_score = score
                best_thr = thr_now
                best_epoch = ep
                no_improve = 0
                torch.save(model.state_dict(), best_pt)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stop at ep {ep}")
                    break
        
        # --- Test ---
        model.load_state_dict(torch.load(best_pt))
        te_loss, prob_te, y_te = evaluate_model(model, te_loader, device, "Test", dd_adj, pp_adj, bi_adj_norm, d_feat, p_feat)
        te_m = compute_metrics(prob_te, y_te, thr=best_thr)
        te_m["thr"] = best_thr
        te_m["fold"] = fold_id
        
        print(f"[TEST Fold {fold}] AUC {te_m['auroc']:.4f} | AUPR {te_m['auprc']:.4f} | F1 {te_m['f1']:.4f} | ACC {te_m['acc']:.4f} | SEN {te_m['sen']:.4f} | MCC {te_m['mcc']:.4f}")
        
        fold_metrics.append(te_m)
        
    # --- Summary ---
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys + ["thr"])
        for m in fold_metrics:
            w.writerow([m["fold"]] + [m.get(k, "nan") for k in keys] + [m["thr"]])
            
    mean_m = {k: np.nanmean([m[k] for m in fold_metrics]) for k in keys}
    std_m = {k: np.nanstd([m[k] for m in fold_metrics]) for k in keys}
    
    print("\n" + "="*40)
    print("FINAL CV RESULTS")
    for k in keys:
        print(f"{k.upper()}: {mean_m[k]:.4f} ± {std_m[k]:.4f}")
    print("="*40)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Script crashed: {e}")
        traceback.print_exc()