# src/model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, T, D), m: (B, T) bool (True=valid)
    return: (B, D)
    """
    m = m.to(dtype=x.dtype)
    num = (x * m.unsqueeze(-1)).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps).unsqueeze(-1)
    return num / den


class PocketGraphEncoder(nn.Module):
    """
    Lightweight message passing for pocket graphs (no PyG dependency).

    Input per sample:
      node_scalar_feat: (N, Fin) float
      edge_index: (2, E) long  (or (E,2) will be transposed)
    Output:
      global graph token: (d_model,)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_self = nn.Linear(hidden_dim, hidden_dim)
        self.fc_nei = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_scalar: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if node_scalar is None or node_scalar.numel() == 0:
            device = edge_index.device if (edge_index is not None and torch.is_tensor(edge_index)) else torch.device("cpu")
            return torch.zeros((self.fc_out.out_features,), device=device, dtype=torch.float32)

        x = F.relu(self.fc_in(node_scalar))
        x = self.dropout(x)
        N = x.size(0)

        if edge_index is None or edge_index.numel() == 0:
            h = F.relu(self.fc_self(x))
            return self.fc_out(h.mean(dim=0))

        if edge_index.ndim == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()

        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)
        msg = x[src]                              # (E, hidden)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
        one = torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, one)
        agg = agg / deg.clamp_min(1.0)

        h = F.relu(self.fc_self(x) + self.fc_nei(agg))
        h = self.dropout(h)
        return self.fc_out(h.mean(dim=0))


class CrossAttnGate(nn.Module):
    """
    Cross-attention with a simple residual gate (no entropy / uncertainty terms).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True => ignore
        attn_out, _ = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=~kv_mask,
            need_weights=False,
        )

        g = self.gate(torch.cat([q, attn_out], dim=-1))  # (B, Lq, 1)
        x = q + self.dropout(attn_out) * g
        x = self.ln1(x)
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x


class UGCABlock(nn.Module):
    """
    One UGCA layer:
      P <- D
      D <- P
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.p_from_d = CrossAttnGate(d_model, n_heads, dropout)
        self.d_from_p = CrossAttnGate(d_model, n_heads, dropout)

    def forward(self, P: torch.Tensor, Pm: torch.Tensor, D: torch.Tensor, Dm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        P2 = self.p_from_d(P, D, Dm)
        D2 = self.d_from_p(D, P, Pm)
        return P2, D2


@dataclass
class SeqCfg:
    # input dims
    d_protein: int = 1280
    d_molclr: int = 300
    d_chem: int = 384

    # model dims
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # pocket
    pocket_in_dim: int = 21
    pocket_hidden: int = 128


class UGCASeqPocketModel(nn.Module):
    """
    Sequence-only model (vec removed). Pocket is always used.

    Forward inputs:
      v_prot: (B, Mp, d_protein), m_prot: (B, Mp) bool
      v_mol:  (B, Md, d_molclr),  m_mol:  (B, Md) bool
      v_chem: (B, d_chem)
      pocket_list: list[dict] length B, each with keys: node_scalar_feat, edge_index
    """
    def __init__(self, cfg: SeqCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr, d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem, d, bias=False)

        self.pocket_enc = PocketGraphEncoder(
            in_dim=cfg.pocket_in_dim,
            hidden_dim=cfg.pocket_hidden,
            out_dim=d,
            dropout=cfg.dropout,
        )

        self.blocks = nn.ModuleList([UGCABlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])

        self.head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def _graph_token(self, pocket_list: List[dict], device: torch.device) -> torch.Tensor:
        g_list: List[torch.Tensor] = []
        for pg in pocket_list:
            if not isinstance(pg, dict):
                g_list.append(torch.zeros((self.cfg.d_model,), device=device, dtype=torch.float32))
                continue

            node = pg.get("node_scalar_feat", None)
            edge = pg.get("edge_index", None)

            if node is None or edge is None:
                g_list.append(torch.zeros((self.cfg.d_model,), device=device, dtype=torch.float32))
                continue

            node_t = node if torch.is_tensor(node) else torch.as_tensor(node, dtype=torch.float32, device=device)
            edge_t = edge if torch.is_tensor(edge) else torch.as_tensor(edge, dtype=torch.long, device=device)

            if node_t.numel() == 0:
                g_list.append(torch.zeros((self.cfg.d_model,), device=device, dtype=torch.float32))
                continue

            g_list.append(self.pocket_enc(node_t.to(device=device, dtype=torch.float32),
                                          edge_t.to(device=device, dtype=torch.long)))
        return torch.stack(g_list, dim=0)  # (B,d)

    def forward(
        self,
        v_prot: torch.Tensor, m_prot: torch.Tensor,
        v_mol: torch.Tensor,  m_mol: torch.Tensor,
        v_chem: torch.Tensor,
        pocket_list: List[dict],
        topk_ratio: Optional[float] = None,  # kept for compatibility, unused
    ) -> torch.Tensor:
        _ = topk_ratio
        device = v_prot.device
        B = v_prot.size(0)

        P = self.proj_p(v_prot)  # (B,Mp,d)
        D = self.proj_d(v_mol)   # (B,Md,d)
        Pm = m_prot
        Dm = m_mol

        # [CHEM] token -> prepend to drug
        chem_tok = self.proj_c(v_chem).unsqueeze(1)  # (B,1,d)
        D = torch.cat([chem_tok, D], dim=1)          # (B,Md+1,d)
        Dm = torch.cat([torch.ones((B, 1), dtype=torch.bool, device=device), Dm], dim=1)

        # [POCKET GRAPH] token -> prepend to protein (always)
        graph_tok = self._graph_token(pocket_list, device=device).unsqueeze(1)  # (B,1,d)
        P = torch.cat([graph_tok, P], dim=1)                                    # (B,Mp+1,d)
        Pm = torch.cat([torch.ones((B, 1), dtype=torch.bool, device=device), Pm], dim=1)

        for blk in self.blocks:
            P, D = blk(P, Pm, D, Dm)

        hP = masked_mean(P, Pm)
        hD = masked_mean(D, Dm)
        y = self.head(torch.cat([hP, hD], dim=-1)).squeeze(-1)  # (B,)
        return y


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build sequence+pocket model (pocket always enabled).
    Keep backward compatibility for key names:
      n_heads / nhead, n_layers / nlayers
    """
    n_heads = int(cfg.get("n_heads", cfg.get("nhead", 4)))
    n_layers = int(cfg.get("n_layers", cfg.get("nlayers", 2)))

    mcfg = SeqCfg(
        d_protein=int(cfg.get("d_protein", 1280)),
        d_molclr=int(cfg.get("d_molclr", 300)),
        d_chem=int(cfg.get("d_chem", 384)),
        d_model=int(cfg.get("d_model", 512)),
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=float(cfg.get("dropout", 0.1)),
        pocket_in_dim=int(cfg.get("pocket_in_dim", 21)),
        pocket_hidden=int(cfg.get("pocket_hidden", 128)),
    )
    return UGCASeqPocketModel(mcfg)
