"""
model.py

A lightweight UGCA-DTI model that trains only small projection/interaction layers.
All 4 backbone encodings are precomputed offline.

Inputs:
- molclr vector (D_molclr,)
- chemberta vector (384,)
- esm2 vector (1280,) or pooled from (L,1280)
- pocket_graph pooled vector from GVP features

Core idea:
- project each modality to a small d_model (default 128)
- UGCA: uncertainty-gated collaborative attention between {drug modalities} and {protein modalities}
- lightweight uncertainty-aware fusion head (sum + factorized interactions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvidenceReliability(nn.Module):
    """
    Convert a feature vector to an evidential reliability score r in (0,1).

    We use a tiny linear head to produce evidence for K=2 classes:
      e = softplus(Wx + b) >= 0
      alpha = e + 1
      uncertainty u = K / sum(alpha)
      reliability r = 1 - u
    """
    def __init__(self, in_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d)
        e = F.softplus(self.fc(h))  # (B, K)
        alpha = e + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)  # (B,1)
        u = float(self.num_classes) / (S + 1e-8)
        r = 1.0 - u
        # clamp to keep numerical stability
        return torch.clamp(r, 0.0, 1.0)  # (B,1)


class UGCABlock(nn.Module):
    """
    Uncertainty-Gated Collaborative Attention (UGCA) between:
      drug tokens:  molclr, chemberta
      protein tokens: esm2, pocket

    Implementation is intentionally lightweight:
    - single-head attention
    - tokens are single vectors, so attention is across 2 tokens only
    - gating uses reliability r_d * r_p and a small sigmoid gate
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # shared projection to keep parameters low
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.gate = nn.Linear(2 * d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # per-modality reliability heads (tiny)
        self.rel_mc = EvidenceReliability(d_model)
        self.rel_cb = EvidenceReliability(d_model)
        self.rel_esm = EvidenceReliability(d_model)
        self.rel_poc = EvidenceReliability(d_model)

    def _attend(
        self,
        q_src: torch.Tensor,           # (B,d)
        src_rel: torch.Tensor,         # (B,1)
        kv_list: Tuple[torch.Tensor, torch.Tensor],  # list of (h, rel)
    ) -> torch.Tensor:
        # Build K,V tensors
        hs = [h for (h, _) in kv_list]  # len=2, each (B,d)
        rs = [r for (_, r) in kv_list]  # len=2, each (B,1)

        K = torch.stack([self.Wk(h) for h in hs], dim=1)  # (B,2,d)
        V = torch.stack([self.Wv(h) for h in hs], dim=1)  # (B,2,d)

        q = self.Wq(q_src).unsqueeze(1)  # (B,1,d)
        logits = torch.sum(q * K, dim=-1) / math.sqrt(self.d_model)  # (B,2)

        # Gate each pair
        gates = []
        for h, r in zip(hs, rs):
            g = torch.sigmoid(self.gate(torch.cat([q_src, h], dim=-1)))  # (B,1)
            g = g * (src_rel * r)  # (B,1)
            gates.append(g.squeeze(-1))  # (B,)
        G = torch.stack(gates, dim=1)  # (B,2)

        attn = torch.softmax(logits * G, dim=1)  # (B,2)
        ctx = torch.sum(attn.unsqueeze(-1) * V, dim=1)  # (B,d)

        out = self.norm(q_src + self.dropout(ctx))
        return out

    def forward(
        self,
        h_mc: torch.Tensor,
        h_cb: torch.Tensor,
        h_esm: torch.Tensor,
        h_poc: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        # reliabilities
        r_mc = self.rel_mc(h_mc)
        r_cb = self.rel_cb(h_cb)
        r_esm = self.rel_esm(h_esm)
        r_poc = self.rel_poc(h_poc)

        # drug -> protein
        h_mc2 = self._attend(h_mc, r_mc, ((h_esm, r_esm), (h_poc, r_poc)))
        h_cb2 = self._attend(h_cb, r_cb, ((h_esm, r_esm), (h_poc, r_poc)))

        # protein -> drug
        h_esm2 = self._attend(h_esm, r_esm, ((h_mc, r_mc), (h_cb, r_cb)))
        h_poc2 = self._attend(h_poc, r_poc, ((h_mc, r_mc), (h_cb, r_cb)))

        rel = {"molclr": r_mc, "chemberta": r_cb, "esm2": r_esm, "pocket": r_poc}
        return (h_mc2, h_cb2, h_esm2, h_poc2), rel


class UFiSHFusion(nn.Module):
    """
    U-FiSH: Uncertainty-weighted Factorized Interaction & Sum Head

    Inputs: 4 modality vectors h_i (B,d) and reliabilities r_i (B,1).
    - weights w_i = softmax(r_i / tau)
    - sum fusion: z_sum = Σ w_i h_i
    - factorized interaction: z_int = Σ_{i<j} sqrt(w_i w_j) (h_i ⊙ h_j)
    Output feature: concat(z_sum, z_int)  -> (B, 2d)
    """
    def __init__(self, d_model: int, tau: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.tau = nn.Parameter(torch.tensor(float(tau)))

    def forward(self, hs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], rs: torch.Tensor) -> torch.Tensor:
        # hs: 4*(B,d)
        # rs: (B,4,1) or (B,4)
        if rs.dim() == 3:
            rs = rs.squeeze(-1)
        w = torch.softmax(rs / (self.tau.abs() + 1e-6), dim=1)  # (B,4)

        h_stack = torch.stack(list(hs), dim=1)  # (B,4,d)
        z_sum = torch.sum(w.unsqueeze(-1) * h_stack, dim=1)  # (B,d)

        # factorized interactions (no extra params)
        z_int = torch.zeros_like(z_sum)
        for i in range(4):
            for j in range(i + 1, 4):
                wij = torch.sqrt(w[:, i] * w[:, j]).unsqueeze(-1)  # (B,1)
                z_int = z_int + wij * (h_stack[:, i, :] * h_stack[:, j, :])
        z = torch.cat([z_sum, z_int], dim=-1)  # (B,2d)
        return z


@dataclass
class ModelConfig:
    in_molclr: int
    in_chemberta: int
    in_esm2: int
    in_pocket: int
    d_model: int = 128
    ugca_layers: int = 1
    dropout: float = 0.1


class UGCADTI(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.proj_molclr = MLPProjector(cfg.in_molclr, cfg.d_model, cfg.dropout)
        self.proj_chemberta = MLPProjector(cfg.in_chemberta, cfg.d_model, cfg.dropout)
        self.proj_esm2 = MLPProjector(cfg.in_esm2, cfg.d_model, cfg.dropout)
        self.proj_pocket = MLPProjector(cfg.in_pocket, cfg.d_model, cfg.dropout)

        self.ugca = nn.ModuleList([UGCABlock(cfg.d_model, dropout=cfg.dropout) for _ in range(cfg.ugca_layers)])
        self.fusion = UFiSHFusion(cfg.d_model, tau=1.0)

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        molclr: torch.Tensor,
        chemberta: torch.Tensor,
        esm2: torch.Tensor,
        pocket: torch.Tensor,
        return_aux: bool = False,
    ):
        # project
        h_mc = self.proj_molclr(molclr)
        h_cb = self.proj_chemberta(chemberta)
        h_esm = self.proj_esm2(esm2)
        h_poc = self.proj_pocket(pocket)

        # UGCA interaction
        rel = None
        for layer in self.ugca:
            (h_mc, h_cb, h_esm, h_poc), rel = layer(h_mc, h_cb, h_esm, h_poc)

        # fuse
        # rel dict -> tensor (B,4)
        rs = torch.cat([rel["molclr"], rel["chemberta"], rel["esm2"], rel["pocket"]], dim=1)  # (B,4)
        z = self.fusion((h_mc, h_cb, h_esm, h_poc), rs)

        logits = self.classifier(z)

        if return_aux:
            return logits, {"reliability": rs.detach()}
        return logits
