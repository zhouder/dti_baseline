import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch
import math

class SimpleGVPConv(MessagePassing):
    """
    几何向量感知器卷积层 (GVP-style)
    """
    def __init__(self, s_dim, v_dim):
        super().__init__(aggr='mean')
        self.message_net = nn.Sequential(
            nn.Linear(s_dim * 2 + v_dim + s_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, s_dim)
        )

    def forward(self, s, v, edge_index, edge_s):
        v_norm = torch.norm(v, dim=-1)
        return self.propagate(edge_index, s=s, v_norm=v_norm, edge_s=edge_s)

    def message(self, s_i, s_j, v_norm_j, edge_s):
        return self.message_net(torch.cat([s_i, s_j, v_norm_j, edge_s], dim=-1))

    def update(self, aggr_out, s):
        return s + aggr_out

class PocketGraphProcessor(nn.Module):
    def __init__(self, out_dim=256, dropout=0.1, top_k=32):
        super().__init__()
        self.out_dim = out_dim
        self.top_k = top_k # [修改] 默认降到 32

        self.s_emb = nn.Linear(23, out_dim)
        self.v_emb = nn.Linear(4, 16)
        self.e_emb = nn.Linear(18, out_dim) 
        self.conv = SimpleGVPConv(out_dim, 16)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(out_dim + 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, data, return_tokens=False):
        s, v, edge_index, edge_s = data.node_s, data.node_v, data.edge_index, data.edge_s

        if hasattr(data, "edge_v"):
            edge_v_norm = torch.norm(data.edge_v, dim=-1).view(data.edge_v.size(0), -1)
            edge_s_combined = torch.cat([edge_s, edge_v_norm], dim=-1)
        else:
            edge_s_combined = torch.cat([edge_s, torch.zeros(edge_s.size(0), 1, device=edge_s.device)], dim=-1)

        s = self.s_emb(s)
        edge_s = self.e_emb(edge_s_combined) 
        v = self.v_emb(v.transpose(1,2)).transpose(1,2)
        s = self.conv(s, v, edge_index, edge_s)
        
        v_norm = torch.norm(v, dim=-1) 
        feat = torch.cat([s, v_norm], dim=-1) 
        
        node_feat = self.node_mlp(feat) 
        graph_vec = global_mean_pool(node_feat, data.batch) 

        if not return_tokens:
            return graph_vec
        
        # [修改] Top-K 筛选逻辑 (保留最重要的节点)
        tokens, mask = to_dense_batch(node_feat, data.batch) # (B, N_max, D)
        B, N_max, D = tokens.shape
        
        if N_max > self.top_k:
            scores = torch.norm(tokens, dim=-1) # (B, N_max)
            scores = scores.masked_fill(~mask, float('-inf'))
            _, topk_indices = torch.topk(scores, k=self.top_k, dim=1) # (B, K)
            
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, 1, topk_indices_expanded) # (B, K, D)
            # Mask 对于 topk 来说全为 True (除非样本节点数本来就少于 K)
            mask = torch.gather(mask, 1, topk_indices) 
            
        return graph_vec, tokens, mask

class CrossAttnUGCA(nn.Module):
    """
    [创新点升级] Temperature-Gated Top-M Cross-Attention
    更稳、更轻量。不使用 logit bias 强抑制，而是用不确定性调节温度。
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, top_m=24, temp_c=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.top_m = top_m # 参与 Attention 的最大 Token 数
        self.temp_c = temp_c # 温度调节系数
        
        # 投影层 (手动实现 MHA 以便魔改)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Gate Net: 输入 (Query, Key) -> Evidence
        # 输入维度: 2 * dim (因为是一个个算，也可以加上 diff/prod)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 4, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus() 
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec, kv_tokens, kv_mask):
        """
        q_vec: (B, D)
        kv_tokens: (B, L, D)
        kv_mask: (B, L)
        """
        B, L, D = kv_tokens.shape
        Q_global = q_vec.unsqueeze(1) # (B, 1, D)

        # --- 1. 计算 Token 级不确定性 ---
        # 扩展 Query 与每个 KV Token 交互
        Q_expand = Q_global.expand(-1, L, -1)
        int_feat = torch.cat([
            Q_expand, kv_tokens, 
            torch.abs(Q_expand - kv_tokens), 
            Q_expand * kv_tokens
        ], dim=-1)
        
        e_tok = self.gate_net(int_feat).squeeze(-1) # (B, L)
        u_tok = 1.0 / (e_tok + 1.0) # (B, L)
        g_tok = 1.0 - u_tok         # (B, L) Reliability

        # --- 2. Top-M 筛选 (降低计算量 + 排除噪声) ---
        # 确保不选到 Padding
        g_scores = g_tok.masked_fill(~kv_mask, float('-inf'))
        
        # 动态确定实际 K (如果序列本身就很短)
        actual_k = min(self.top_m, L)
        
        # 选出 Top-M 可靠的 Tokens
        _, topk_idx = torch.topk(g_scores, k=actual_k, dim=1) # (B, M)
        
        # Gather KV and Uncertainty
        # 扩展 idx 用于 gather features
        topk_idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        
        K_selected = torch.gather(kv_tokens, 1, topk_idx_feat) # (B, M, D)
        V_selected = K_selected # K=V
        u_selected = torch.gather(u_tok, 1, topk_idx) # (B, M)

        # --- 3. Temperature-Scaled Cross Attention ---
        # 投影
        q = self.q_proj(Q_global).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, 1, d)
        k = self.k_proj(K_selected).view(B, actual_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, M, d)
        v = self.v_proj(V_selected).view(B, actual_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, M, d)

        # Logits (B, H, 1, M)
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        
        # [核心创新] 温度门控
        # u_selected: (B, M) -> (B, 1, 1, M) 用于广播
        u_gate = u_selected.unsqueeze(1).unsqueeze(1)
        
        # 温度分母: T = 1 + c * u
        # u 越大 -> T 越大 -> attention 分布越平滑 (High Entropy)
        # u 越小 -> T 接近 1 -> 正常 sharp attention
        temp = 1.0 + self.temp_c * u_gate
        attn_logits = attn_logits / temp

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Aggregation
        out = (attn_weights @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.out_proj(out).squeeze(1)

        # --- 4. 融合 ---
        fused = self.norm(q_vec + self.dropout(out))
        
        # 统计平均不确定性 (用于监控)
        u_avg = u_selected.mean(dim=1)
        
        return fused, u_avg.unsqueeze(1), e_tok, attn_weights

class UGCADTI(nn.Module):
    def __init__(self, dim=256, dropout=0.1, num_heads=4):
        super().__init__()
        self.dim = dim
        molclr_dim = 300
        
        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.chemberta_proj = nn.Sequential(
            nn.Linear(384, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout))
        
        # [修改] Pocket Top-K = 32
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim, dropout=dropout, top_k=32)

        # [修改] UGCA (Top-M = 24, Temp-C = 1.0)
        self.ugca_drug = CrossAttnUGCA(dim, num_heads=num_heads, dropout=dropout, top_m=24, temp_c=1.0)
        self.ugca_prot = CrossAttnUGCA(dim, num_heads=num_heads, dropout=dropout, top_m=24, temp_c=1.0)

        # 普通 Evidential Head (拼接)
        self.fusion_evidence = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.dim, 2), nn.Softplus()
        )

    def forward(self, batch, epoch=None):
        # --- Drug Tokens (B, 2, D) ---
        # [修改] 不再切分，直接用 MolCLR 和 ChemBERTa 各 1 个
        d1 = self.molclr_proj(batch['molclr']) 
        d2 = self.chemberta_proj(batch['chemberta'])
        
        # Query: Sum
        h_d = d1 + d2
        
        # KV: 2 Tokens
        drug_tokens = torch.stack([d1, d2], dim=1) 
        drug_mask = torch.ones(drug_tokens.size(0), 2, dtype=torch.bool, device=drug_tokens.device)

        # --- Protein Tokens (B, 2+K, D) ---
        p1 = self.esm2_proj(batch['esm2']) 
        # Pocket Global + Tokens
        pocket_vec, pocket_tokens, pocket_mask = self.pocket_proc(batch['graph'], return_tokens=True)
        
        # Query: ESM2 + Pocket Global
        h_p = p1 + pocket_vec 

        # KV: [ESM2, Pocket_Global, Pocket_Nodes...]
        p_tokens = torch.cat([p1.unsqueeze(1), pocket_vec.unsqueeze(1), pocket_tokens], dim=1)
        p_mask = torch.cat([
            torch.ones(p1.size(0), 2, dtype=torch.bool, device=p1.device), # 2 global tokens
            pocket_mask
        ], dim=1)

        # --- UGCA Interaction (Temp Gate + Top-M) ---
        # 此时无需 epoch warmup，因为 Temp 机制比较温和
        z_d, u_d, e_d, attn_dp = self.ugca_drug(h_d, p_tokens, p_mask)
        z_p, u_p, e_p, attn_pd = self.ugca_prot(h_p, drug_tokens, drug_mask)

        # --- 朴素拼接 ---
        z = torch.cat([z_d, z_p], dim=-1)
        
        # --- Prediction ---
        e = self.fusion_evidence(z)
        alpha = e + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        p = alpha / S
        u = 2.0 / S

        return p, u, alpha, u_d, u_p, attn_dp, attn_pd