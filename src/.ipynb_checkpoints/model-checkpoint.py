import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch

class SimpleGVPConv(MessagePassing):
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
    def __init__(self, out_dim=256, dropout=0.1, top_k=16):
        super().__init__()
        self.out_dim = out_dim
        self.top_k = top_k # [修改] 默认 Top-K = 16 (Short Sequence)

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
        
        # [核心] Top-K 筛选
        tokens, mask = to_dense_batch(node_feat, data.batch) # (B, N_max, D)
        B, N_max, D = tokens.shape
        
        if N_max > self.top_k:
            # 按特征范数排序，选最重要的节点
            scores = torch.norm(tokens, dim=-1) # (B, N_max)
            scores = scores.masked_fill(~mask, float('-inf'))
            _, topk_indices = torch.topk(scores, k=self.top_k, dim=1) # (B, K)
            
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            tokens = torch.gather(tokens, 1, topk_indices_expanded) # (B, K, D)
            mask = torch.gather(mask, 1, topk_indices) # (B, K)
            
        return graph_vec, tokens, mask

class CrossAttnUGCA(nn.Module):
    """
    UGCA: Uncertainty-Gated Cross-Attention (Route B)
    Mechanism: Evidence -> Gate -> Centered Logit Bias
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Gate Net
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 4, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Softplus()
        )
        
        # Init bias to open gate initially
        nn.init.constant_(self.gate_net[-2].bias, 2.0)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_vec, kv_tokens, kv_mask, beta=1.0):
        """
        beta: Warmup coefficient for logit bias (0->1)
        """
        B, L, D = kv_tokens.shape
        Q = q_vec.unsqueeze(1) # (B, 1, D)

        # --- 1. Compute Uncertainty Gate ---
        Q_expand = Q.expand(-1, L, -1)
        int_feat = torch.cat([
            Q_expand, kv_tokens, 
            torch.abs(Q_expand - kv_tokens), 
            Q_expand * kv_tokens
        ], dim=-1)
        
        e_tok = self.gate_net(int_feat).squeeze(-1) # (B, L)
        g_tok = e_tok / (e_tok + 1.0 + 1e-8) # Reliability Gate (0~1)
        
        # Soft clamp to avoid log(0)
        g_tok = g_tok.clamp(0.01, 0.99)
        
        # [核心] Centered Logit Bias
        # log(g) - mean(log(g))
        # 确保门控只改变 Token 间的相对权重，而不整体压低 logits
        log_g = torch.log(g_tok)
        log_g_centered = log_g - log_g.mean(dim=1, keepdim=True)
        
        # Apply Beta Warmup
        attn_bias = beta * log_g_centered # (B, L)
        
        # Match MHA shape: (B*H, 1, L)
        attn_bias = attn_bias.unsqueeze(1).repeat_interleave(self.num_heads, dim=0)

        # Handle Padding (Merge into bias)
        is_padding = ~kv_mask
        is_padding_expanded = is_padding.unsqueeze(1).repeat_interleave(self.num_heads, dim=0)
        attn_bias = attn_bias.masked_fill(is_padding_expanded, float('-inf'))

        # --- 2. Cross Attention ---
        ctx, _ = self.attn(Q, kv_tokens, kv_tokens, key_padding_mask=None, attn_mask=attn_bias)
        ctx = ctx.squeeze(1)

        # --- 3. Fusion ---
        fused = self.norm(q_vec + self.dropout(ctx))
        
        return fused, g_tok

class UGCADTI(nn.Module):
    def __init__(self, dim=256, dropout=0.1, num_heads=4):
        super().__init__()
        self.dim = dim
        self.warmup_epochs = 10 
        
        self.molclr_proj = nn.Sequential(
            nn.Linear(300, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.chemberta_proj = nn.Sequential(
            nn.Linear(384, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.LayerNorm(self.dim), nn.ReLU(), nn.Dropout(dropout)
        )
        
        # [修改] Top-K = 16 (Small K for stability)
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim, dropout=dropout, top_k=16)

        self.ugca_drug = CrossAttnUGCA(dim, num_heads=num_heads, dropout=dropout)
        self.ugca_prot = CrossAttnUGCA(dim, num_heads=num_heads, dropout=dropout)

        # Simple Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, 1)
        )

    def get_beta(self, epoch):
        if epoch is None or epoch < 0: return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    def forward(self, batch, epoch=None):
        beta = self.get_beta(epoch)

        # --- 1. Drug Inputs ---
        d1 = self.molclr_proj(batch['molclr']) 
        d2 = self.chemberta_proj(batch['chemberta'])
        h_d = d1 + d2 # Query
        
        # Drug KV: [d1, d2] (Length=2)
        drug_kv = torch.stack([d1, d2], dim=1)
        drug_mask = torch.ones(d1.size(0), 2, dtype=torch.bool, device=d1.device)

        # --- 2. Protein Inputs ---
        p1 = self.esm2_proj(batch['esm2']) 
        # Get global vec AND top-k nodes
        pocket_vec, pocket_nodes, pocket_mask = self.pocket_proc(batch['graph'], return_tokens=True)
        h_p = p1 + pocket_vec # Query

        # Protein KV: [p1, pocket_vec, pocket_nodes] (Length=2+K)
        # p1 and pocket_vec are global tokens (mask=True)
        global_mask = torch.ones(p1.size(0), 2, dtype=torch.bool, device=p1.device)
        
        prot_kv = torch.cat([p1.unsqueeze(1), pocket_vec.unsqueeze(1), pocket_nodes], dim=1)
        prot_mask = torch.cat([global_mask, pocket_mask], dim=1)

        # --- 3. UGCA Interaction ---
        z_d, g_d = self.ugca_drug(q_vec=h_d, kv_tokens=prot_kv, kv_mask=prot_mask, beta=beta)
        z_p, g_p = self.ugca_prot(q_vec=h_p, kv_tokens=drug_kv, kv_mask=drug_mask, beta=beta)

        # --- 4. Prediction ---
        z = torch.cat([z_d, z_p], dim=-1)
        logits = self.classifier(z)

        return logits, g_d, g_p