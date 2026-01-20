import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

# ==========================================
# 1. Light GVP (Graph Encoder)
# ==========================================
class SimpleGVPConv(MessagePassing):
    """Lightweight Geometric Vector Perceptron Convolution"""
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
    """Encodes PocketGraph into a vector"""
    def __init__(self, out_dim=256):
        super().__init__()
        # [关键修复] 设置为正确的默认维度，避免Resume报错
        # s: 21(AA) + 1(pLDDT) + 1(RelPos) = 23
        self.s_emb = nn.Linear(23, out_dim) 
        # v: 4 vectors -> 16 channels
        self.v_emb = nn.Linear(4, 16)       
        # e: 16(RBF) + 1(SeqSep) = 17
        self.e_emb = nn.Linear(17, out_dim)  
        
        self.conv = SimpleGVPConv(out_dim, 16)
        self.out = nn.Sequential(
            nn.Linear(out_dim + 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, data):
        s, v, edge_index, edge_s = data.node_s, data.node_v, data.edge_index, data.edge_s
        
        # 动态适配 (保险起见保留，但默认值已修正)
        if s.shape[1] != self.s_emb.in_features: 
            self.s_emb = nn.Linear(s.shape[1], 256).to(s.device)
        if v.shape[1] != self.v_emb.in_features: 
            self.v_emb = nn.Linear(v.shape[1], 16).to(v.device)
        if edge_s.shape[1] != self.e_emb.in_features: 
            self.e_emb = nn.Linear(edge_s.shape[1], 256).to(edge_s.device)

        s = self.s_emb(s)
        edge_s = self.e_emb(edge_s)
        # v: (N, 4, 3) -> (N, 3, 4) -> Linear -> (N, 3, 16) -> (N, 16, 3)
        v = self.v_emb(v.transpose(1,2)).transpose(1,2)
        
        s = self.conv(s, v, edge_index, edge_s)
        
        v_norm = torch.norm(v, dim=-1)
        feat = torch.cat([s, v_norm], dim=-1)
        graph_vec = global_mean_pool(feat, data.batch)
        
        return self.out(graph_vec)

# ==========================================
# 2. Distributional UGCA (Uncertainty Core)
# ==========================================
class DistributionalUGCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Uncertainty Estimator (Log-Variance)
        self.log_var_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1) 
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x_q, x_kv):
        B, D = x_q.shape
        
        # 1. Uncertainty Estimation
        log_var = self.log_var_estimator(x_q)
        # [关键修复] 增加数值稳定性 Clamp
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        variance = F.softplus(log_var) 
        
        # 2. Gating
        gate = 1.0 / (1.0 + variance)
        
        # 3. Cross-Attention
        q = self.q(x_q).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, D)
        out = self.out_proj(out)
        
        # 4. Gated Residual
        fused = x_q + self.dropout(out * gate)
        return self.norm1(fused)

# ==========================================
# 3. Main Model
# ==========================================
class UGCADTI(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 256
        molclr_dim = 300 
        
        # Encoders
        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.chemberta_proj = nn.Sequential(
            nn.Linear(384, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim)
        
        # Interaction (UGCA)
        self.ugca_drug = DistributionalUGCA(self.dim)
        self.ugca_prot = DistributionalUGCA(self.dim)
        
        # Prediction Head
        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(self.dim, 1)
        )
        
    def forward(self, batch):
        d1 = self.molclr_proj(batch['molclr'])
        d2 = self.chemberta_proj(batch['chemberta'])
        h_d = d1 + d2 
        
        p1 = self.esm2_proj(batch['esm2'])
        p2 = self.pocket_proc(batch['graph'])
        h_p = p1 + p2
        
        z_d = self.ugca_drug(x_q=h_d, x_kv=h_p) 
        z_p = self.ugca_prot(x_q=h_p, x_kv=h_d) 
        
        z = torch.cat([z_d, z_p], dim=-1)
        logits = self.fusion(z)
        
        return logits