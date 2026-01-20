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
    def __init__(self, out_dim=256, dropout=0.1):
        super().__init__()
        # Dimensions must match src/encode/encode_pocket_gvp.py
        self.s_emb = nn.Linear(23, out_dim) 
        self.v_emb = nn.Linear(4, 16)       
        self.e_emb = nn.Linear(17, out_dim)  
        
        self.conv = SimpleGVPConv(out_dim, 16)
        self.out = nn.Sequential(
            nn.Linear(out_dim + 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout) # Configurable dropout
        )

    def forward(self, data):
        s, v, edge_index, edge_s = data.node_s, data.node_v, data.edge_index, data.edge_s
        
        # Safety check for dimensions
        if s.shape[1] != self.s_emb.in_features: 
            self.s_emb = nn.Linear(s.shape[1], self.s_emb.out_features).to(s.device)
        if v.shape[1] != self.v_emb.in_features:
            self.v_emb = nn.Linear(v.shape[1], self.v_emb.out_features).to(v.device)
        if edge_s.shape[1] != self.e_emb.in_features: 
            self.e_emb = nn.Linear(edge_s.shape[1], self.e_emb.out_features).to(edge_s.device)

        s = self.s_emb(s)
        edge_s = self.e_emb(edge_s)
        v = self.v_emb(v.transpose(1,2)).transpose(1,2)
        
        s = self.conv(s, v, edge_index, edge_s)
        
        v_norm = torch.norm(v, dim=-1)
        feat = torch.cat([s, v_norm], dim=-1)
        graph_vec = global_mean_pool(feat, data.batch)
        
        return self.out(graph_vec)

# ==========================================
# 2. Robust UGCA (MLP Gating)
# ==========================================
class RobustUGCA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid() 
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) # Configurable dropout

    def forward(self, x_q, x_kv):
        B, D = x_q.shape
        u = self.gate_net(x_q) 
        gate = 1.0 - u
        
        q = self.q(x_q).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, D)
        out = self.out_proj(out)
        
        fused = x_q + self.dropout(out * gate)
        return self.norm1(fused)

# ==========================================
# 3. Main Model Architecture
# ==========================================
class UGCADTI(nn.Module):
    def __init__(self, dim=256, dropout=0.1, num_heads=4):
        super().__init__()
        self.dim = dim
        molclr_dim = 300 
        
        # Encoders with configurable dropout
        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.chemberta_proj = nn.Sequential(
            nn.Linear(384, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim, dropout=dropout)
        
        # Interaction (UGCA)
        self.ugca_drug = RobustUGCA(self.dim, num_heads=num_heads, dropout=dropout)
        self.ugca_prot = RobustUGCA(self.dim, num_heads=num_heads, dropout=dropout)
        
        # Prediction Head
        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
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