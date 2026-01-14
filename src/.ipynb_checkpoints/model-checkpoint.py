import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

# --- 1. Light GVP Sub-module ---
class SimpleGVPConv(MessagePassing):
    """Message passing handling scalar(s) and vector(v) features."""
    def __init__(self, s_dim, v_dim):
        super().__init__(aggr='mean')
        self.message_net = nn.Sequential(
            nn.Linear(s_dim * 2 + v_dim + s_dim, s_dim), # s_i, s_j, v_norm_j, edge_s
            nn.ReLU(),
            nn.Linear(s_dim, s_dim)
        )
    
    def forward(self, s, v, edge_index, edge_s):
        v_norm = torch.norm(v, dim=-1) # (N, v_dim)
        return self.propagate(edge_index, s=s, v_norm=v_norm, edge_s=edge_s)
    
    def message(self, s_i, s_j, v_norm_j, edge_s):
        # Concatenate: [Source, Target, VectorMagnitude, Edge]
        return self.message_net(torch.cat([s_i, s_j, v_norm_j, edge_s], dim=-1))
        
    def update(self, aggr_out, s):
        return s + aggr_out # Residual

class PocketGraphProcessor(nn.Module):
    """Encodes PocketGraph (.npz) into a fixed-length vector."""
    def __init__(self, out_dim=256):
        super().__init__()
        # Assumption on input dims based on common ESMFold/GVP formats
        self.s_emb = nn.Linear(29, out_dim) # node_s dim approx 29
        self.v_emb = nn.Linear(3, 16)       # node_v dim 3 -> project to 16 channels
        self.e_emb = nn.Linear(5, out_dim)  # edge_s dim approx 5
        
        self.conv = SimpleGVPConv(out_dim, 16)
        self.out = nn.Linear(out_dim + 16, out_dim) # s + v_norm

    def forward(self, data):
        s, v, edge_index, edge_s = data.node_s, data.node_v, data.edge_index, data.edge_s
        
        # Dynamic shape adaptation (prevents crashes on dim mismatch)
        if s.shape[1] != self.s_emb.in_features: self.s_emb = nn.Linear(s.shape[1], 256).to(s.device)
        if v.shape[1] != self.v_emb.in_features: self.v_emb = nn.Linear(v.shape[1], 16).to(v.device)
        if edge_s.shape[1] != self.e_emb.in_features: self.e_emb = nn.Linear(edge_s.shape[1], 256).to(edge_s.device)

        # Embedding
        s = self.s_emb(s)
        edge_s = self.e_emb(edge_s)
        v = self.v_emb(v.transpose(1,2)).transpose(1,2)
        
        # GVP Conv
        s = self.conv(s, v, edge_index, edge_s)
        
        # Global Pooling (Mean)
        v_norm = torch.norm(v, dim=-1)
        feat = torch.cat([s, v_norm], dim=-1)
        graph_vec = global_mean_pool(feat, data.batch)
        
        return self.out(graph_vec)

# --- 2. UGCA Module (Uncertainty Gated Cross Attention) ---
class UGCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.num_heads = 4
        self.dim = dim
        self.scale = (dim // 4) ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Uncertainty Gate
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_q, x_kv):
        B, D = x_q.shape
        
        # Calculate Uncertainty
        u = self.uncertainty_gate(x_q) # (B, 1)
        gate = 1.0 - u # Higher certainty -> Higher gate
        
        # Attention
        q = self.q(x_q).reshape(B, 1, 4, D//4).permute(0,2,1,3)
        k = self.k(x_kv).reshape(B, 1, 4, D//4).permute(0,2,1,3)
        v = self.v(x_kv).reshape(B, 1, 4, D//4).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, D)
        out = self.proj(out)
        
        # Gated Fusion
        return self.norm(x_q + out * gate)

# --- 3. Fusion & Loss ---
class EvidentialHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Linear(dim, 1)
        self.beta = nn.Linear(dim, 1)
        
    def forward(self, x):
        # Softplus + 1 ensures parameters > 1
        return F.softplus(self.alpha(x)) + 1.0, F.softplus(self.beta(x)) + 1.0

def evidential_loss(alpha, beta, y):
    # Binary Evidential Loss
    S = alpha + beta
    p = alpha / S
    # MSE Risk part of EDL
    loss = (y - p) ** 2 + (alpha * beta) / (S ** 2 * (S + 1))
    return torch.mean(loss)

# --- 4. Main Model ---
class UGCADTI(nn.Module):
    def __init__(self, molclr_dim=300):
        super().__init__()
        self.dim = 256
        
        # Encoders
        self.molclr_proj = nn.Linear(molclr_dim, self.dim)
        self.chemberta_proj = nn.Linear(384, self.dim)
        self.esm2_proj = nn.Linear(1280, self.dim)
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim)
        
        # Interaction
        self.ugca_drug = UGCA(self.dim)
        self.ugca_prot = UGCA(self.dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim // 2)
        )
        self.head = EvidentialHead(self.dim // 2)
        
    def forward(self, batch):
        # 1. Drug Features
        d1 = self.molclr_proj(batch['molclr'])
        d2 = self.chemberta_proj(batch['chemberta'])
        h_d = d1 + d2
        
        # 2. Protein Features
        p1 = self.esm2_proj(batch['esm2'])
        p2 = self.pocket_proc(batch['graph'])
        h_p = p1 + p2
        
        # 3. UGCA
        z_d = self.ugca_drug(h_d, h_p)
        z_p = self.ugca_prot(h_p, h_d)
        
        # 4. Predict
        feat = self.fusion(torch.cat([z_d, z_p], dim=-1))
        alpha, beta = self.head(feat)
        
        return alpha, beta