import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

# ==========================================
# 1. Light GVP (图编码器)
# ==========================================
class SimpleGVPConv(MessagePassing):
    """轻量级几何感知卷积，处理标量(s)和向量(v)"""
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
    """将 PocketGraph 编码为向量"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.s_emb = nn.Linear(29, out_dim) 
        self.v_emb = nn.Linear(3, 16)       
        self.e_emb = nn.Linear(5, out_dim)  
        
        self.conv = SimpleGVPConv(out_dim, 16)
        self.out = nn.Sequential(
            nn.Linear(out_dim + 16, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, data):
        s, v, edge_index, edge_s = data.node_s, data.node_v, data.edge_index, data.edge_s
        
        # 动态维度适配
        if s.shape[1] != self.s_emb.in_features: self.s_emb = nn.Linear(s.shape[1], 256).to(s.device)
        if v.shape[1] != self.v_emb.in_features: self.v_emb = nn.Linear(v.shape[1], 16).to(v.device)
        if edge_s.shape[1] != self.e_emb.in_features: self.e_emb = nn.Linear(edge_s.shape[1], 256).to(edge_s.device)

        s = self.s_emb(s)
        edge_s = self.e_emb(edge_s)
        v = self.v_emb(v.transpose(1,2)).transpose(1,2)
        
        s = self.conv(s, v, edge_index, edge_s)
        
        v_norm = torch.norm(v, dim=-1)
        feat = torch.cat([s, v_norm], dim=-1)
        graph_vec = global_mean_pool(feat, data.batch)
        
        return self.out(graph_vec)

# ==========================================
# 2. 升级版 UGCA (基于分布方差)
# ==========================================
class DistributionalUGCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Attention Projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # --- 核心升级: 分布不确定性估计 ---
        # 我们预测特征分布的 对数方差 (Log Variance)
        # 假设特征服从高斯分布，方差越大，信息越模糊(不确定性越高)
        self.log_var_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1) # 输出 log(sigma^2)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_q, x_kv):
        """
        x_q: Query (主模态)
        x_kv: Key/Value (辅助模态)
        """
        B, D = x_q.shape
        
        # 1. 估算不确定性 (Uncertainty Estimation)
        # log_var: (-inf, +inf)
        log_var = self.log_var_estimator(x_q) 
        
        # 将 log_var 转换为方差 sigma^2
        # 使用 Softplus 保证平滑正值，variance 代表不确定性程度
        variance = F.softplus(log_var) 
        
        # 2. 生成门控 (Gating)
        # 逻辑: 方差(不确定性)越大，Gate 越小 (关闭交互)
        # Gate = 1 / (1 + Variance) -> 范围 (0, 1]
        gate = 1.0 / (1.0 + variance)
        
        # 3. Standard Cross-Attention
        q = self.q(x_q).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v(x_kv).reshape(B, 1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1,2).reshape(B, D)
        out = self.out_proj(out)
        
        # 4. 门控残差连接
        # 如果 Gate 很小(不确定)，则抑制 out，保留 x_q
        fused = x_q + self.dropout(out * gate)
        return self.norm1(fused)

# ==========================================
# 3. 主模型架构
# ==========================================
class UGCADTI(nn.Module):
    def __init__(self, molclr_dim=300):
        super().__init__()
        self.dim = 256 # 隐层维度
        
        # --- 编码器 ---
        self.molclr_proj = nn.Sequential(
            nn.Linear(molclr_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU()
        )
        self.chemberta_proj = nn.Sequential(
            nn.Linear(384, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU()
        )
        self.esm2_proj = nn.Sequential(
            nn.Linear(1280, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU()
        )
        self.pocket_proc = PocketGraphProcessor(out_dim=self.dim)
        
        # --- 交互层 (UGCA) ---
        # 使用升级版 DistributionalUGCA
        self.ugca_drug = DistributionalUGCA(self.dim)
        self.ugca_prot = DistributionalUGCA(self.dim)
        
        # --- 预测头 (BCE) ---
        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Dropout(0.2), # 增加 Dropout 防止过拟合
            nn.Linear(self.dim, 1) # 直接输出 Logits
        )
        
    def forward(self, batch):
        # 1. 药物特征融合
        d1 = self.molclr_proj(batch['molclr'])
        d2 = self.chemberta_proj(batch['chemberta'])
        h_d = d1 + d2 
        
        # 2. 蛋白特征融合
        p1 = self.esm2_proj(batch['esm2'])
        p2 = self.pocket_proc(batch['graph'])
        h_p = p1 + p2
        
        # 3. 双向不确定性交互
        z_d = self.ugca_drug(x_q=h_d, x_kv=h_p) # Drug query Prot
        z_p = self.ugca_prot(x_q=h_p, x_kv=h_d) # Prot query Drug
        
        # 4. 预测
        z = torch.cat([z_d, z_p], dim=-1)
        logits = self.fusion(z) # (B, 1)
        
        return logits