import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        # ----------------------------------------
        # 1. Drug Branch (GIN Graph Neural Network)
        # ----------------------------------------
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # Input: (Batch_Nodes, 78) -> Output: (Batch_Nodes, 32)
        nn1 = nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = nn.Linear(dim, output_dim)

        # ----------------------------------------
        # 2. Protein Branch (1D CNN)
        # ----------------------------------------
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        # Conv Layer 1
        # Input: (Batch, 128, 1000) -> 128 channels, 1000 length
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)

        # Conv Layer 2
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)

        # Conv Layer 3
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 3, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)

        # Flatten calculation:
        # Len=1000 -> C1(k=8):993 -> P1(3):331
        #          -> C2(k=8):324 -> P2(3):108
        #          -> C3(k=8):101 -> P3(3):33
        # Output: 32*3 (channels) * 33 (len) = 96 * 33 = 3168
        self.fc1_xt = nn.Linear(3168, output_dim)

        # ----------------------------------------
        # 3. Fusion Head
        # ----------------------------------------
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # --- Drug ---
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        # Global Pooling (Batch aggregation)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # --- Protein ---
        # 【关键修复】: 将 PyG 拼平的 target (Batch*1000) 还原为 (Batch, 1000)
        target = data.target
        # 假设最大序列长度固定为 1000 (baseline_data.py 里定义的)
        # 这里的 view(-1, 1000) 会自动根据数据总量推断 Batch Size
        target = target.view(-1, 1000)

        embedded_xt = self.embedding_xt(target)  # [Batch, 1000, 128]

        # Conv1d 需要 (Batch, Channel, Length) -> (Batch, 128, 1000)
        conv_xt = embedded_xt.permute(0, 2, 1)

        conv_xt = self.conv_xt_1(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)

        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)

        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # Flatten
        xt = conv_xt.view(-1, 32 * 3 * 33)
        xt = F.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)

        # --- Fusion ---
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out