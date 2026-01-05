import pandas as pd
import numpy as np
import os
import hashlib
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import KFold
from rdkit import Chem
import networkx as nx

# --- 1. Graph & Feature Utils (保持不变) ---
def atom_features(atom):
    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
               'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
               'Pt', 'Hg', 'Pb', 'X']
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), symbols) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def seq_cat(prot, max_seq_len=1000):
    seq_voc = "ACDEFGHIKLMNPQRSTVWY"
    seq_dict = {v: i for i, v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): x[i] = seq_dict.get(ch, 0)
    return x

# --- 2. Dataset Class (保持不变) ---
class GraphDTADataset(InMemoryDataset):
    def __init__(self, root, df, transform=None, pre_transform=None):
        self.df = df
        super(GraphDTADataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return ['data.pt']
    def download(self): pass

    def process(self):
        print("\n[INFO] Generating 78-dim features for new cache...")
        data_list = []
        for i, row in self.df.iterrows():
            smile = row['smiles']
            target = row['protein']
            label = float(row['label'])
            try:
                c_size, features, edge_index = smile_to_graph(smile)
                x = torch.tensor(np.array(features), dtype=torch.float)
                edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
            except:
                continue
            target_feat = torch.tensor(seq_cat(target), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, target=target_feat, y=torch.tensor([label], dtype=torch.float))
            data.c_size = torch.tensor([c_size], dtype=torch.long)
            data.did = hashlib.sha1(smile.encode()).hexdigest()[:24]
            data.pid = hashlib.sha1(target.encode()).hexdigest()[:24]
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("[INFO] Data generation complete.")

# --- 3. 核心修正: 严格的划分逻辑 ---
def get_kfold_indices(dataset, split_mode='cold-protein', n_splits=5, seed=42):
    all_dids = np.array([d.did for d in dataset])
    all_pids = np.array([d.pid for d in dataset])
    indices = np.arange(len(dataset))
    
    # 获取唯一的 drug 和 protein ID
    unique_dids = np.unique(all_dids)
    unique_pids = np.unique(all_pids)
    
    # 随机打乱 ID 列表
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_dids)
    rng.shuffle(unique_pids)
    
    splits = []
    
    if split_mode == 'warm':
        print(f"Splitting: 5-Fold Warm (Random Split)")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(indices))
        
    elif split_mode == 'cold-drug':
        print(f"Splitting: 5-Fold Cold-Drug (Disjoint Drugs)")
        # 将药物分成 5 份
        kf = KFold(n_splits=n_splits, shuffle=False) # 已预先 shuffle
        for train_did_idx, test_did_idx in kf.split(unique_dids):
            test_dids_set = set(unique_dids[test_did_idx])
            # Test: 包含测试集药物的所有样本
            test_mask = np.array([d in test_dids_set for d in all_dids])
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            splits.append((train_idx, test_idx))
            
    elif split_mode == 'cold-protein':
        print(f"Splitting: 5-Fold Cold-Protein (Disjoint Proteins)")
        # 将蛋白分成 5 份
        kf = KFold(n_splits=n_splits, shuffle=False)
        for train_pid_idx, test_pid_idx in kf.split(unique_pids):
            test_pids_set = set(unique_pids[test_pid_idx])
            # Test: 包含测试集蛋白的所有样本
            test_mask = np.array([p in test_pids_set for p in all_pids])
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            splits.append((train_idx, test_idx))
            
    elif split_mode == 'cold-both':
        print(f"Splitting: 5-Fold STRICT Cold-Both (Disjoint Drugs AND Proteins)")
        # 这是一个极其严格的划分，会丢弃部分数据以防止泄露
        # 逻辑：将 Drug 和 Protein 分别分为 k 份。
        # Fold k 的 Test Set = (Drug_Fold_k) X (Protein_Fold_k)
        # Fold k 的 Train Set = (NOT Drug_Fold_k) X (NOT Protein_Fold_k)
        # 注意：这里会丢弃 (Train_Drug, Test_Protein) 和 (Test_Drug, Train_Protein) 的样本
        
        kf = KFold(n_splits=n_splits, shuffle=False)
        
        # 为了同步 fold，我们手动生成 split indices
        drug_splits = list(kf.split(unique_dids))
        prot_splits = list(kf.split(unique_pids))
        
        for fold in range(n_splits):
            _, test_did_idx = drug_splits[fold]
            _, test_pid_idx = prot_splits[fold]
            
            test_dids_set = set(unique_dids[test_did_idx])
            test_pids_set = set(unique_pids[test_pid_idx])
            
            # 标记每个样本的归属
            is_test_drug = np.array([d in test_dids_set for d in all_dids])
            is_test_prot = np.array([p in test_pids_set for p in all_pids])
            
            # Test Set: 必须同时是 TestDrug 和 TestProt
            test_mask = is_test_drug & is_test_prot
            
            # Train Set: 必须同时是 TrainDrug 和 TrainProt (严格隔离)
            # 即：既不是 TestDrug，也不是 TestProt
            train_mask = (~is_test_drug) & (~is_test_prot)
            
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            
            # 检查是否有数据，防止空 fold
            if len(test_idx) == 0:
                print(f"[WARN] Fold {fold} has EMPTY test set! Dataset too sparse for Cold-Both.")
            
            splits.append((train_idx, test_idx))
            
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
        
    return splits


