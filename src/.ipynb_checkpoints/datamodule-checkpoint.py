import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

class DTIDataset(Dataset):
    def __init__(self, df, root_dir, dataset_name, cache=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        
        # 路径设置
        base = os.path.join(root_dir, dataset_name)
        self.h5_path = os.path.join(base, f"{dataset_name}_data.h5")
        
        # 检查 HDF5 是否存在
        self.use_h5 = os.path.exists(self.h5_path)
        
        if self.use_h5:
            print(f"Found HDF5 file: {self.h5_path}. Using accelerated HDF5 loader.")
        else:
            print(f"HDF5 not found at {self.h5_path}. Using slow file loader (Recommend running preprocess_hdf5.py).")
            self.paths = {
                'molclr': os.path.join(base, 'molclr'),
                'chemberta': os.path.join(base, 'chemberta'),
                'esm2': os.path.join(base, 'esm2'),
                'pocket': os.path.join(base, 'pocket_graph')
            }

        # 重要：不要在 __init__ 中打开 h5py 文件！
        # 必须在 worker 进程中懒加载，否则 num_workers > 0 时会报错或死锁。
        self.h5_file = None

    def _open_h5(self):
        """Worker 进程专用的文件打开方法"""
        if self.h5_file is None:
            # swmr=True (Single Writer Multiple Reader) 模式更安全，但通常 'r' 就够了
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')

    def _get_from_h5(self, did, pid):
        # 确保当前进程已打开文件
        self._open_h5()
        
        # 1. 读取药物特征
        # 使用 [()] 读取 Dataset 内容为 Numpy 数组
        try:
            d_grp = self.h5_file['drugs'][did]
            molclr = torch.from_numpy(d_grp['molclr'][()]).float()
            chemberta = torch.from_numpy(d_grp['chemberta'][()]).float()
        except KeyError:
            # 容错：如果 ID 找不到
            molclr = torch.zeros(300)
            chemberta = torch.zeros(384)

        # 2. 读取蛋白特征
        try:
            p_grp = self.h5_file['proteins'][pid]
            esm2 = torch.from_numpy(p_grp['esm2'][()]).float()
            
            # 读取 Graph
            g_grp = p_grp['pocket']
            graph = Data(
                node_s=torch.from_numpy(g_grp['node_s'][()]).float(),
                node_v=torch.from_numpy(g_grp['node_v'][()]).float(),
                edge_index=torch.from_numpy(g_grp['edge_index'][()]).long(),
                edge_s=torch.from_numpy(g_grp['edge_s'][()]).float(),
                edge_v=torch.from_numpy(g_grp['edge_v'][()]).float()
            )
        except KeyError:
            esm2 = torch.zeros(1280)
            graph = Data(node_s=torch.zeros(1,1), edge_index=torch.zeros(2,0).long())
            
        return molclr, chemberta, esm2, graph

    def _get_from_files(self, did, pid):
        """旧的文件读取逻辑 (Fallback)"""
        # MolCLR
        try:
            m = np.load(os.path.join(self.paths['molclr'], f"{did}.npy"))
            if m.ndim == 2: m = np.mean(m, axis=0)
            molclr = torch.from_numpy(m).float()
        except: molclr = torch.zeros(300)
        # ChemBERTa
        try:
            c = np.load(os.path.join(self.paths['chemberta'], f"{did}.npy"))
            if c.ndim == 2: c = np.mean(c, axis=0)
            chemberta = torch.from_numpy(c).float()
        except: chemberta = torch.zeros(384)
        # ESM2
        try:
            e_dat = np.load(os.path.join(self.paths['esm2'], f"{pid}.npz"))
            k = e_dat.files[0]
            e = e_dat[k]
            if e.ndim == 2: e = np.mean(e, axis=0)
            esm2 = torch.from_numpy(e).float()
        except: esm2 = torch.zeros(1280)
        # Graph
        try:
            pg = np.load(os.path.join(self.paths['pocket'], f"{pid}.npz"))
            graph = Data(
                node_s=torch.from_numpy(pg['node_s']).float(),
                node_v=torch.from_numpy(pg['node_v']).float(),
                edge_index=torch.from_numpy(pg['edge_index']).long(),
                edge_s=torch.from_numpy(pg['edge_s']).float(),
                edge_v=torch.from_numpy(pg['edge_v']).float()
            )
        except:
             graph = Data(node_s=torch.zeros(1,1), edge_index=torch.zeros(2,0).long())
        
        return molclr, chemberta, esm2, graph

    def check_integrity(self):
        if self.use_h5: return True
        # 简单的文件检查
        check_limit = min(5, len(self.df))
        for i in range(check_limit):
            row = self.df.iloc[i]
            if not os.path.exists(os.path.join(self.paths['molclr'], f"{row['did']}.npy")): return False
        return True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        did, pid = row['did'], row['pid']
        label = float(row['label'])

        if self.use_h5:
            molclr, chemberta, esm2, graph = self._get_from_h5(did, pid)
        else:
            molclr, chemberta, esm2, graph = self._get_from_files(did, pid)

        return {
            'molclr': molclr, 
            'chemberta': chemberta, 
            'esm2': esm2, 
            'graph': graph, 
            'label': torch.tensor(label, dtype=torch.float)
        }
    
    # 当 Dataset 销毁时关闭文件句柄 (虽然 Python GC 会处理，但显式关闭是个好习惯)
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def collate_fn(batch):
    return {
        'molclr': torch.stack([b['molclr'] for b in batch]),
        'chemberta': torch.stack([b['chemberta'] for b in batch]),
        'esm2': torch.stack([b['esm2'] for b in batch]),
        'graph': Batch.from_data_list([b['graph'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch])
    }

def get_dataloader(df, root_dir, dataset_name, batch_size=32, shuffle=True, num_workers=8, cache=False):
    ds = DTIDataset(df, root_dir, dataset_name)
    # 开启 persistent_workers 可以让 worker 进程保持存活，
    # 避免每个 Epoch 重新打开 HDF5 文件的开销，这对 HDF5 提速非常关键。
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )