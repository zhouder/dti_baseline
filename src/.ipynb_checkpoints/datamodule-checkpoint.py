import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# [移除 RDKit 引用，防止报错]
# from rdkit import Chem 

class DTIDataset(Dataset):
    def __init__(self, df, root_dir, dataset_name, cache=False, verbose=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        
        base = os.path.join(root_dir, dataset_name)
        self.h5_path = os.path.join(base, f"{dataset_name}_data.h5")
        self.use_h5 = os.path.exists(self.h5_path)
        
        if verbose:
            if self.use_h5:
                print(f">>> Data Source: HDF5 ({self.h5_path})")
            else:
                print(f">>> Data Source: Raw Files (Slow) at {base}")

        if not self.use_h5:
            self.paths = {
                'molclr': os.path.join(base, 'molclr'),
                'chemberta': os.path.join(base, 'chemberta'),
                'esm2': os.path.join(base, 'esm2'),
                'pocket': os.path.join(base, 'pocket_graph')
            }
        self.h5_file = None

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')

    def _get_from_h5(self, did, pid):
        self._open_h5()
        try:
            d = self.h5_file['drugs'][did]
            molclr = torch.from_numpy(d['molclr'][()]).float()
            chemberta = torch.from_numpy(d['chemberta'][()]).float()
        except:
            molclr = torch.zeros(300); chemberta = torch.zeros(384)

        try:
            p = self.h5_file['proteins'][pid]
            esm2 = torch.from_numpy(p['esm2'][()]).float()
            g = p['pocket']
            graph = Data(
                node_s=torch.from_numpy(g['node_s'][()]).float(),
                node_v=torch.from_numpy(g['node_v'][()]).float(),
                edge_index=torch.from_numpy(g['edge_index'][()]).long(),
                edge_s=torch.from_numpy(g['edge_s'][()]).float(),
                edge_v=torch.from_numpy(g['edge_v'][()]).float()
            )
        except:
            esm2 = torch.zeros(1280)
            graph = Data(node_s=torch.zeros(1,1), edge_index=torch.zeros(2,0).long())
            
        return molclr, chemberta, esm2, graph

    def _get_from_files(self, did, pid):
        try:
            m = np.load(os.path.join(self.paths['molclr'], f"{did}.npy"))
            if m.ndim == 2: m = np.mean(m, axis=0)
            molclr = torch.from_numpy(m).float()
        except: molclr = torch.zeros(300)
        
        try:
            c = np.load(os.path.join(self.paths['chemberta'], f"{did}.npy"))
            if c.ndim == 2: c = np.mean(c, axis=0)
            chemberta = torch.from_numpy(c).float()
        except: chemberta = torch.zeros(384)
        
        try:
            e_dat = np.load(os.path.join(self.paths['esm2'], f"{pid}.npz"))
            k = e_dat.files[0]
            e = e_dat[k]
            if e.ndim == 2: e = np.mean(e, axis=0)
            esm2 = torch.from_numpy(e).float()
        except: esm2 = torch.zeros(1280)
        
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
            'molclr': molclr, 'chemberta': chemberta, 'esm2': esm2, 'graph': graph,
            'label': torch.tensor(label, dtype=torch.float)
        }
    
    def __del__(self):
        if self.h5_file is not None:
            try: self.h5_file.close()
            except: pass

def collate_fn(batch):
    return {
        'molclr': torch.stack([b['molclr'] for b in batch]),
        'chemberta': torch.stack([b['chemberta'] for b in batch]),
        'esm2': torch.stack([b['esm2'] for b in batch]),
        'graph': Batch.from_data_list([b['graph'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch])
    }

def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                      collate_fn=collate_fn, pin_memory=True, persistent_workers=True)