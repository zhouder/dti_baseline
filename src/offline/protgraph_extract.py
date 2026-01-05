# -*- coding: utf-8 -*-
import os, argparse, hashlib, re, numpy as np, pandas as pd

def sha(s: str): return hashlib.sha1(s.encode('utf-8')).hexdigest()[:24]

def load_coords(pdb_path):
    """返回 (n,3) 的 Cα 坐标；优先 Biopython，失败则文本兜底。"""
    # 兜底的轻量文本解析
    def _fallback():
        xs=[]
        try:
            with open(pdb_path,'r') as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip()=="CA":
                        x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                        xs.append((x,y,z))
        except Exception:
            return np.zeros((0,3), dtype=np.float32)
        return np.asarray(xs, dtype=np.float32)

    try:
        from Bio.PDB import PDBParser
        parser=PDBParser(QUIET=True)
        st=parser.get_structure('x', pdb_path)
        xs=[]
        for m in st:
            for ch in m:
                for res in ch:
                    if 'CA' in res: xs.append(res['CA'].coord)
        return np.asarray(xs, dtype=np.float32) if xs else _fallback()
    except Exception:
        return _fallback()

def edges(coords, thr=8.0):
    """基于 Cα 8Å 的无向图，返回 (m,2) 的边索引（去除自环）。"""
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    d2 = np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
    mask = d2 <= (thr * thr)
    np.fill_diagonal(mask, False)
    src, dst = np.where(mask)
    if src.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([src, dst], axis=1).astype(np.int64)

def try_pyg(coords, dim=256):
    """若安装了 PyG，用极简 GAT 读出一个全局向量；失败返回 None。"""
    try:
        import torch, torch.nn as nn
        from torch_geometric.data import Data
        from torch_geometric.nn import GATv2Conv
        if len(coords)==0: return None
        e = edges(coords)
        if len(e)==0: return None
        x = torch.tensor(coords, dtype=torch.float32)
        edge_index = torch.tensor(e.T, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        class G(nn.Module):
            def __init__(self, d=256):
                super().__init__()
                self.emb=nn.Linear(3,64)
                self.g1=GATv2Conv(64,128, heads=2, concat=True)
                self.g2=GATv2Conv(256,128, heads=2, concat=False)
                self.fc=nn.Linear(128,d)
            def forward(self, data):
                x=self.emb(data.x)
                x=self.g1(x,data.edge_index).relu()
                x=self.g2(x,data.edge_index).relu()
                return self.fc(x).mean(dim=0)

        g=G(dim).eval()
        with torch.no_grad():
            v = g(data).detach().cpu().numpy().astype(np.float16)
        return v
    except Exception:
        return None

def handcrafted(coords, dim=256):
    """手工特征：距离直方图 + 度直方图 + 几何统计；无 SciPy 亦可运行。"""
    v=np.zeros((dim,), dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if n>0:
        # 距离分布
        try:
            from scipy.spatial.distance import pdist
            ds=pdist(coords, 'euclidean')
        except Exception:
            iu = np.triu_indices(n, 1)
            dif = coords[iu[0]] - coords[iu[1]]
            ds = np.sqrt(np.sum(dif*dif, axis=1))
        hist,_=np.histogram(ds, bins=64, range=(0,32.0))
        v[:64]=hist/(ds.size+1e-6)
        # 度分布（8Å 图）
        e=edges(coords, thr=8.0)
        deg=np.bincount(e[:,0], minlength=n) if e.shape[0]>0 else np.zeros(n, dtype=np.int64)
        hist2,_=np.histogram(deg, bins=64, range=(0,64))
        v[64:128]=hist2/(deg.size+1e-6)
        # 均值/方差
        mu=coords.mean(axis=0); std=coords.std(axis=0)+1e-6
        v[128:131]=mu; v[131:134]=std
    return v.astype(np.float16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--struct-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dim', type=int, default=256)
    args=ap.parse_args()
    os.makedirs(args.out-dir if hasattr(args,'out-dir') else args.out_dir, exist_ok=True)  # ensure dir

    df=pd.read_csv(args.csv)
    prots=sorted(set(df['protein'].astype(str).tolist()))
    for p in prots:
        pid=sha(p); out=os.path.join(args.out_dir, f'{pid}.npz')
        if os.path.exists(out): continue
        pdb=os.path.join(args.struct_dir, f'{pid}.pdb')
        coords=load_coords(pdb)
        vec=try_pyg(coords, dim=args.dim)
        if vec is None: vec=handcrafted(coords, dim=args.dim)
        np.savez_compressed(out, vec=vec)  # DataModule 会读 'vec' 键
        print('wrote', out)

if __name__=='__main__':
    main()
