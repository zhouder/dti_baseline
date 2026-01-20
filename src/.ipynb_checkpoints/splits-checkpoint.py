import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

def generate_ids(df):
    if 'smiles' not in df.columns or 'seq' not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'seq'")
    
    def get_hash(text):
        if pd.isna(text) or text == '': return 'unknown'
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:24]

    df = df.copy()
    df['did'] = df['smiles'].apply(get_hash)
    df['pid'] = df['seq'].apply(get_hash)
    return df

def get_kfold_indices(df, mode='warm', n_splits=5, seed=42):
    """
    [回溯版本] 使用 Shuffle + Split 逻辑，解决 Fold 4 偏差问题。
    """
    # Fix seed
    rng = np.random.RandomState(seed)
    indices = np.arange(len(df))
    outer_folds = []
    
    if mode == 'warm':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices))
        
    elif mode in ['cold-drug', 'cold-protein', 'cold-both']:
        # 1. Get Groups
        col = 'did' if mode == 'cold-drug' else 'pid'
        if mode == 'cold-both': col = 'did'
        
        unique_groups = df[col].unique()
        
        # 2. SHUFFLE (关键步骤: 打乱组顺序)
        rng.shuffle(unique_groups)
        
        # 3. Split
        group_chunks = np.array_split(unique_groups, n_splits)
        
        for i in range(n_splits):
            test_groups = group_chunks[i]
            is_test = df[col].isin(test_groups)
            test_indices = indices[is_test]
            train_indices = indices[~is_test]
            outer_folds.append((train_indices, test_indices))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Inner Validation Split
    final_splits = []
    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        temp_df = df.iloc[temp_train_indices]
        
        inner_col = None
        if mode == 'cold-drug': inner_col = 'did'
        elif mode == 'cold-protein': inner_col = 'pid'
            
        if inner_col:
            inner_groups = temp_df[inner_col].values
            unique_inner = np.unique(inner_groups)
            rng.shuffle(unique_inner)
            
            # 1/8 Validation
            n_val = max(1, len(unique_inner) // 8)
            val_groups = unique_inner[:n_val]
            
            is_val = np.isin(inner_groups, val_groups)
            train_idx = temp_train_indices[~is_val]
            val_idx = temp_train_indices[is_val]
        else:
            kf_inner = KFold(n_splits=8, shuffle=True, random_state=seed)
            train_sub, val_sub = next(kf_inner.split(temp_train_indices))
            train_idx = temp_train_indices[train_sub]
            val_idx = temp_train_indices[val_sub]
            
        final_splits.append((train_idx, val_idx, test_indices))
        
    return final_splits