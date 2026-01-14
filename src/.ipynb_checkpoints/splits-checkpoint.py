import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

def generate_ids(df):
    """
    Generate SHA1 IDs for drugs and proteins.
    """
    if 'smiles' not in df.columns or 'seq' not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'seq' columns")

    def get_hash(text):
        if pd.isna(text) or text == '':
            return 'unknown'
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:24]

    # Use copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['did'] = df['smiles'].apply(get_hash)
    df['pid'] = df['seq'].apply(get_hash)
    return df

def get_kfold_indices(df, mode='warm', n_splits=5, seed=42):
    """
    Returns iterator of (train_idx, val_idx, test_idx).
    Ratio -> Train:Val:Test = 7:1:2
    """
    indices = np.arange(len(df))
    
    # 1. Outer Split (For Test Set)
    if mode == 'warm':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices))
        
    elif mode == 'cold-drug':
        groups = df['did'].values
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=groups))
        
    elif mode == 'cold-protein':
        groups = df['pid'].values
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=groups))
        
    elif mode == 'cold-both':
        print("Note: Cold-Both mode uses Drug grouping primarily.")
        groups = df['did'].values
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=groups))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Inner Split (For Validation)
    final_splits = []
    
    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        temp_df = df.iloc[temp_train_indices]
        temp_train_inner_idx = np.arange(len(temp_train_indices))
        
        inner_groups = None
        if mode == 'cold-drug':
            inner_groups = temp_df['did'].values
        elif mode == 'cold-protein':
            inner_groups = temp_df['pid'].values
            
        # Split 1/8 for Validation (resulting in 10% of total data)
        if inner_groups is not None:
            gkf_inner = GroupKFold(n_splits=8)
            train_inner_mask, val_inner_mask = next(gkf_inner.split(temp_train_inner_idx, groups=inner_groups))
        else:
            kf_inner = KFold(n_splits=8, shuffle=True, random_state=seed)
            train_inner_mask, val_inner_mask = next(kf_inner.split(temp_train_inner_idx))
            
        train_idx = temp_train_indices[train_inner_mask]
        val_idx = temp_train_indices[val_inner_mask]
        
        final_splits.append((train_idx, val_idx, test_indices))
        
    return final_splits