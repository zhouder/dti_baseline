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
    Strict Baseline Alignment:
    - Outer Split: GroupKFold (Deterministic, NO SHUFFLE). 
      This mimics Baseline's outer loop which respects CSV order.
    - Inner Split: Random Shuffle with specific seed offset.
      This mimics Baseline's `rng = np.random.default_rng(seed + 3000 + fold)`.
    """
    indices = np.arange(len(df))
    outer_folds = []
    
    # 1. Outer Split: GroupKFold (No Shuffle)
    if mode == 'warm':
        # Baseline warm uses StratifiedKFold or KFold with shuffle=True
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices))
        
    elif mode in ['cold-drug', 'cold-protein', 'cold-both']:
        # Baseline cold uses GroupKFold directly (No Shuffle arg available/used)
        if mode == 'cold-drug': groups = df['did'].values
        elif mode == 'cold-protein': groups = df['pid'].values
        elif mode == 'cold-both': groups = df['did'].values # Baseline uses drug priority
            
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=groups))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Inner Split: Train -> Train + Val
    final_splits = []
    
    # Baseline logic uses `overall_val=0.1`.
    # Val size inside train pool = 0.1 / (1 - 1/5) = 0.1 / 0.8 = 0.125
    val_frac_in_pool = 0.125

    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        baseline_fold_num = fold_i + 1 # Baseline 1-based fold index
        temp_df = df.iloc[temp_train_indices]
        
        inner_groups = None
        if mode == 'cold-drug': inner_groups = temp_df['did'].values
        elif mode == 'cold-protein': inner_groups = temp_df['pid'].values
        elif mode == 'cold-both': inner_groups = temp_df['did'].values
            
        if inner_groups is not None:
            unique_inner = np.unique(inner_groups)
            
            # --- CRITICAL: Match Baseline Seeding Logic ---
            # Baseline: rng = np.random.default_rng(seed + 3000 + fold) for cold-drug
            # Baseline: rng = np.random.default_rng(seed + 2000 + fold) for cold-protein
            if mode == 'cold-protein':
                offset = 2000
            else:
                offset = 3000 # drug & pair & both use 3000/4000/5000, assuming drug here
            
            magic_seed = seed + offset + baseline_fold_num
            rng = np.random.RandomState(magic_seed)
            
            # Shuffle unique groups (This is the only randomness in Cold mode)
            rng.shuffle(unique_inner)
            
            # Select Val groups
            n_val = max(1, int(round(val_frac_in_pool * len(unique_inner))))
            val_groups = set(unique_inner[:n_val])
            
            is_val = np.array([g in val_groups for g in inner_groups])
            train_idx = temp_train_indices[~is_val]
            val_idx = temp_train_indices[is_val]
        else:
            # Warm mode inner split
            magic_seed = seed + 1000 + baseline_fold_num
            kf_inner = KFold(n_splits=8, shuffle=True, random_state=magic_seed)
            train_sub, val_sub = next(kf_inner.split(temp_train_indices))
            train_idx = temp_train_indices[train_sub]
            val_idx = temp_train_indices[val_sub]
            
        final_splits.append((train_idx, val_idx, test_indices))
        
    return final_splits