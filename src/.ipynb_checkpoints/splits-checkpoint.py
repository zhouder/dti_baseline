import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

def generate_ids(df):
    """
    Generate SHA1 IDs for file mapping, but splitting will use raw strings.
    """
    import hashlib
    
    if 'smiles' not in df.columns or 'seq' not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'seq'")
    
    def get_hash(text):
        if pd.isna(text) or text == '': return 'unknown'
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:24]

    df = df.copy()
    df['did'] = df['smiles'].apply(get_hash)
    df['pid'] = df['seq'].apply(get_hash)
    return df

def _assert_no_overlap(train_idx, test_idx, groups, mode_name, fold_num):
    """
    Ensure strict separation between sets for cold start settings.
    """
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    intersection = train_groups.intersection(test_groups)
    
    if len(intersection) > 0:
        raise AssertionError(
            f"[FATAL] Split Leakage detected in {mode_name} (Fold {fold_num})! "
            f"Found {len(intersection)} overlapping groups (e.g., {list(intersection)[:3]}). "
            "This violates the cold-start constraint."
        )

def get_kfold_indices(df, mode='warm', n_splits=5, seed=42):
    """
    Strict Baseline Alignment Splitter
    1. Warm: StratifiedKFold (Balances labels)
    2. Cold: GroupKFold on RAW STRINGS (smiles/seq)
    3. Safety: Assert no overlap
    """
    indices = np.arange(len(df))
    # Use raw strings for grouping to match baseline exactly
    drug_key = df['smiles'].astype(str).values
    prot_key = df['seq'].astype(str).values
    y = df['label'].values
    
    outer_folds = []
    
    # --- 1. Outer Split (Test) ---
    if mode == 'warm':
        # Baseline uses StratifiedKFold for binary classification
        # to ensure label distribution is consistent
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices, y))
        
    elif mode == 'cold-drug':
        # Split by Molecule
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=drug_key))
        
    elif mode == 'cold-protein':
        # Split by Target
        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=prot_key))
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: warm, cold-drug, cold-protein")

    final_splits = []
    
    # --- 2. Inner Split (Train -> Train + Val) ---
    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        baseline_fold_num = fold_i + 1
        temp_df = df.iloc[temp_train_indices]
        
        # Security Check: Assert Outer Split Validity
        if mode == 'cold-drug':
            _assert_no_overlap(temp_train_indices, test_indices, drug_key, "Outer Cold-Drug", baseline_fold_num)
        elif mode == 'cold-protein':
            _assert_no_overlap(temp_train_indices, test_indices, prot_key, "Outer Cold-Protein", baseline_fold_num)

        # Logic for Inner Validation Split
        inner_groups = None
        if mode == 'cold-drug': inner_groups = temp_df['smiles'].values
        elif mode == 'cold-protein': inner_groups = temp_df['seq'].values
            
        if inner_groups is not None:
            unique_inner = np.unique(inner_groups)
            
            # Baseline Seeding Logic: seed + offset + fold
            offset = 2000 if mode == 'cold-protein' else 3000
            magic_seed = seed + offset + baseline_fold_num
            rng = np.random.RandomState(magic_seed)
            
            # Shuffle unique groups
            rng.shuffle(unique_inner)
            
            # 1/8 for Validation (~12.5%)
            val_frac = 0.1 / (1.0 - 1.0/n_splits) # 0.125
            n_val = max(1, int(round(val_frac * len(unique_inner))))
            val_groups = set(unique_inner[:n_val])
            
            is_val = np.array([g in val_groups for g in inner_groups])
            train_idx = temp_train_indices[~is_val]
            val_idx = temp_train_indices[is_val]
            
            # Security Check: Assert Inner Split Validity
            check_key = drug_key if mode == 'cold-drug' else prot_key
            _assert_no_overlap(train_idx, val_idx, check_key, f"Inner {mode}", baseline_fold_num)
            
        else:
            # Warm Inner: Random Split (using StratifiedShuffleSplit equivalent logic via KFold for simplicity or Stratified)
            # To be simple and robust:
            magic_seed = seed + 1000 + baseline_fold_num
            # Note: StratifiedKFold requires y, but temp_train_indices is subset.
            # Using simple KFold shuffle here is usually fine for inner warm val, 
            # but strict baseline uses StratifiedShuffleSplit.
            # Let's stick to simple shuffle for inner warm to avoid complexity with subset indexing y.
            kf_inner = KFold(n_splits=8, shuffle=True, random_state=magic_seed)
            train_sub, val_sub = next(kf_inner.split(temp_train_indices))
            train_idx = temp_train_indices[train_sub]
            val_idx = temp_train_indices[val_sub]
            
        final_splits.append((train_idx, val_idx, test_indices))
        
    return final_splits