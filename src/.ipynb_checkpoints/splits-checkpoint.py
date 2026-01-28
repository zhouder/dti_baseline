import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    StratifiedShuffleSplit,
)
import hashlib


def generate_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "smiles" not in df.columns or "seq" not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'seq'")

    def get_hash(text):
        if pd.isna(text) or text == "":
            return "unknown"
        return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:24]

    df = df.copy()
    df["did"] = df["smiles"].apply(get_hash)
    df["pid"] = df["seq"].apply(get_hash)
    return df


def _assert_no_overlap(a_idx, b_idx, groups, mode_name, fold_num):
    a = set(groups[np.asarray(a_idx, dtype=int)])
    b = set(groups[np.asarray(b_idx, dtype=int)])
    inter = a.intersection(b)
    if inter:
        raise AssertionError(
            f"[FATAL] Split Leakage detected in {mode_name} (Fold {fold_num})! "
            f"Found {len(inter)} overlapping groups. "
            "This violates the cold-start constraint."
        )


def _group_pos_rate(df: pd.DataFrame, group_col: str, label_col: str = "label") -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      group, pos_rate, count
    """
    g = df.groupby(group_col)[label_col].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={group_col: "group", "mean": "pos_rate", "count": "count"})
    g["group"] = g["group"].astype(str)
    return g


def _outer_splits_cold_by_group_shuffle(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Shuffle unique groups then assign with round-robin (sizes differ <=1).
    Returns list of (train_pool_idx, test_idx).
    """
    groups = groups.astype(str)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    folds = [uniq[k::n_splits] for k in range(n_splits)]

    all_idx = np.arange(len(groups))
    out = []
    for k in range(n_splits):
        test_groups = set(folds[k].tolist())
        te_mask = np.array([g in test_groups for g in groups], dtype=bool)
        te_idx = all_idx[te_mask]
        tr_pool = all_idx[~te_mask]
        out.append((tr_pool, te_idx))
    return out


def _outer_splits_cold_by_group_stratified(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
    seed: int,
    n_bins: int = 8,
    label_col: str = "label",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Stratify groups by their pos_rate:
      1) compute pos_rate per group
      2) bin pos_rate into quantile bins
      3) StratifiedKFold over groups (not rows)
    Returns list of (train_pool_idx, test_idx).
    """
    # group stats
    gdf = _group_pos_rate(df, group_col=group_col, label_col=label_col)
    uniq_groups = gdf["group"].values
    pos_rate = gdf["pos_rate"].values.astype(float)

    # build bins by quantiles
    n_groups = len(uniq_groups)
    nb = int(max(2, min(n_bins, n_groups)))  # safe
    qs = np.linspace(0.0, 1.0, nb + 1)
    edges = np.quantile(pos_rate, qs)
    edges = np.unique(edges)

    # if edges collapse (many identical rates), fallback to shuffle
    if len(edges) <= 2:
        return _outer_splits_cold_by_group_shuffle(
            groups=df[group_col].astype(str).values,
            n_splits=n_splits,
            seed=seed,
        )

    # digitize into bins (0..B-1)
    # edges[1:-1] are interior cut points
    bins = np.digitize(pos_rate, edges[1:-1], right=True)

    # if any bin is too small for StratifiedKFold, fallback to shuffle
    bincount = np.bincount(bins, minlength=bins.max() + 1)
    if np.any(bincount < n_splits):
        return _outer_splits_cold_by_group_shuffle(
            groups=df[group_col].astype(str).values,
            n_splits=n_splits,
            seed=seed,
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_idx = np.arange(n_groups)

    folds = []
    for _, te_g in skf.split(group_idx, bins):
        folds.append(uniq_groups[te_g])

    groups_all = df[group_col].astype(str).values
    all_idx = np.arange(len(groups_all))
    out = []
    for k in range(n_splits):
        test_groups = set(folds[k].tolist())
        te_mask = np.array([g in test_groups for g in groups_all], dtype=bool)
        te_idx = all_idx[te_mask]
        tr_pool = all_idx[~te_mask]
        out.append((tr_pool, te_idx))
    return out


def get_kfold_indices(
    df: pd.DataFrame,
    mode: str = "warm",
    n_splits: int = 5,
    seed: int = 42,
    overall_val: float = 0.1,
    cold_outer: str = "shuffle",  # "groupkfold" | "shuffle" | "stratified"
    cold_bins: int = 8,              # for stratified mode
):
    """
    Strict splitter for DTI with warm/cold settings.

    Outer (test):
      - warm/hot: StratifiedKFold on labels (rows)
      - cold-drug: group split on smiles (rows grouped by smiles)
      - cold-protein: group split on seq
        cold_outer:
          * groupkfold  : sklearn GroupKFold (no shuffle)
          * shuffle      : shuffle unique groups then round-robin assign
          * stratified  : stratify groups by per-group pos_rate bins then split

    Inner (train_pool -> train/val):
      - warm/hot: StratifiedShuffleSplit within pool
      - cold-*: sample a fraction of unique groups as val groups (strict group separation)

    Seed policy:
      - inner split uses seed + 100 + fold_num  (no offset)
    """
    if not {"smiles", "seq", "label"}.issubset(df.columns):
        raise ValueError("df must contain columns: smiles, seq, label")

    indices = np.arange(len(df))
    drug_key = df["smiles"].astype(str).values
    prot_key = df["seq"].astype(str).values
    y = df["label"].values

    if mode == "hot":
        mode = "warm"

    # MolTrans-style: val_frac_in_pool = overall_val / (1 - 1/K)
    val_frac = float(overall_val) / (1.0 - 1.0 / n_splits)

    # ---------- outer folds ----------
    if mode == "warm":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices, y))  # (train_pool, test)
    elif mode == "cold-drug":
        if cold_outer == "groupkfold":
            gkf = GroupKFold(n_splits=n_splits)
            outer_folds = list(gkf.split(indices, groups=drug_key))
        elif cold_outer == "shuffle":
            outer_folds = _outer_splits_cold_by_group_shuffle(drug_key, n_splits, seed)
        elif cold_outer == "stratified":
            outer_folds = _outer_splits_cold_by_group_stratified(
                df=df, group_col="smiles", n_splits=n_splits, seed=seed, n_bins=cold_bins, label_col="label"
            )
        else:
            raise ValueError("cold_outer must be one of: groupkfold, shuffle, stratified")
    elif mode == "cold-protein":
        if cold_outer == "groupkfold":
            gkf = GroupKFold(n_splits=n_splits)
            outer_folds = list(gkf.split(indices, groups=prot_key))
        elif cold_outer == "shuffle":
            outer_folds = _outer_splits_cold_by_group_shuffle(prot_key, n_splits, seed)
        elif cold_outer == "stratified":
            outer_folds = _outer_splits_cold_by_group_stratified(
                df=df, group_col="seq", n_splits=n_splits, seed=seed, n_bins=cold_bins, label_col="label"
            )
        else:
            raise ValueError("cold_outer must be one of: groupkfold, shuffle, stratified")
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: warm, hot, cold-drug, cold-protein")

    # ---------- build final splits with inner val ----------
    final_splits = []
    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        fold_num = fold_i + 1
        temp_train_indices = np.asarray(temp_train_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        # outer leak check
        if mode == "cold-drug":
            _assert_no_overlap(temp_train_indices, test_indices, drug_key, "Outer Cold-Drug", fold_num)
        elif mode == "cold-protein":
            _assert_no_overlap(temp_train_indices, test_indices, prot_key, "Outer Cold-Protein", fold_num)

        # unified inner seed (+100)
        magic_seed = int(seed) + 100 + int(fold_num)

        if mode in ("cold-drug", "cold-protein"):
            # inner group-based val split (strict)
            if mode == "cold-drug":
                inner_groups = df.iloc[temp_train_indices]["smiles"].astype(str).values
                check_key = drug_key
                inner_name = "drug"
            else:
                inner_groups = df.iloc[temp_train_indices]["seq"].astype(str).values
                check_key = prot_key
                inner_name = "protein"

            unique_inner = np.unique(inner_groups)
            rng = np.random.RandomState(magic_seed)
            rng.shuffle(unique_inner)

            n_val = max(1, int(round(val_frac * len(unique_inner))))
            val_groups = set(unique_inner[:n_val])

            is_val = np.array([g in val_groups for g in inner_groups], dtype=bool)
            train_idx = temp_train_indices[~is_val]
            val_idx = temp_train_indices[is_val]

            _assert_no_overlap(train_idx, val_idx, check_key, f"Inner Cold-{inner_name}", fold_num)

        else:
            # warm inner: stratified shuffle split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=magic_seed)
            tr_sub, va_sub = next(sss.split(temp_train_indices, y[temp_train_indices]))
            train_idx = temp_train_indices[tr_sub]
            val_idx = temp_train_indices[va_sub]

        final_splits.append((train_idx, val_idx, test_indices))

    return final_splits