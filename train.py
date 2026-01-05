import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
import shutil
import warnings
import time

warnings.filterwarnings("ignore")

# ==========================================
# 0. 环境与导入修复
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import models
    import stream
    import config
    from config import BIN_config_DBPE
    from stream import BIN_Data_Encoder

    # 自动探测模型类名
    if hasattr(models, 'MolTransModel'):
        MolTransModel = models.MolTransModel
    elif hasattr(models, 'BIN_Interaction_Flat'):
        MolTransModel = models.BIN_Interaction_Flat
    else:
        # 兜底搜索
        candidates = [x for x in dir(models) if
                      isinstance(getattr(models, x), type) and issubclass(getattr(models, x), nn.Module)]
        if 'BIN_Interaction_Flat' in candidates:
            MolTransModel = models.BIN_Interaction_Flat
        else:
            raise ImportError("Could not find model class (BIN_Interaction_Flat) in models.py")

except ImportError as e:
    print(f"\n[Critical Import Error]: {e}")
    raise e


# ==========================================
# 1. 核心指标计算 (与 UGCA 完全一致)
# ==========================================
def compute_metrics(y_true, y_logits):
    y_true = np.array(y_true).flatten()
    y_logits = np.array(y_logits).flatten()
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    best_f1, best_thresh = 0, 0.5
    thresholds = np.arange(0.1, 0.95, 0.05)
    for thresh in thresholds:
        y_pred_tmp = (y_prob > thresh).astype(int)
        f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
        if f1_tmp > best_f1:
            best_f1, best_thresh = f1_tmp, thresh

    y_pred = (y_prob > best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUROC': auroc, 'AUPRC': auprc, 'F1': best_f1, 'Acc': acc,
        'Sens': sensitivity, 'Spec': specificity, 'Prec': precision, 'MCC': mcc,
        'Thresh': best_thresh
    }


def print_metrics(metrics, prefix="Val"):
    print(f"{prefix:<5} | Loss: {metrics['Loss']:.4f} | AUC: {metrics['AUROC']:.4f} | "
          f"AUPRC: {metrics['AUPRC']:.4f} | F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f}")


# ==========================================
# 2. 数据处理与 Dataset
# ==========================================
class MolTransDataset(Dataset):
    def __init__(self, data_dict):
        self.d_v = data_dict['d_v']
        self.d_m = data_dict['d_m']
        self.p_v = data_dict['p_v']
        self.p_m = data_dict['p_m']
        self.y = data_dict['label']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'd_v': torch.tensor(self.d_v[idx]).long(),
            'd_m': torch.tensor(self.d_m[idx]).float(),
            'p_v': torch.tensor(self.p_v[idx]).long(),
            'p_m': torch.tensor(self.p_m[idx]).float(),
            'label': torch.tensor(self.y[idx]).float()
        }


class MolTransDataManager:
    def __init__(self, args):
        self.args = args
        self.data_root = args.data_root
        self.dataset_name = args.dataset
        self.encoder = None

        # 路径回退
        self.espf_path = './ESPF/'
        if not os.path.exists(self.espf_path):
            if os.path.exists('../ESPF/'):
                self.espf_path = '../ESPF/'
            elif os.path.exists(os.path.join(current_dir, 'ESPF')):
                self.espf_path = os.path.join(current_dir, 'ESPF')

    def load_and_encode_all(self):
        csv_path = os.path.join(self.data_root, self.dataset_name, 'all.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 重命名列以适配 MolTrans
        df_moltrans = df.rename(columns={'smiles': 'SMILES', 'protein': 'Target Sequence', 'label': 'Label'})

        # 准备 encoder 所需参数
        list_IDs = df_moltrans.index.values
        labels = {i: l for i, l in zip(list_IDs, df_moltrans['Label'].values)}

        print("Initializing BIN_Data_Encoder...")
        self.encoder = BIN_Data_Encoder(list_IDs, labels, df_moltrans)

        print("Encoding Drugs & Proteins (BPE)...")
        # 使用 encoder 内部方法进行编码
        d_enc_full = self.encoder.obj_enc_d.transform(df['smiles'].values)
        p_enc_full = self.encoder.obj_enc_p.transform(df['protein'].values)

        self.full_data = {
            'd_v': d_enc_full['index'],
            'd_m': d_enc_full['mask'],
            'p_v': p_enc_full['index'],
            'p_m': p_enc_full['mask'],
            'label': df['label'].values,
            'smiles': df['smiles'].values,
            'protein': df['protein'].values
        }
        self.df = df
        print("Encoding Done.")

    def get_kfold_indices(self, fold_idx):
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        df = self.df
        splits = []

        if self.args.split_mode == 'warm' or self.args.split_mode == 'random':
            splits = list(kf.split(df))

        elif self.args.split_mode == 'cold_drug':
            unique_drugs = df['smiles'].unique()
            drug_splits = list(kf.split(unique_drugs))
            for train_d_idx, test_d_idx in drug_splits:
                train_drugs = unique_drugs[train_d_idx]
                test_drugs = unique_drugs[test_d_idx]
                train_idx = df[df['smiles'].isin(train_drugs)].index.to_numpy()
                test_idx = df[df['smiles'].isin(test_drugs)].index.to_numpy()
                splits.append((train_idx, test_idx))

        elif self.args.split_mode == 'cold_protein':
            unique_prots = df['protein'].unique()
            prot_splits = list(kf.split(unique_prots))
            for train_p_idx, test_p_idx in prot_splits:
                train_prots = unique_prots[train_p_idx]
                test_prots = unique_prots[test_p_idx]
                train_idx = df[df['protein'].isin(train_prots)].index.to_numpy()
                test_idx = df[df['protein'].isin(test_prots)].index.to_numpy()
                splits.append((train_idx, test_idx))

        elif self.args.split_mode == 'cold_both':
            splits = list(kf.split(df))

        warm_indices, test_indices = splits[fold_idx]
        np.random.seed(42 + fold_idx)
        np.random.shuffle(warm_indices)
        n_val = int(len(warm_indices) * 0.125)
        val_indices = warm_indices[:n_val]
        train_indices = warm_indices[n_val:]

        return train_indices, val_indices, test_indices

    def get_loader(self, indices, batch_size, shuffle):
        subset_data = {
            'd_v': self.full_data['d_v'][indices],
            'd_m': self.full_data['d_m'][indices],
            'p_v': self.full_data['p_v'][indices],
            'p_m': self.full_data['p_m'][indices],
            'label': self.full_data['label'][indices]
        }
        ds = MolTransDataset(subset_data)
        # [核心修复] drop_last=True，确保 batch_size 始终为 64，避免 MolTrans 内部 shape 错误
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=self.args.num_workers, drop_last=True)


# ==========================================
# 3. Arguments
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="MolTrans Baseline Auto 5-Fold")
    parser.add_argument('--dataset', type=str, default='DAVIS')
    parser.add_argument('--split-mode', type=str, default='warm')
    parser.add_argument('--data-root', type=str, default='/root/lanyun-tmp')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=10)

    return parser.parse_args()


# ==========================================
# 4. Training Loop
# ==========================================
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_logits_list = []
    all_targets_list = []

    pbar = tqdm(loader, desc="Train" if is_train else "Val/Test", leave=False)
    for batch in pbar:
        d_v = batch['d_v'].to(device)
        d_m = batch['d_m'].to(device)
        p_v = batch['p_v'].to(device)
        p_m = batch['p_m'].to(device)
        label = batch['label'].to(device)

        if is_train:
            optimizer.zero_grad()
            logits = model(d_v, p_v, d_m, p_m)
            loss = criterion(logits.squeeze(), label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(d_v, p_v, d_m, p_m)
                loss = criterion(logits.squeeze(), label)

        total_loss += loss.item()
        all_logits_list.append(logits.detach().cpu().numpy())
        all_targets_list.append(label.detach().cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    all_logits = np.concatenate(all_logits_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)
    metrics = compute_metrics(all_targets, all_logits)
    metrics['Loss'] = avg_loss
    return metrics


def train_single_fold(args, fold_idx, dm):
    exp_name = f"{args.dataset}_{args.split_mode}"
    base_dir = os.path.join(args.data_root, 'moltrans-baseline', exp_name)
    fold_dir = os.path.join(base_dir, f"fold{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'=' * 50}\nFold {fold_idx + 1}/5 | Output: {fold_dir}\n{'=' * 50}")

    train_idx, val_idx, test_idx = dm.get_kfold_indices(fold_idx)
    train_loader = dm.get_loader(train_idx, args.batch_size, shuffle=True)
    val_loader = dm.get_loader(val_idx, args.batch_size, shuffle=False)
    test_loader = dm.get_loader(test_idx, args.batch_size, shuffle=False)

    config = BIN_config_DBPE()
    # Correct model initialization
    model = MolTransModel(**config).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch, best_auprc, patience = 0, 0.0, 0
    last_ckpt = os.path.join(fold_dir, 'last.pt')
    best_ckpt = os.path.join(fold_dir, 'best.pt')

    log_path = os.path.join(fold_dir, 'log.csv')
    cols = ["Epoch", "lr", "Train_Loss", "Train_AUC", "Train_AUPRC", "Train_F1", "Train_Acc",
            "Val_Loss", "Val_AUC", "Val_AUPRC", "Val_F1", "Val_Acc", "Val_Sens", "Val_Spec", "Val_Prec", "Val_MCC"]
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f: f.write(",".join(cols) + "\n")

    for epoch in range(args.epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"\nFold {fold_idx + 1} | Ep {epoch + 1}/{args.epochs} | LR: {curr_lr:.2e}")

        train_res = run_epoch(model, train_loader, criterion, optimizer, args.device, is_train=True)
        val_res = run_epoch(model, val_loader, criterion, optimizer, args.device, is_train=False)

        print_metrics(train_res, "Train")
        print_metrics(val_res, "Val  ")

        row = [epoch + 1, curr_lr,
               train_res['Loss'], train_res['AUROC'], train_res['AUPRC'], train_res['F1'], train_res['Acc'],
               val_res['Loss'], val_res['AUROC'], val_res['AUPRC'], val_res['F1'], val_res['Acc'],
               val_res['Sens'], val_res['Spec'], val_res['Prec'], val_res['MCC']]
        with open(log_path, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")

        if val_res['AUPRC'] > best_auprc:
            best_auprc = val_res['AUPRC']
            patience = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f">>> New Best! (AUPRC: {best_auprc:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping.");
                break

    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt))

    test_res = run_epoch(model, test_loader, criterion, optimizer, args.device, is_train=False)
    print("\nTest Results:");
    print_metrics(test_res, "Test ")

    res_df = pd.DataFrame([test_res])
    res_df.to_csv(os.path.join(fold_dir, 'result.csv'), index=False)

    summary_path = os.path.join(base_dir, 'summary.csv')
    fold_res = test_res.copy();
    fold_res['Fold'] = fold_idx + 1
    df_new = pd.DataFrame([fold_res])
    if not os.path.exists(summary_path):
        df_new.to_csv(summary_path, index=False)
    else:
        df_new.to_csv(summary_path, mode='a', header=False, index=False)
    return test_res


def main():
    args = get_args()
    dm = MolTransDataManager(args)
    dm.load_and_encode_all()

    all_metrics = []
    print(f"Starting Auto 5-Fold on {args.dataset} ({args.split_mode})")

    for i in range(5):
        metrics = train_single_fold(args, fold_idx=i, dm=dm)
        all_metrics.append(metrics)

    print("\n" + "=" * 50 + "\nALL FOLDS COMPLETED. FINAL AVERAGED RESULTS:\n" + "=" * 50)
    df = pd.DataFrame(all_metrics)
    mean, std = df.mean(numeric_only=True), df.std(numeric_only=True)

    for k in mean.index:
        if k not in ['Fold', 'Thresh']: print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    exp_name = f"{args.dataset}_{args.split_mode}"
    path = os.path.join(args.data_root, 'moltrans-baseline', exp_name, 'summary_full.csv')
    df.loc['Mean'] = mean;
    df.loc['Std'] = std
    df.to_csv(path)
    print(f"\nFinal Summary saved to: {path}")


if __name__ == '__main__':
    main()