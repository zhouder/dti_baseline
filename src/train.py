import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import random
import h5py
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    recall_score,
    matthews_corrcoef
)
from torch.utils.data import Subset

from src.splits import generate_ids, get_kfold_indices
from src.datamodule import get_dataloader, DTIDataset
from src.model import UGCADTI

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs > threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {'AUPRC': 0.0, 'AUROC': 0.0, 'F1': 0.0, 'ACC': 0.0, 'SEN': 0.0, 'MCC': 0.0}
    
    return {
        'AUPRC': average_precision_score(y_true, y_probs),
        'AUROC': roc_auc_score(y_true, y_probs),
        'F1': f1_score(y_true, y_pred),
        'ACC': accuracy_score(y_true, y_pred),
        'SEN': recall_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

def run_epoch(model, loader, criterion, optimizer, device, epoch_idx, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    total_loss = 0
    y_true, y_probs = [], []
    
    mean_gate_d = 0
    mean_gate_p = 0

    pbar = tqdm(loader, leave=False, desc="Train" if is_train else "Val", ncols=100)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train: optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            # 传入 epoch 用于 beta warmup
            logits, g_d, g_p = model(batch, epoch=epoch_idx)
            labels = batch['label'].float().view(-1, 1)
            
            # 1. Main Loss
            loss_bce = criterion(logits, labels)
            
            # 2. Anti-Collapse Regularization (Small weight)
            # 鼓励 Gate 具有一定的方差，避免全员躺平
            if is_train and g_d.shape[0] > 1:
                reg_loss = -0.001 * (g_d.var() + g_p.var())
            else:
                reg_loss = torch.tensor(0.0).to(device)
            
            loss = loss_bce + reg_loss
            
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        y_true.extend(labels.cpu().numpy().flatten())
        y_probs.extend(probs.detach().cpu().numpy().flatten())
        
        mean_gate_d += g_d.mean().item()
        mean_gate_p += g_p.mean().item()
        
        pbar.set_postfix(bce=f"{loss_bce.item():.4f}", gate=f"{g_d.mean().item():.2f}")

    metrics = compute_metrics(np.array(y_true), np.array(y_probs))
    metrics['Loss'] = total_loss / len(loader)
    metrics['Gate_D'] = mean_gate_d / len(loader)
    metrics['Gate_P'] = mean_gate_p / len(loader)
    return metrics

def align_data_with_hdf5(df, h5_path):
    if not os.path.exists(h5_path):
        print(f"Warning: HDF5 not found at {h5_path}. Skipping alignment.")
        return df
    print("Aligning data with HDF5 keys...")
    with h5py.File(h5_path, 'r') as f:
        valid_drugs = set(f['drugs'].keys())
        valid_prots = set(f['proteins'].keys())
    mask = df['did'].isin(valid_drugs) & df['pid'].isin(valid_prots)
    clean_df = df[mask].reset_index(drop=True)
    return clean_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root', type=str, default='/root/lanyun-fs')
    parser.add_argument('--output_root', type=str, default='/root/lanyun-tmp/ugca-run')
    parser.add_argument('--mode', type=str, default='cold-drug')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()

    seed_everything(args.seed)
    exp_name = os.path.join(args.output_root, f"{args.dataset.upper()}_{args.mode}")
    os.makedirs(exp_name, exist_ok=True)

    csv_path = os.path.join(args.root, args.dataset, f"{args.dataset}.csv")
    print(f"Loading {csv_path}...")
    raw_df = pd.read_csv(csv_path).dropna(subset=['label']).reset_index(drop=True)
    df_with_ids = generate_ids(raw_df)
    clean_df = align_data_with_hdf5(df_with_ids, os.path.join(args.root, args.dataset, f"{args.dataset}_data.h5"))
    full_dataset = DTIDataset(clean_df, args.root, args.dataset, verbose=True)

    splits = get_kfold_indices(clean_df, mode=args.mode, seed=args.seed)
    results = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold_id = fold_i + 1
        fold_dir = os.path.join(exp_name, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        result_csv = os.path.join(fold_dir, 'result.csv')
        if os.path.exists(result_csv):
            print(f"\n=== Fold {fold_id} already finished. Skipping. ===")
            try:
                res = pd.read_csv(result_csv).iloc[0].to_dict()
                results.append(res)
            except: pass
            continue

        print(f"\n=== Fold {fold_id} ===")
        print(f"Dataset Sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        train_loader = get_dataloader(Subset(full_dataset, train_idx), args.batch_size, True, args.num_workers)
        val_loader = get_dataloader(Subset(full_dataset, val_idx), args.batch_size, False, args.num_workers)
        test_loader = get_dataloader(Subset(full_dataset, test_idx), args.batch_size, False, args.num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UGCADTI(dim=args.dim, dropout=args.dropout, num_heads=args.num_heads).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_auprc = 0.0
        patience_counter = 0
        start_epoch = 0

        last_ckpt_path = os.path.join(fold_dir, 'last.pt')
        if args.resume and os.path.exists(last_ckpt_path):
            print(f">>> Resuming from {last_ckpt_path} ...")
            ckpt = torch.load(last_ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_auprc = ckpt['best_auprc']
            patience_counter = ckpt.get('patience', 0)
            print(f">>> Resumed at Epoch {start_epoch}, Best AUPRC: {best_auprc:.4f}")

        log_file = os.path.join(fold_dir, 'log.csv')
        if not (args.resume and os.path.exists(log_file)):
            with open(log_file, 'w') as f:
                f.write("Epoch,TrainLoss,ValLoss,GateD,GateP,AUPRC,AUROC,F1,ACC,Time\n")

        prev_lr = optimizer.param_groups[0]['lr']

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_met = run_epoch(model, train_loader, criterion, optimizer, device, epoch, is_train=True)
            val_met = run_epoch(model, val_loader, criterion, optimizer, device, epoch, is_train=False)

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                print(f"LR updated: {prev_lr:.2e} -> {current_lr:.2e}")
                prev_lr = current_lr

            dur = time.time() - t0
            
            if val_met['AUPRC'] > best_auprc:
                best_auprc = val_met['AUPRC']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best.pt'))
            else:
                patience_counter += 1
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auprc': best_auprc,
                'patience': patience_counter
            }, last_ckpt_path)

            print(f"Ep {epoch+1:03d} | Loss: {train_met['Loss']:.4f} | AUPRC: {val_met['AUPRC']:.4f} | "
                  f"Gate(D/P): {val_met['Gate_D']:.3f}/{val_met['Gate_P']:.3f} | Pat: {patience_counter}")

            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{train_met['Loss']:.5f},{val_met['Loss']:.5f},"
                        f"{val_met['Gate_D']:.5f},{val_met['Gate_P']:.5f},"
                        f"{val_met['AUPRC']:.5f},{val_met['AUROC']:.5f},{val_met['F1']:.5f},{val_met['ACC']:.5f},{dur:.1f}\n")

            if patience_counter >= args.patience:
                print("Early Stopping."); break

        if os.path.exists(os.path.join(fold_dir, 'best.pt')):
            model.load_state_dict(torch.load(os.path.join(fold_dir, 'best.pt')))
            test_met = run_epoch(model, test_loader, criterion, optimizer, device, epoch, is_train=False)
            pd.DataFrame([test_met]).to_csv(result_csv, index=False)
            results.append(test_met)
            print(f"Fold {fold_id} Result: AUPRC={test_met['AUPRC']:.4f}")

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(exp_name, 'final_results.csv'), index=False)
        mean_res = df_res.mean(numeric_only=True)
        std_res = df_res.std(numeric_only=True)
        
        print("\n" + "="*40)
        print(f"FINAL AGGREGATED RESULTS ({len(results)} Folds)")
        print("="*40)
        for col in mean_res.index:
            if col not in ['Loss', 'Gate_D', 'Gate_P']:
                print(f"{col}: {mean_res[col]:.4f} ± {std_res[col]:.4f}")
        print("="*40)

        with open(os.path.join(exp_name, 'final_results.csv'), 'a') as f:
            f.write("\n\nMetric,Mean,Std\n")
            for col in mean_res.index:
                f.write(f"{col},{mean_res[col]:.6f},{std_res[col]:.6f}\n")

if __name__ == '__main__':
    main()