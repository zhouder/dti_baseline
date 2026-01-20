import argparse
import os
import torch
import torch.nn as nn
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
    matthews_corrcoef,
    confusion_matrix
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
    
    # 防止单一类别报错
    if len(np.unique(y_true)) < 2: 
        return {
            'AUPRC': 0.0, 'AUROC': 0.0, 'F1': 0.0, 
            'ACC': 0.0, 'SEN': 0.0, 'MCC': 0.0
        }
    
    # 基础指标
    auprc = average_precision_score(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    # 敏感度 (Sensitivity) / 召回率 (Recall)
    # SEN = TP / (TP + FN)
    sen = recall_score(y_true, y_pred)
    
    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'AUPRC': auprc,
        'AUROC': auroc,
        'F1': f1,
        'ACC': acc,
        'SEN': sen,
        'MCC': mcc
    }

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train: model.train()
    else: model.eval()
    total_loss = 0
    y_true, y_prob = [], []
    
    pbar = tqdm(loader, leave=False, desc="Train" if is_train else "Val", ncols=80)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train: optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(batch)
            labels = batch['label'].unsqueeze(1)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    metrics = compute_metrics(np.array(y_true), np.array(y_prob))
    metrics['Loss'] = total_loss / len(loader)
    return metrics

def align_data_with_hdf5(df, h5_path):
    """
    Filter DataFrame to match valid keys in HDF5.
    This replaces RDKit cleaning by relying on the pre-processed HDF5 file
    which should ideally contain only valid molecules.
    """
    if not os.path.exists(h5_path):
        print(f"Warning: HDF5 not found at {h5_path}. Skipping alignment.")
        return df
        
    print("Aligning data with HDF5 keys...")
    with h5py.File(h5_path, 'r') as f:
        # Use keys from HDF5 as the source of truth
        valid_drugs = set(f['drugs'].keys())
        valid_prots = set(f['proteins'].keys())
    
    # Filter df
    mask = df['did'].isin(valid_drugs) & df['pid'].isin(valid_prots)
    clean_df = df[mask].reset_index(drop=True)
    
    removed = len(df) - len(clean_df)
    if removed > 0:
        print(f"Filtered {removed} rows (missing in HDF5/Invalid SMILES).")
        
    return clean_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root', type=str, default='/root/lanyun-fs/')
    parser.add_argument('--output_root', type=str, default='/root/lanyun-tmp/ugca-runs')
    parser.add_argument('--mode', type=str, default='cold-drug')
    
    # Hyperparams matching Baseline
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    
    # Model Params
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--factor', type=float, default=0.8)
    parser.add_argument('--scheduler_patience', type=int, default=15)

    args = parser.parse_args()

    seed_everything(args.seed)
    exp_name = os.path.join(args.output_root, f"{args.dataset.upper()}_{args.mode}")
    os.makedirs(exp_name, exist_ok=True)
    
    # 1. Load Data
    csv_path = os.path.join(args.root, args.dataset, f"{args.dataset}.csv")
    print(f"Loading {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    raw_df = raw_df.dropna(subset=['label']).reset_index(drop=True)
    
    # 2. Generate IDs FIRST (Fixes KeyError)
    df_with_ids = generate_ids(raw_df)
    
    # 3. Clean based on HDF5 (Fixes Alignment without RDKit)
    h5_path = os.path.join(args.root, args.dataset, f"{args.dataset}_data.h5")
    clean_df = align_data_with_hdf5(df_with_ids, h5_path)
    
    print("-" * 40)
    print(f"Original Rows: {len(raw_df)}")
    print(f"Final Cleaned Rows: {len(clean_df)}")
    print("-" * 40)

    # 4. Init Dataset with CLEANED DF
    full_dataset = DTIDataset(clean_df, args.root, args.dataset, verbose=True)

    # 5. Split
    splits = get_kfold_indices(clean_df, mode=args.mode, seed=args.seed)
    results = []
    
    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold_id = fold_i + 1
        fold_dir = os.path.join(exp_name, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Check if done
        result_csv = os.path.join(fold_dir, 'result.csv')
        if os.path.exists(result_csv):
            print(f"\n=== Fold {fold_id} already finished. Skipping. ===")
            try: results.append(pd.read_csv(result_csv).iloc[0].to_dict())
            except: pass
            continue

        print(f"\n=== Fold {fold_id} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} ===")
        
        train_loader = get_dataloader(Subset(full_dataset, train_idx), args.batch_size, True, args.num_workers)
        val_loader = get_dataloader(Subset(full_dataset, val_idx), args.batch_size, False, args.num_workers)
        test_loader = get_dataloader(Subset(full_dataset, test_idx), args.batch_size, False, args.num_workers)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UGCADTI(dim=args.dim, dropout=args.dropout, num_heads=args.num_heads).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.factor, patience=args.scheduler_patience, verbose=True
        )
        
        best_auprc = 0.0
        patience_counter = 0
        start_epoch = 0
        
        # Resume
        last_ckpt = os.path.join(fold_dir, 'last.pt')
        if args.resume and os.path.exists(last_ckpt):
            print(">>> Resuming from last.pt...")
            ckpt = torch.load(last_ckpt)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_auprc = ckpt['best_auprc']
            patience_counter = ckpt.get('patience', 0)

        log_file = os.path.join(fold_dir, 'log.csv')
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f: 
                # [Change] Added ACC, SEN, MCC to header
                f.write("Epoch,TrainLoss,ValLoss,AUPRC,AUROC,F1,ACC,SEN,MCC,Time\n")
            
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_met = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_met = run_epoch(model, val_loader, criterion, optimizer, device, False)
            
            scheduler.step(val_met['AUPRC'])
            dur = time.time() - t0
            
            improved = val_met['AUPRC'] > best_auprc
            if improved:
                best_auprc = val_met['AUPRC']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best.pt'))
            else:
                patience_counter += 1
            
            status = "*" if improved else ""
            # [Change] Added ACC, SEN, MCC to print
            print(f"Ep {epoch+1:03d} | AUPRC: {val_met['AUPRC']:.4f} | AUROC: {val_met['AUROC']:.4f} | "
                  f"F1: {val_met['F1']:.4f} | ACC: {val_met['ACC']:.4f} | SEN: {val_met['SEN']:.4f} | "
                  f"MCC: {val_met['MCC']:.4f} | Time: {dur:.1f}s | Pat: {patience_counter} {status}")
            
            with open(log_file, 'a') as f:
                # [Change] Added ACC, SEN, MCC to log file
                f.write(f"{epoch+1},{train_met['Loss']:.5f},{val_met['Loss']:.5f},"
                        f"{val_met['AUPRC']:.5f},{val_met['AUROC']:.5f},{val_met['F1']:.5f},"
                        f"{val_met['ACC']:.5f},{val_met['SEN']:.5f},{val_met['MCC']:.5f},{dur:.1f}\n")
            
            torch.save({
                'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 'best_auprc': best_auprc, 'patience': patience_counter
            }, last_ckpt)
            
            if patience_counter >= args.patience: 
                print("Early Stopping."); break
        
        if os.path.exists(os.path.join(fold_dir, 'best.pt')):
            model.load_state_dict(torch.load(os.path.join(fold_dir, 'best.pt')))
            test_met = run_epoch(model, test_loader, criterion, optimizer, device, False)
            pd.DataFrame([test_met]).to_csv(result_csv, index=False)
            results.append(test_met)
            # [Change] Added ACC, SEN, MCC to final fold print
            print(f"Fold {fold_id} Test Results: AUPRC={test_met['AUPRC']:.4f}, AUROC={test_met['AUROC']:.4f}, "
                  f"F1={test_met['F1']:.4f}, ACC={test_met['ACC']:.4f}, SEN={test_met['SEN']:.4f}, MCC={test_met['MCC']:.4f}")

    if results:
        pd.DataFrame(results).describe().loc[['mean', 'std']].to_csv(os.path.join(exp_name, 'summary.csv'))

if __name__ == '__main__':
    main()