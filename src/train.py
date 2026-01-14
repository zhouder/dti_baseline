import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    accuracy_score, matthews_corrcoef, recall_score
)

from src.splits import generate_ids, get_kfold_indices
from src.datamodule import get_dataloader, DTIDataset
from src.model import UGCADTI

def compute_metrics(y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return {k: 0.0 for k in ['AUPRC', 'AUROC', 'F1', 'ACC', 'SEN', 'MCC']}
    try:
        return {
            'AUPRC': average_precision_score(y_true, y_probs),
            'AUROC': roc_auc_score(y_true, y_probs),
            'F1': f1_score(y_true, y_pred),
            'ACC': accuracy_score(y_true, y_pred),
            'SEN': recall_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
    except:
        return {k: 0.0 for k in ['AUPRC', 'AUROC', 'F1', 'ACC', 'SEN', 'MCC']}

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train: model.train()
    else: model.eval()
    
    total_loss = 0
    y_true, y_prob = [], []
    
    pbar = tqdm(loader, leave=False, desc="Train" if is_train else "Val  ", ncols=100)
    
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
        probs = torch.sigmoid(logits) # BCE Logits -> Prob
        
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probs.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    metrics = compute_metrics(np.array(y_true), np.array(y_prob))
    metrics['Loss'] = total_loss / len(loader)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--root', type=str, default='/root/lanyun-fs/')
    parser.add_argument('--output_root', type=str, default='/root/lanyun-tmp/ugca-runs')
    parser.add_argument('--mode', type=str, default='cold-drug')
    parser.add_argument('--molclr_dim', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-4) # 稍微调大一点初始LR配合调度器
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) # 大数据集建议 128
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    exp_name = os.path.join(args.output_root, f"{args.dataset.upper()}_{args.mode}")
    os.makedirs(exp_name, exist_ok=True)
    
    # Load Data
    csv_path = os.path.join(args.root, args.dataset, f"{args.dataset}.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = generate_ids(df)
    
    # Split
    splits = get_kfold_indices(df, mode=args.mode)
    results = []
    
    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold_id = fold_i + 1
        fold_dir = os.path.join(exp_name, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n{'='*10} Fold {fold_id} {'='*10}")
        
        # Loaders (HDF5 Auto-detection in Datamodule)
        train_loader = get_dataloader(df.iloc[train_idx], args.root, args.dataset, args.batch_size, True, args.num_workers)
        val_loader = get_dataloader(df.iloc[val_idx], args.root, args.dataset, args.batch_size, False, args.num_workers)
        test_loader = get_dataloader(df.iloc[test_idx], args.root, args.dataset, args.batch_size, False, args.num_workers)
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UGCADTI(molclr_dim=args.molclr_dim).to(device)
        
        # 优化策略: AdamW + CosineAnnealingWarmRestarts
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Resume logic omitted for brevity, compatible with previous structure
        start_epoch = 0
        best_auprc = 0.0
        patience = 0
        log_file = os.path.join(fold_dir, 'log.csv')
        
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("Epoch,TrainLoss,ValLoss,AUPRC,F1,Time\n")
        
        for epoch in range(start_epoch, args.epochs):
            t_start = time.time()
            
            train_met = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_met = run_epoch(model, val_loader, criterion, optimizer, device, False)
            
            # Step Scheduler
            scheduler.step()
            
            t_end = time.time()
            dur = t_end - t_start
            
            # Print
            print(f"Ep {epoch+1:03d} | Lr: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Loss: {train_met['Loss']:.4f}/{val_met['Loss']:.4f} | "
                  f"AUPRC: {val_met['AUPRC']:.4f} | F1: {val_met['F1']:.4f}", end="")
            
            if val_met['AUPRC'] > best_auprc:
                best_auprc = val_met['AUPRC']
                patience = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best.pt'))
                print(" *")
            else:
                patience += 1
                print("")
                
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{train_met['Loss']:.5f},{val_met['Loss']:.5f},{val_met['AUPRC']:.5f},{val_met['F1']:.5f},{dur:.1f}\n")

            if patience >= args.patience: break

        # Test
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best.pt')))
        test_met = run_epoch(model, test_loader, criterion, optimizer, device, False)
        pd.DataFrame([test_met]).to_csv(os.path.join(fold_dir, 'result.csv'), index=False)
        results.append(test_met)
        print(f"Fold {fold_id} Test AUPRC: {test_met['AUPRC']:.4f}")

    # Summary
    pd.DataFrame(results).describe().loc[['mean', 'std']].to_csv(os.path.join(exp_name, 'summary.csv'))

if __name__ == '__main__':
    main()