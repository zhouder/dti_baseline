import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)
from tqdm import tqdm

# Import local modules
# Assuming models are in 'models' directory as per GraphDTA structure
from models.ginconv import GINConvNet
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from src.baseline_data import GraphDTADataset, get_kfold_indices


# ------------------------------------------------------------------------------
# 1. Metrics Calculation
# ------------------------------------------------------------------------------
def calculate_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_score)
    except:
        auprc = 0

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'AUROC': auroc, 'AUPRC': auprc, 'F1': f1, 'Accuracy': acc,
        'Sensitivity': rec, 'Specificity': spec, 'Precision': prec, 'MCC': mcc
    }


def find_best_threshold(y_true, y_score):
    best_f1 = 0
    best_thr = 0.5
    thresholds = np.linspace(0, 1, 101)
    for thr in thresholds:
        f1 = f1_score(y_true, (y_score >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# ------------------------------------------------------------------------------
# 2. Training Helper
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true_all, y_score_all = [], []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)  # GraphDTA output is [batch, 1]

        # Classification Loss
        loss = criterion(output.view(-1), data.y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        y_true_all.extend(data.y.cpu().numpy())
        y_score_all.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(y_true_all), np.array(y_score_all), threshold=0.5)
    metrics['loss'] = avg_loss
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true_all, y_score_all = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            y_true_all.extend(data.y.cpu().numpy())
            y_score_all.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)

    # Just return raw arrays for threshold search later, plus standard metrics
    y_true = np.array(y_true_all)
    y_score = np.array(y_score_all)

    metrics = calculate_metrics(y_true, y_score, threshold=0.5)
    metrics['loss'] = avg_loss
    return metrics, y_true, y_score


# ------------------------------------------------------------------------------
# 3. Main Loop
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (DAVIS/BindingDB/BioSNAP)')
    parser.add_argument('--data-root', type=str, required=True, help='Path containing <dataset>/all.csv')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='GIN', choices=['GIN', 'GAT', 'GCN', 'GAT_GCN'])
    parser.add_argument('--split-mode', type=str, default='cold-protein',
                        choices=['cold-protein', 'cold-drug', 'warm', 'cold-both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # 1. Setup Data
    csv_path = os.path.join(args.data_root, args.dataset, 'all.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path}")

    df = pd.read_csv(csv_path)

    # Cache processed data specifically for this baseline to avoid conflicts
    cache_path = os.path.join(args.data_root, 'baseline_cache', args.dataset)
    os.makedirs(cache_path, exist_ok=True)

    dataset = GraphDTADataset(root=cache_path, df=df)

    # 2. Split
    splits = get_kfold_indices(dataset, args.split_mode)

    # 3. Output directories
    run_dir = os.path.join(args.output_dir, f"{args.dataset}_{args.split_mode}_{args.model}")
    os.makedirs(run_dir, exist_ok=True)

    summary_metrics = []

    # 4. K-Fold Training
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n========== Fold {fold + 1} / 5 ==========")
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Inner split for Val (12.5% of train = 10% of total)
        # We need a validation set for early stopping
        # Simple random split on training indices to keep it simple but separate
        np.random.shuffle(train_idx)
        n_val = int(len(train_idx) * 0.125)
        val_idx = train_idx[:n_val]
        real_train_idx = train_idx[n_val:]

        train_loader = DataLoader(dataset[real_train_idx], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False)

        # Init Model
        if args.model == 'GIN':
            model = GINConvNet()  # Default GraphDTA GIN
        elif args.model == 'GAT':
            model = GATNet()
        elif args.model == 'GCN':
            model = GCNNet()
        elif args.model == 'GAT_GCN':
            model = GAT_GCN()

        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.BCEWithLogitsLoss()  # IMPORTANT: Classification Loss

        # Logging
        log_file = open(os.path.join(fold_dir, "log.csv"), "w")
        log_file.write("epoch,split,loss,AUROC,AUPRC,F1,Accuracy,Sensitivity,Specificity,Precision,MCC,time\n")

        best_val_auprc = -1
        patience_counter = 0
        best_model_path = os.path.join(fold_dir, "best.pt")

        # Epoch Loop
        for epoch in range(1, args.epochs + 1):
            t_start = time.time()

            # Train
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

            # Val
            val_metrics, val_y, val_score = validate(model, val_loader, criterion, args.device)

            t_end = time.time()
            duration = t_end - t_start

            # Log
            def log_line(split, m):
                return f"{epoch},{split},{m['loss']:.4f},{m['AUROC']:.4f},{m['AUPRC']:.4f},{m['F1']:.4f},{m['Accuracy']:.4f},{m['Sensitivity']:.4f},{m['Specificity']:.4f},{m['Precision']:.4f},{m['MCC']:.4f},{duration:.1f}"

            log_file.write(log_line('train', train_metrics) + "\n")
            log_file.write(log_line('val', val_metrics) + "\n")
            log_file.flush()

            print(f"Ep {epoch} | Val AUPRC: {val_metrics['AUPRC']:.4f} | Loss: {val_metrics['loss']:.4f}")

            # Early Stopping Check
            if val_metrics['AUPRC'] > best_val_auprc:
                best_val_auprc = val_metrics['AUPRC']
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print("  -> New Best Saved!")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        log_file.close()

        # Final Evaluation on TEST set using BEST Threshold found on VAL
        print("Evaluating on Test Set...")
        model.load_state_dict(torch.load(best_model_path))

        # 1. Get predictions on Val to find Threshold
        _, val_y, val_score = validate(model, val_loader, criterion, args.device)
        best_thr = find_best_threshold(val_y, val_score)
        print(f"Best Threshold found on Val: {best_thr:.4f}")

        # 2. Predict on Test
        _, test_y, test_score = validate(model, test_loader, criterion, args.device)
        final_metrics = calculate_metrics(test_y, test_score, threshold=best_thr)
        final_metrics['threshold'] = best_thr
        final_metrics['fold'] = fold + 1

        summary_metrics.append(final_metrics)

        # Save Result.csv for this fold
        res_df = pd.DataFrame([final_metrics])
        res_df.to_csv(os.path.join(fold_dir, "result.csv"), index=False)

    # 5. Summary
    summ_df = pd.DataFrame(summary_metrics)
    summ_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # Calculate Mean/Std
    mean_row = summ_df.mean(numeric_only=True)
    std_row = summ_df.std(numeric_only=True)

    print("\n========== 5-Fold Summary ==========")
    print(summ_df)
    print("\nAverage:")
    print(mean_row)


if __name__ == "__main__":
    main()