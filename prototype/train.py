"""
学習ループ: train_epoch, validate_epoch, compute_metrics, main
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import wandb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

from model import CNN_LSTM_Fracture
from dataset import PatientSplitter, SpineSequenceDataset, get_train_transforms, get_val_transforms


def compute_metrics(y_true, y_pred_proba):
    """
    全評価指標を計算 (F1最大化閾値を探索)

    Args:
        y_true: (N,) ground truth labels
        y_pred_proba: (N,) predicted probabilities

    Returns:
        dict: 評価指標 (best_threshold含む)
    """
    # Precision-Recall曲線でF1を最大化する閾値を探索
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    # Best閾値で予測
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    metrics = {
        'pr_auc': average_precision_score(y_true, y_pred_proba),  # 主指標
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'best_threshold': best_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # Sensitivity & Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=True):
    """1エポックのトレーニング"""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(sequences)
                loss = criterion(logits, labels).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences)
            loss = criterion(logits, labels).mean()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validation/Test評価"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            labels = labels.to(device).float()

            logits = model(sequences)
            loss = criterion(logits, labels).mean()
            total_loss += loss.item()

            # スライスレベルで予測確率を収集
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels_flat)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))

    return avg_loss, metrics


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """メイン実行関数"""

    # シード固定
    torch.manual_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)

    # デバイス設定 (GPU番号指定)
    device = torch.device(f"cuda:{cfg.experiment.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # wandb初期化
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.experiment.name}_fold{cfg.experiment.fold}",
            config=dict(cfg),
            tags=cfg.wandb.tags
        )

    # 患者分割
    annotation_csv_path = to_absolute_path(cfg.data.annotation_csv)
    splitter = PatientSplitter(
        annotation_csv_path,
        cfg.data.n_folds,
        cfg.data.test_size,
        cfg.data.random_seed
    )
    splits = splitter.split_patients()

    # 実行するfoldを選択
    fold = cfg.experiment.fold
    if fold == -1:
        folds_to_run = range(cfg.data.n_folds)
    else:
        folds_to_run = [fold]

    for fold_idx in folds_to_run:
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}")
        print(f"{'='*60}")

        # Dataset作成
        train_patients = splits['folds'][fold_idx]['train']
        val_patients = splits['folds'][fold_idx]['val']

        # アノテーションCSVから該当患者のデータを抽出
        df = pd.read_csv(annotation_csv_path)
        train_df = df[df['sample_id'].isin(train_patients)]
        val_df = df[df['sample_id'].isin(val_patients)]

        print(f"Train: {len(train_patients)} patients, {len(train_df)} slices")
        print(f"Val: {len(val_patients)} patients, {len(val_df)} slices")

        # Datasetとローダー
        data_dir_abs = to_absolute_path(cfg.data.data_dir)
        train_transform = get_train_transforms(cfg) if cfg.augmentation.enabled else None
        train_dataset = SpineSequenceDataset(
            train_df,
            data_dir=data_dir_abs,
            transform=train_transform,
            sequence_length=cfg.data.sequence_length
        )
        
        val_transform = get_val_transforms(cfg) if cfg.augmentation.enabled else None
        
        val_dataset = SpineSequenceDataset(
            val_df,
            data_dir=data_dir_abs,
            transform=val_transform,
            sequence_length=cfg.data.sequence_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.experiment.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.experiment.num_workers,
            pin_memory=True
        )

        # モデル初期化
        model = CNN_LSTM_Fracture(
            encoder_name=cfg.model.encoder_name,
            lstm_hidden=cfg.model.lstm_hidden,
            lstm_layers=cfg.model.lstm_layers,
            bidirectional=cfg.model.bidirectional,
            dropout=cfg.model.dropout
        ).to(device)

        # 損失関数・Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs
        )

        # トレーニングループ
        best_pr_auc = 0
        patience_counter = 0

        for epoch in range(cfg.train.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.train.epochs}")

            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                device, cfg.train.use_amp
            )

            # Validation
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device
            )

            scheduler.step()

            # ログ出力
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"PR-AUC: {val_metrics['pr_auc']:.4f}, "
                  f"ROC-AUC: {val_metrics['roc_auc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"Best Threshold: {val_metrics['best_threshold']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            print(f"Sensitivity: {val_metrics['sensitivity']:.4f}, "
                  f"Specificity: {val_metrics['specificity']:.4f}")

            # wandbログ
            if cfg.wandb.enabled:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })

            # Best model保存
            if val_metrics['pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['pr_auc']
                patience_counter = 0

                checkpoint_path = f"checkpoints/fold{fold_idx}_best.pth"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_pr_auc': best_pr_auc,
                    'val_metrics': val_metrics,
                    'config': dict(cfg)
                }, checkpoint_path)
                print(f"✓ Best model saved (PR-AUC: {best_pr_auc:.4f})")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= cfg.train.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nFold {fold_idx} completed. Best PR-AUC: {best_pr_auc:.4f}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
