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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

from model import FractureMILModel
from loss import fracture_mil_loss
from dataset import PatientSplitter, SpineSliceDataset, get_train_transforms, get_val_transforms


def train_epoch(model, dataloader, optimizer, device, cfg, scaler=None):
    """
    1エポックの学習

    Args:
        model: FractureMILModel
        dataloader: 訓練DataLoader
        optimizer: オプティマイザ
        device: デバイス（cuda/cpu）
        cfg: Hydra設定
        scaler: AMP用GradScaler（use_amp=Trueの場合）

    Returns:
        avg_loss: 平均Loss
        avg_components: 各Loss項の平均値（dict）
    """
    model.train()
    total_loss = 0
    loss_components = {'L_bg': 0, 'L_max': 0, 'L_mean': 0}

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)  # (B, 3, 128, 128)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()

        # Forward + Loss計算
        if cfg.train.use_amp:
            with torch.cuda.amp.autocast():
                patch_probs, bone_mask = model(images)

            # Loss計算はfloat32で実行（binary_cross_entropy対応）
            with torch.cuda.amp.autocast(enabled=False):
                loss, components = fracture_mil_loss(
                    patch_probs.float(), bone_mask, labels.float(),
                    w_bg=cfg.loss.w_bg,
                    w_max=cfg.loss.w_max,
                    w_mean=cfg.loss.w_mean
                )

            # Backward（AMP）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Backward（通常）
            patch_probs, bone_mask = model(images)
            loss, components = fracture_mil_loss(
                patch_probs, bone_mask, labels,
                w_bg=cfg.loss.w_bg,
                w_max=cfg.loss.w_max,
                w_mean=cfg.loss.w_mean
            )
            loss.backward()
            optimizer.step()

        # 統計記録
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v

    # 平均化
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}

    return avg_loss, avg_components


def validate_epoch(model, dataloader, device, cfg):
    """
    1エポックの検証

    Args:
        model: FractureMILModel
        dataloader: 検証DataLoader
        device: デバイス
        cfg: Hydra設定（Loss重み取得用）

    Returns:
        metrics: 評価指標（dict）
        val_loss: 平均検証Loss
        val_loss_components: 各Loss項の平均値（dict）
    """
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0
    loss_components = {'L_bg': 0, 'L_max': 0, 'L_mean': 0}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            patch_probs, bone_mask = model(images)

            # Loss計算（検証時もLossを記録）
            loss, components = fracture_mil_loss(
                patch_probs, bone_mask, labels,
                w_bg=cfg.loss.w_bg,
                w_max=cfg.loss.w_max,
                w_mean=cfg.loss.w_mean
            )
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

            # スライス確率（骨パッチの最大値）
            slice_probs = []
            for b in range(images.shape[0]):
                bone_idx = bone_mask[b]
                if bone_idx.sum() > 0:
                    slice_probs.append(patch_probs[b, bone_idx].max().item())
                else:
                    slice_probs.append(0.0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(slice_probs)

    # 評価指標計算
    y_true = np.array(all_labels)
    y_pred = np.array(all_probs)

    metrics = {}
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred)

        # F1最適化（閾値探索）
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            y_pred_bin = (y_pred >= th).astype(int)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th

        metrics['f1_score'] = best_f1
        metrics['best_threshold'] = best_th

        # Confusion Matrix（最良閾値で）
        y_pred_bin = (y_pred >= best_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Recall と Precision を追加
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Sensitivity
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    except Exception as e:
        print(f"⚠ Metric calculation error: {e}")
        metrics = {k: 0.0 for k in ['roc_auc', 'pr_auc', 'f1_score', 'sensitivity', 'specificity', 'recall', 'precision']}
        metrics['best_threshold'] = 0.5

    # Loss平均化
    val_loss = total_loss / len(dataloader)
    val_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}

    return metrics, val_loss, val_loss_components


@hydra.main(version_base=None, config_path="config", config_name="mil_train")
def main(cfg: DictConfig):
    """
    メイン学習関数

    Hydra + W&B統合、患者レベル分割、Early Stopping対応。
    """
    print("=" * 80)
    print("prototype3: MILベース頸椎骨折検出システム")
    print("=" * 80)
    print(f"Fold: {cfg.experiment.fold}")
    print(f"GPU: {cfg.experiment.gpu}")
    print(f"Batch Size: {cfg.train.batch_size}")
    print(f"Loss重み: w_bg={cfg.loss.w_bg}, w_max={cfg.loss.w_max}, w_mean={cfg.loss.w_mean}")
    print("=" * 80)

    # --- シード固定（再現性確保） ---
    torch.manual_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.data.random_seed)

    # --- デバイス設定 ---
    device = torch.device(f"cuda:{cfg.experiment.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- W&B初期化 ---
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.experiment.name}_fold{cfg.experiment.fold}",
            config=dict(cfg),
            tags=cfg.wandb.tags
        )

    # --- データセット準備 ---
    annotation_csv_path = to_absolute_path(cfg.data.annotation_csv)
    data_dir_abs = to_absolute_path(cfg.data.data_dir)

    # 患者レベル分割
    splitter = PatientSplitter(
        annotation_csv_path,
        cfg.data.n_folds,
        cfg.data.test_size,
        cfg.data.random_seed
    )
    splits = splitter.split_patients()

    fold_idx = cfg.experiment.fold
    train_patients = splits['folds'][fold_idx]['train']
    val_patients = splits['folds'][fold_idx]['val']

    print(f"訓練患者数: {len(train_patients)}")
    print(f"検証患者数: {len(val_patients)}")
    print(f"テスト患者数: {len(splits['test'])}\n")

    # DataFrameフィルタリング
    df = pd.read_csv(annotation_csv_path)
    train_df = df[df['sample_id'].isin(train_patients)]
    val_df = df[df['sample_id'].isin(val_patients)]

    print(f"訓練スライス数: {len(train_df)}")
    print(f"検証スライス数: {len(val_df)}")
    print(f"訓練骨折率: {train_df['label'].mean():.3f}")
    print(f"検証骨折率: {val_df['label'].mean():.3f}\n")

    # Dataset作成
    train_dataset = SpineSliceDataset(
        train_df, data_dir_abs,
        transform=get_train_transforms(cfg)
    )
    val_dataset = SpineSliceDataset(
        val_df, data_dir_abs,
        transform=get_val_transforms(cfg)
    )

    # DataLoader作成
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

    # --- モデル作成 ---
    model = FractureMILModel(
        mask_threshold=cfg.model.mask_threshold,
        pretrained=cfg.model.pretrained
    ).to(device)
    print(f"モデル: FractureMILModel (mask_threshold={cfg.model.mask_threshold})")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- オプティマイザ ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay
    )

    # --- AMP用Scaler ---
    scaler = torch.cuda.amp.GradScaler() if cfg.train.use_amp else None

    # --- 学習ループ ---
    best_pr_auc = 0.0
    patience_counter = 0

    for epoch in range(cfg.train.epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{cfg.train.epochs}")
        print(f"{'=' * 80}")

        # 訓練
        train_loss, train_comps = train_epoch(
            model, train_loader, optimizer, device, cfg, scaler
        )
        print(f"[Train] Loss: {train_loss:.4f} | " +
              f"L_bg: {train_comps['L_bg']:.4f}, L_max: {train_comps['L_max']:.4f}, L_mean: {train_comps['L_mean']:.4f}")

        # 検証
        val_metrics, val_loss, val_loss_comps = validate_epoch(model, val_loader, device, cfg)
        print(f"[Val] Loss: {val_loss:.4f} | " +
              f"L_bg: {val_loss_comps['L_bg']:.4f}, L_max: {val_loss_comps['L_max']:.4f}, L_mean: {val_loss_comps['L_mean']:.4f}")
        print(f"      PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | " +
              f"F1: {val_metrics['f1_score']:.4f} (th={val_metrics['best_threshold']:.2f})")
        print(f"      Recall: {val_metrics['recall']:.4f} | Precision: {val_metrics['precision']:.4f} | " +
              f"Sensitivity: {val_metrics['sensitivity']:.4f} | Specificity: {val_metrics['specificity']:.4f}")

        # W&Bロギング
        if cfg.wandb.enabled:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_L_bg': train_comps['L_bg'],
                'train_L_max': train_comps['L_max'],
                'train_L_mean': train_comps['L_mean'],
                'val_loss': val_loss,
                'val_L_bg': val_loss_comps['L_bg'],
                'val_L_max': val_loss_comps['L_max'],
                'val_L_mean': val_loss_comps['L_mean'],
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })

        # Early Stopping（PR-AUCで判定）
        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            patience_counter = 0
            print(f"✓ Best PR-AUC更新: {best_pr_auc:.4f}")

            # モデル保存
            if cfg.experiment.save_best_model:
                save_path = f"best_model_fold{fold_idx}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_pr_auc': best_pr_auc,
                    'config': dict(cfg)
                }, save_path)
                print(f"   モデル保存: {save_path}")
        else:
            patience_counter += 1
            print(f"   Patience: {patience_counter}/{cfg.train.early_stopping_patience}")

            if patience_counter >= cfg.train.early_stopping_patience:
                print("\n⚠ Early Stopping発動")
                break

    print(f"\n{'=' * 80}")
    print(f"学習完了！Best PR-AUC: {best_pr_auc:.4f}")
    print(f"{'=' * 80}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
