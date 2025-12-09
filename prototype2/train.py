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
from sklearn.metrics import roc_auc_score, average_precision_score

from model import MILResNet
from dataset import PatientSplitter, SpineSequenceDataset, get_train_transforms, get_val_transforms

def mil_loss(logits, bag_labels, k=3):
    """
    Top-k MIL Loss (修正版: BCEWithLogitsLossを使用)
    Sigmoidをかけずにlogitのままソートし、BCEWithLogitsLossで計算することで
    Mixed Precision (autocast) でのエラーを回避します。
    """
    # 1. logitのままソート (Sigmoidは単調増加関数なので、大小関係は変わらない)
    # (B, T) -> (B, T)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    
    # 2. Top-kを取得
    current_k = min(k, sorted_logits.shape[1])
    top_k_logits = sorted_logits[:, :current_k] # (B, k)
    
    # 3. ターゲットの拡張
    targets = bag_labels.view(-1, 1).expand_as(top_k_logits)
    
    # 4. BCEWithLogitsLoss
    # 内部でSigmoid + BCELossを計算するため数値的に安定しており、autocast対応
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(top_k_logits, targets)
    
    return loss

# --- 追加: Center Loss関数 ---
def center_loss(probs, bag_labels):
    """
    Center Loss (論文式(2)準拠)
    正常スキャン(bag_label=0)の予測スコアを、そのスキャン内の平均値(center)に近づける。
    これにより、正常データのばらつき(分散)を抑制し、過学習を防ぐ。
    
    Args:
        probs: (B, T) - Sigmoid後の確率値
        bag_labels: (B,) - 0 or 1
    """
    # 正常スキャン(label=0)のインデックスを取得
    normal_indices = (bag_labels == 0).nonzero(as_tuple=True)[0]
    
    # バッチ内に正常データがない場合はLoss 0
    if len(normal_indices) == 0:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)
    
    # 正常データの予測値: (N_normal, T)
    normal_probs = probs[normal_indices]
    
    # 各スキャンごとの中心(平均)を計算: (N_normal, 1)
    centers = normal_probs.mean(dim=1, keepdim=True)
    
    # 各要素とCenterの距離(L2ノルムの二乗)を計算
    # 論文: || s_ij - c_i ||
    loss = torch.mean((normal_probs - centers) ** 2)
    
    return loss


def train_epoch(model, dataloader, optimizer, device, cfg, use_amp=True):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None # Warningが出る場合は torch.amp.GradScaler('cuda', ...)

    # cfgからパラメータ取得
    k = cfg.mil.k
    lambda_dmil = cfg.mil.lambda_dmil
    lambda_c = cfg.mil.lambda_c

    for sequences, bag_labels in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        bag_labels = bag_labels.to(device)

        optimizer.zero_grad()

        # Forward & Loss計算
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(sequences) # (B, T)
                probs = torch.sigmoid(logits) # Center Loss用に確率化
                
                # 1. DMIL Loss (異常検知) - Logitsを使用
                l_dmil = mil_loss(logits, bag_labels, k=k)
                
                # 2. Center Loss (過学習抑制) - Probsを使用
                l_c = center_loss(probs, bag_labels)
                
                # 合計Loss
                loss = (lambda_dmil * l_dmil) + (lambda_c * l_c)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences)
            probs = torch.sigmoid(logits)
            
            l_dmil = mil_loss(logits, bag_labels, k=k)
            l_c = center_loss(probs, bag_labels)
            
            loss = (lambda_dmil * l_dmil) + (lambda_c * l_c)
            
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device, k):
    """
    検証: ここでは単純にBagレベル(ウィンドウレベル)での予測精度を見る
    """
    model.eval()
    total_loss = 0
    all_bag_labels = []
    all_bag_probs = []

    with torch.no_grad():
        for sequences, bag_labels in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            bag_labels = bag_labels.to(device)

            logits = model(sequences) # (B, T)
            loss = mil_loss(logits, bag_labels, k=k)
            total_loss += loss.item()
            
            # 推論: ウィンドウ(Bag)としての異常スコアは、
            # 最も異常度の高いスライスのスコア (Max Pooling) とする
            probs = torch.sigmoid(logits)
            bag_prob, _ = torch.max(probs, dim=1) # (B,)
            
            all_bag_probs.extend(bag_prob.cpu().numpy())
            all_bag_labels.extend(bag_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    
    # Metrics
    y_true = np.array(all_bag_labels)
    y_pred = np.array(all_bag_probs)
    
    try:
        auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
    except:
        auc = 0.5
        pr_auc = 0.0

    return avg_loss, auc, pr_auc

@hydra.main(version_base=None, config_path="config", config_name="mil_train")
def main(cfg: DictConfig):
    
    # 準備など (省略せず記述)
    torch.manual_seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)
    device = torch.device(f"cuda:{cfg.experiment.gpu}" if torch.cuda.is_available() else "cpu")
    
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.experiment.name}_fold{cfg.experiment.fold}",
            config=dict(cfg),
            tags=cfg.wandb.tags
        )

    # Dataset準備
    annotation_csv_path = to_absolute_path(cfg.data.annotation_csv)
    splitter = PatientSplitter(annotation_csv_path, cfg.data.n_folds, cfg.data.test_size, cfg.data.random_seed)
    splits = splitter.split_patients()
    
    fold_idx = cfg.experiment.fold
    train_patients = splits['folds'][fold_idx]['train']
    val_patients = splits['folds'][fold_idx]['val']
    
    df = pd.read_csv(annotation_csv_path)
    train_df = df[df['sample_id'].isin(train_patients)]
    val_df = df[df['sample_id'].isin(val_patients)]
    
    data_dir_abs = to_absolute_path(cfg.data.data_dir)
    
    train_dataset = SpineSequenceDataset(
        train_df, data_dir_abs, 
        transform=get_train_transforms(cfg), 
        sequence_length=cfg.data.sequence_length
    )
    val_dataset = SpineSequenceDataset(
        val_df, data_dir_abs, 
        transform=get_val_transforms(cfg), 
        sequence_length=cfg.data.sequence_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.experiment.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.experiment.num_workers, pin_memory=True)

    # Model
    model = MILResNet(encoder_name=cfg.model.encoder_name, dropout=cfg.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(cfg.train.epochs):
        print(f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg, use_amp=cfg.train.use_amp)
        val_loss, val_auc, val_pr_auc = validate_epoch(model, val_loader, device, k=cfg.mil.k)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if cfg.wandb.enabled:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_auc': val_auc, 'val_pr_auc': val_pr_auc})
            
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pth")
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()