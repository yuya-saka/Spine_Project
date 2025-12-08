# Prototype: 頸椎骨折検出 (ResNet18 + LSTM)

## 概要
axialスライス画像からの骨折二値分類システムのプロトタイプ実装。

**アーキテクチャ**: ResNet18 Encoder + Bidirectional LSTM
**主指標**: PR-AUC (F1最大化閾値)

## クイックスタート

### 1. 環境セットアップ
```bash
# プロジェクトルートで依存関係をインストール
cd /mnt/nfs1/home/yamamoto-hiroto/research/cervical-spine
uv sync
```

### 2. 学習実行
```bash
cd prototype
python train.py                        # fold 0, GPU 0で実行
```

### 3. その他のオプション
```bash
# GPU指定
python train.py experiment.gpu=1

# 特定fold実行
python train.py experiment.fold=2

# 全fold実行
python train.py experiment.fold=-1

# ハイパーパラメータ変更
python train.py train.batch_size=8 model.lstm_hidden=512

# wandb無効化 (デバッグ用)
python train.py wandb.enabled=false
```

## ファイル構成
```
prototype/
├── config/
│   └── train.yaml          # Hydra設定ファイル
├── model.py                # CNN_LSTM_Fractureモデル定義
├── dataset.py              # Dataset, PatientSplitter定義
├── train.py                # 学習ループメインスクリプト
├── context.md              # 実装メモ・進捗管理
└── README.md               # このファイル
```

## モデル詳細

### アーキテクチャ
```
Input: (batch, 64, 3, 128, 128)  # 64スライスのシーケンス
  ↓
ResNet18 Encoder (pretrained)
  ↓ (batch*64, 512)
Reshape: (batch, 64, 512)
  ↓
Bidirectional LSTM (hidden=256, layers=2)
  ↓ (batch, 64, 512)
Classification Head (FC: 512→128→1)
  ↓
Output: (batch, 64)  # 各スライスのlogits
```

### データ処理
- **患者レベル分割**: Test 6患者 + 5-fold CV (35患者)
- **入力形式**: 椎骨単位で64スライス
  - Ch0: ボーン窓
  - Ch1: ソフト窓
  - Ch2: 椎骨マスク
- **データ拡張**: HorizontalFlip, Rotate(±10°), RandomBrightnessContrast

### 評価指標
- **主指標**: PR-AUC (不均衡データ対策)
- **F1最大化閾値**: Precision-Recall曲線で自動探索
- **副指標**: ROC-AUC, Precision, Recall, F1, Sensitivity, Specificity

## 設定ファイル (config/train.yaml)

### 主要パラメータ
```yaml
data:
  n_folds: 5
  test_size: 6
  random_seed: 42

model:
  encoder_name: resnet18
  lstm_hidden: 256
  lstm_layers: 2
  bidirectional: true
  dropout: 0.3

train:
  batch_size: 4
  epochs: 50
  learning_rate: 0.0001
  early_stopping_patience: 10
  use_amp: true

experiment:
  gpu: 0
  num_workers: 4
  fold: 0
```

## 出力

### チェックポイント
```
checkpoints/
└── fold{i}_best.pth        # 各foldのベストモデル
```

### wandbログ
- `train_loss`, `val_loss`
- `val_pr_auc`, `val_roc_auc`
- `val_best_threshold`
- `val_precision`, `val_recall`, `val_f1`
- `val_sensitivity`, `val_specificity`

## トラブルシューティング

| 問題 | 対策 |
|------|------|
| CUDA Out of Memory | `train.batch_size=2` |
| 学習が進まない | `train.learning_rate=0.00001` |
| wandbログインエラー | `wandb login` |

## 実装メモ
詳細は [context.md](context.md) を参照。
