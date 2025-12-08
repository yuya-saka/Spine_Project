# Prototype: ResNet18 + LSTM骨折検出

## 概要
- **アーキテクチャ**: ResNet18 Encoder + Bidirectional LSTM
- **タスク**: axialスライスからの骨折二値分類 (スライスレベル)
- **データ**: 41患者、18,368スライス (陽性18.1%)
- **主指標**: PR-AUC (F1最大化閾値探索)

## ファイル構成
```
prototype/
├── config/
│   └── train.yaml          # Hydra設定
├── model.py                # ResNet18 + LSTM
├── dataset.py              # PatientSplitter, SpineSequenceDataset
└── train.py                # 学習ループ
```

## 実装内容

### モデル (model.py)
- **Encoder**: ResNet18 (ImageNet pretrained)
  - 各スライス(3, 128, 128)を独立に処理 → 512次元特徴量
- **LSTM**: Bidirectional LSTM (hidden=256, layers=2)
  - 64スライスのシーケンスを処理
- **分類ヘッド**: FC(512→128→1) + Dropout

### Dataset (dataset.py)
- **PatientSplitter**: 患者レベル分割
  - Test: 6患者 (holdout)
  - Train+Val: 35患者 → 5-fold CV
  - データリーケージ防止
- **SpineSequenceDataset**: 椎骨単位で64スライスを返す
  - 入力: (64, 3, 128, 128)
  - ラベル: (64,) スライスごとのラベル
- **データ拡張**: HorizontalFlip, Rotate, RandomBrightnessContrast

### 学習ループ (train.py)
- **評価指標**:
  - 主指標: PR-AUC
  - 副指標: ROC-AUC, Precision, Recall, F1, Sensitivity, Specificity
  - F1最大化閾値: Precision-Recall曲線で自動探索
- **最適化**: AdamW + CosineAnnealingLR
- **Early Stopping**: patience=10
- **Mixed Precision Training**: AMP対応
- **wandb**: 自動ログ記録

## 実行コマンド

### 基本実行
```bash
cd prototype
python train.py                        # fold 0, GPU 0
python train.py experiment.gpu=1       # GPU指定
python train.py experiment.fold=-1     # 全fold実行
```

### ハイパーパラメータ変更
```bash
python train.py train.batch_size=8 model.lstm_hidden=512
python train.py wandb.enabled=false    # wandb無効化
```

## ハイパーパラメータ (デフォルト)
- **Batch size**: 4椎骨 (256スライス)
- **Epochs**: 50
- **Learning rate**: 1e-4
- **LSTM hidden**: 256
- **LSTM layers**: 2
- **Dropout**: 0.3

## 重要な設計判断

### 患者レベル分割
- 同一患者の全データを同じfoldに配置
- テスト6患者は学習に一切使用しない
- データリーケージを完全に防止

### F1最大化閾値
- Precision-Recall曲線を計算
- F1スコアが最大となる閾値を自動探索
- その閾値で全評価指標を計算

### RSNA2022参考ポイント
- 各axialスライスを2D CNNでエンコード
- Bidirectional LSTMでスライス連続性を考慮
- 患者レベル分割必須

## 進捗メモ

### 2025-12-08: 初期実装完了
- [x] model.py実装 (ResNet18 + LSTM)
- [x] dataset.py実装 (PatientSplitter, SpineSequenceDataset)
- [x] train.py実装 (学習ループ)
- [x] config/train.yaml作成
- [ ] 動作確認 (1 fold実行)
- [ ] ベースライン結果取得

## 次のステップ
1. 1 foldで動作確認
2. PR-AUCベースライン取得
3. 5-fold CV実行
4. ハイパーパラメータ探索
