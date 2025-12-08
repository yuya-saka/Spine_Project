# 頸椎骨折検出 - ResNet18 + LSTM実装計画

## 概要
- **アーキテクチャ**: ResNet18 + Bidirectional LSTM
- **タスク**: axialスライスからの骨折二値分類
- **データ**: 41患者、18,368スライス (陽性18.1%)
- **評価**: PR-AUC (主指標)、F1最大化閾値探索

## ファイル構成 (prototypeディレクトリ)
```
prototype/
├── config/
│   └── train.yaml             # Hydra設定
├── model.py                   # ResNet18 + LSTM
├── dataset.py                 # PatientSplitter, SpineSequenceDataset
├── train.py                   # 学習ループ
├── context.md                 # 実装メモ・進捗管理
└── README.md                  # ドキュメント

pyproject.toml (ルート)        # 依存関係追加済み
```

## 実装完了 (2025-12-08)
✅ 全ファイル実装完了
✅ prototypeディレクトリに配置
✅ パス修正完了 (../spine_data)

## 実装ポイント
- GPU番号選択可能 (`experiment.gpu`)
- 患者レベル分割 (5-fold CV + Test 6患者)
- F1最大化閾値探索 (Precision-Recall曲線)
- wandb可視化
- Mixed Precision Training

## 実行コマンド
```bash
python train.py                        # 基本実行 (fold 0, GPU 0)
python train.py experiment.gpu=1       # GPU指定
python train.py experiment.fold=-1     # 全fold実行
```
