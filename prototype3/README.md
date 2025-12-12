# prototype3: MILベース頸椎骨折検出システム

## 📋 概要

パッチレベルMIL（Multiple Instance Learning）による頸椎骨折検出システムです。
1枚のスライス画像を16個のパッチ（4×4グリッド）に分割し、各パッチの骨折確率を予測。
**ヒートマップ生成**により骨折部位の局在化を実現します。

### 主要な特徴

- **Bag定義**: 1枚のスライス画像
- **Instance定義**: 特徴マップ上の4×4グリッド（16パッチ）
- **出力**: パッチレベル確率 + ヒートマップ（128×128）
- **Loss**: Fracture MIL Loss（背景抑制 + 最大値 + 平均値の3項構成）
- **マスク活用**: 椎体マスクで骨/背景を判定し、背景での誤検知を削減

### prototype2との違い

| 項目 | prototype2 | prototype3 |
|------|-----------|-----------|
| Bag定義 | 16枚のスライスシーケンス | **1枚のスライス画像** |
| Instance定義 | 各スライス | **4×4グリッド（16パッチ）** |
| 出力 | Bag-level確率のみ | **Patch-level確率 + ヒートマップ** |
| Loss | Top-k MIL + Center Loss | **Fracture MIL Loss（3項）** |
| 局在化 | なし | **ヒートマップで可視化** |

---

## 📁 ファイル構成

```
prototype3/
├── model.py              # モデル定義（BackboneFeatureExtractor, MILHead, FractureMILModel）
├── loss.py               # カスタムLoss（fracture_mil_loss）
├── dataset.py            # データセット（PatientSplitter, SpineSliceDataset）
├── train.py              # 学習スクリプト（Hydra + W&B統合）
├── evaluate.py           # 評価・ヒートマップ可視化スクリプト
├── config/
│   └── mil_train.yaml    # 設定ファイル（ハイパーパラメータ）
├── docs.md               # 要件定義書（詳細仕様）
└── README.md             # このファイル
```

---

## 🚀 使用方法

### 1. 環境準備

```bash
# 必要なパッケージ（prototype2と同じ環境を想定）
pip install torch torchvision numpy pandas scikit-learn albumentations hydra-core wandb matplotlib
```

### 2. 学習実行

#### 単一Fold学習

```bash
cd prototype3
python train.py experiment.fold=0
```

#### 5-fold Cross Validation

```bash
# Fold 0-4を順次実行
for fold in {0..4}; do
    python train.py experiment.fold=$fold
done
```

#### 設定のカスタマイズ

```bash
# Loss重みの調整
python train.py loss.w_bg=2.0 loss.w_mean=0.3

# バッチサイズ変更
python train.py train.batch_size=128

# GPU指定
python train.py experiment.gpu=0

# W&B無効化(デバッグ実行)
uv run python train.py wandb.enabled=false train.epochs=2 
```

### 3. ヒートマップ生成

学習完了後、学習済みモデルでヒートマップを生成:

```bash
python evaluate.py \
    --model_path best_model_fold0.pth \
    --data_dir ../spine_data \
    --annotation_csv ../spine_data/slice_annotations.csv \
    --save_dir ./heatmaps \
    --num_samples 20
```

#### 出力

- `heatmaps/heatmap_*.png`: ランダムサンプルのヒートマップ
- `heatmaps/normal_*.png`: 正常サンプルのヒートマップ
- `heatmaps/fracture_*.png`: 骨折サンプルのヒートマップ

---

## ⚙️ 設定ファイル（`config/mil_train.yaml`）

### 主要パラメータ

#### モデル設定
- `model.mask_threshold`: 骨パッチ判定閾値（デフォルト: 0.1）
  - パッチの10%以上が椎体なら「骨」と判定
  - 値を上げる → 骨領域を厳密化（背景混入を削減）

#### Loss設定
- `loss.w_bg`: 背景抑制Lossの重み（デフォルト: 1.5）
  - 背景での誤検知を強くペナルティ
  - 値を上げる → False Positive削減（ただし過度に上げるとRecall低下）
- `loss.w_max`: 最大値Lossの重み（デフォルト: 1.0）
  - MIL主タスク、基準値
- `loss.w_mean`: 平均値Lossの重み（デフォルト: 0.5）
  - 学習安定化、弱めに設定
  - 値を上げる → 安定化だが局在化精度が低下する可能性

#### 学習設定
- `train.batch_size`: 64（prototype2の2倍）
  - 単一スライスなのでメモリ効率良好
- `train.learning_rate`: 0.0001
- `train.early_stopping_patience`: 15エポック

---

## 📊 評価指標

### 定量評価
- **PR-AUC** (最重要): クラス不均衡に強い、医療AIで推奨
- **ROC-AUC**: 一般的な性能指標
- **F1-Score**: 実用的な閾値決定
- **Sensitivity / Specificity**: 医療AI特有の要求

### 定性評価
- **ヒートマップ可視化**:
  - True Positive: 骨折線・変形部位を指しているか
  - False Positive: 誤検知の原因分析（アーチファクト、椎間板等）
  - 背景への漏れが少ないか

---

## 🎯 目標性能

### Validation（5-fold CV平均）
- PR-AUC ≥ 0.75（目標: 0.80）
- ROC-AUC ≥ 0.85
- F1-Score ≥ 0.70
- Sensitivity ≥ 0.80（見逃し削減）

### Test Set（最終評価）
- PR-AUC ≥ 0.72（汎化性能確認）

---

## 🔧 トラブルシューティング

### 問題1: Loss が発散する

**原因**: Loss重みのバランス不良、学習率が高すぎる

**対策**:
```bash
# Loss重みを下げる
python train.py loss.w_bg=1.0 loss.w_mean=0.3

# 学習率を下げる
python train.py train.learning_rate=0.00005
```

### 問題2: False Positive が多い

**原因**: 背景抑制が不足、正常サンプルの学習不足

**対策**:
```bash
# 背景抑制Lossを強化
python train.py loss.w_bg=2.0

# マスク閾値を上げて骨領域を厳密化
python train.py model.mask_threshold=0.15
```

### 問題3: ヒートマップの局在化精度が低い

**原因**: L_meanの重みが強すぎて骨折パッチが分散

**対策**:
```bash
# L_meanを弱める
python train.py loss.w_mean=0.3

# マスク閾値を調整
python train.py model.mask_threshold=0.05
```

### 問題4: GPU メモリ不足

**対策**:
```bash
# バッチサイズを減らす
python train.py train.batch_size=32

# AMPを有効化（通常デフォルトで有効）
python train.py train.use_amp=true
```

---

## 📖 技術的詳細

### MILアーキテクチャ

#### BackboneFeatureExtractor
- ResNet18（ImageNet事前学習）
- avgpool・fcを削除 → 特徴マップ(B, 512, 4, 4)を出力

#### MILHead
1. **パッチ化**: (B, 512, 4, 4) → (B, 16, 512)
2. **Instance分類**: Linear(512→1) + Sigmoid → 各パッチの骨折確率
3. **マスク処理**: Ch2を4×4にダウンサンプリング → 閾値処理で骨/背景判定

#### 推論時
- **スライス確率**: 骨パッチの最大値（Max Pooling）
- **ヒートマップ**: 4×4を128×128にBilinear Upsampling

### Fracture MIL Loss（3項構成）

#### 項1: 背景抑制 (L_bg)
```
L_bg = BCE(背景パッチの予測, 0)
```
- 役割: 骨のない場所での誤検知（False Positive）を削減
- 重み: w_bg = 1.5（強め）

#### 項2: 骨領域・最大値 (L_max)
```
L_max = BCE(骨パッチの最大値, スライスラベル)
```
- 役割: 「どこか一箇所でも骨折があればスライス全体が陽性」というMILの核心
- 重み: w_max = 1.0（基準）

#### 項3: 骨領域・平均値 (L_mean)
```
L_mean = BCE(骨パッチの平均値, スライスラベル)
```
- 役割: 学習の安定化（正常→全パッチ低確率、骨折→複数パッチ高確率）
- 重み: w_mean = 0.5（弱め、局在性維持のため）

---

## 🔬 実験ノート

### ハイパーパラメータ探索

#### 実験1: Loss重みの影響
```bash
# w_bg = 1.0, 1.5, 2.0
# w_mean = 0.3, 0.5, 0.7
```

#### 実験2: マスク閾値の影響
```bash
# mask_threshold = 0.05, 0.1, 0.15
```

#### 実験3: データ拡張の強度
```bash
# rotate_p = 0.3, 0.5, 0.7
# brightness_contrast_p = 0.2, 0.3, 0.5
```

### 結果記録

| 実験 | PR-AUC | ROC-AUC | F1 | 備考 |
|------|--------|---------|-----|------|
| baseline | - | - | - | デフォルト設定 |
| exp001 | - | - | - | w_bg=2.0 |
| exp002 | - | - | - | mask_threshold=0.15 |

---

## 📚 参考文献

- **MIL**: [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712)
- **医療画像AI**: Weakly Supervised Localization in Medical Imaging

---

## ✅ チェックリスト

### 実装完了
- [x] model.py（BackboneFeatureExtractor, MILHead, FractureMILModel）
- [x] loss.py（fracture_mil_loss）
- [x] dataset.py（PatientSplitter, SpineSliceDataset）
- [x] config/mil_train.yaml
- [x] train.py（学習パイプライン）
- [x] evaluate.py（ヒートマップ生成）

### 検証項目
- [ ] 小規模テスト実行（5エポック、fold=0）
- [ ] Loss が正常に減少することを確認
- [ ] ヒートマップが生成されることを確認
- [ ] 5-fold Cross Validation実行
- [ ] 目標PR-AUC（≥0.75）達成確認

### データ整合性
- [ ] PatientSplitterの動作確認（患者ID重複なし）
- [ ] スライスラベルの正確性確認
- [ ] マスク品質の目視確認（数サンプル）

---

## 📞 サポート

問題が発生した場合:
1. [docs.md](docs.md) で詳細仕様を確認
2. トラブルシューティングセクションを参照
3. W&BログでLoss推移を確認
4. ヒートマップを可視化して定性評価

---

**Last Updated**: 2025-12-11
**Version**: 1.0.0
**Author**: prototype3 implementation team
