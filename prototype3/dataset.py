import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import albumentations as A


class PatientSplitter:
    """
    患者レベルで5-fold CV + Test分割を行うクラス

    医療データのリーケージ防止のため、同一患者のデータは同じfoldに配置。
    prototype2から流用（変更なし）。

    Args:
        annotation_csv: スライスアノテーションCSVのパス
        n_folds: Fold数（デフォルト: 5）
        test_size: テストセットの患者数（デフォルト: 6）
        random_seed: ランダムシード（再現性確保）
    """
    def __init__(self, annotation_csv, n_folds=5, test_size=6, random_seed=42):
        self.annotation_csv = annotation_csv
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_seed = random_seed

    def split_patients(self):
        """
        患者IDリストを取得し、テスト用と訓練/検証用に分割

        Returns:
            dict: {
                'test': [test_patient_ids],
                'folds': [
                    {'train': [train_patient_ids], 'val': [val_patient_ids]},
                    ...  # n_folds個
                ]
            }
        """
        df = pd.read_csv(self.annotation_csv)
        patient_ids = sorted(df['sample_id'].unique())

        # ランダムシャッフル（固定シード）
        np.random.seed(self.random_seed)
        patients_shuffled = np.random.permutation(patient_ids)

        # テストセット分離（最初のtest_size人）
        test_patients = patients_shuffled[:self.test_size].tolist()

        # 訓練/検証セット（残りの患者）
        train_val_patients = patients_shuffled[self.test_size:]

        # K-Fold Cross Validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        folds = []
        for train_idx, val_idx in kfold.split(train_val_patients):
            folds.append({
                'train': train_val_patients[train_idx].tolist(),
                'val': train_val_patients[val_idx].tolist()
            })

        return {'test': test_patients, 'folds': folds}


class SpineSliceDataset(Dataset):
    """
    単一スライスを返すMIL用データセット

    prototype2のSpineSequenceDatasetから大幅簡略化。
    シーケンス（16枚のウィンドウ）ではなく、1枚のスライス単位で返す。

    Args:
        annotation_df: スライスアノテーションDataFrame
        data_dir: データディレクトリのパス
        transform: Albumentationsの変換パイプライン（オプション）
    """
    def __init__(self, annotation_df, data_dir, transform=None):
        self.annotation_df = annotation_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

        # 高速化のため、NPYパスをキャッシュ
        # (sample_id, vertebra) -> npy_path の辞書
        self.npy_cache = {}
        for _, row in annotation_df.iterrows():
            key = (row['sample_id'], row['vertebra'])
            if key not in self.npy_cache:
                self.npy_cache[key] = row['npy_path']

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        """
        指定インデックスのスライスを取得

        Returns:
            slice_img: (3, 128, 128) - Ch0: Bone, Ch1: Soft, Ch2: Mask
            label: float - スライスレベルの骨折ラベル（0 or 1）
        """
        row = self.annotation_df.iloc[idx]
        sample_id = row['sample_id']
        vertebra = row['vertebra']
        slice_idx = int(row['slice_index'])
        label = float(row['label'])

        # --- NPY読み込み ---
        # prototype2と同じパス解決ロジック
        npy_path_in_csv = self.npy_cache[(sample_id, vertebra)]
        path_parts = npy_path_in_csv.replace('\\', '/').split('/')
        if len(path_parts) > 2 and path_parts[-3] == 'spine_data':
            npy_rel_path = os.path.join(path_parts[-2], path_parts[-1])
        else:
            npy_rel_path = os.path.join(*path_parts[-2:])
        npy_path = os.path.join(self.data_dir, npy_rel_path)

        # ボリューム読み込み: (3, 64, 128, 128)
        volume = np.load(npy_path).astype(np.float32)

        # --- スライス抽出: (3, 64, 128, 128) → (3, 128, 128) ---
        slice_img = volume[:, slice_idx, :, :]

        # --- データ拡張（Albumentations） ---
        if self.transform:
            # Albumentationsは (H, W, C) 形式を期待
            slice_img = slice_img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
            transformed = self.transform(image=slice_img)
            slice_img = transformed['image'].transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

        # --- マスク前処理: Ch2をバイナリ化 ---
        # 0/1に明確化（データ拡張で0.5近辺が発生する可能性があるため）
        slice_img[2] = (slice_img[2] > 0.5).astype(np.float32)

        return torch.from_numpy(slice_img), torch.tensor(label, dtype=torch.float32)


def get_train_transforms(cfg):
    """
    訓練用のデータ拡張パイプライン

    prototype2と同じ構成:
    - HorizontalFlip: 左右反転
    - Rotate: 回転（±15度）
    - RandomBrightnessContrast: 輝度・コントラスト変更
    - Normalize: ImageNet統計で正規化

    注意: マスク（Ch2）も同時に変換される。
          幾何変換（Flip, Rotate）は正しく適用されるが、
          色変換（BrightnessContrast）はCh2にも影響する。
          ただしCh2は0/1なので影響は小さい。

    Args:
        cfg: Hydra設定オブジェクト

    Returns:
        A.Compose: 変換パイプライン
    """
    return A.Compose([
        A.HorizontalFlip(p=cfg.augmentation.horizontal_flip_p),
        A.Rotate(
            limit=cfg.augmentation.rotate_limit,
            p=cfg.augmentation.rotate_p,
            border_mode=0,  # BORDER_CONSTANT（黒で埋める）
            value=0
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=cfg.augmentation.brightness_contrast_p
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
    ])


def get_val_transforms(cfg):
    """
    検証用のデータ変換パイプライン

    拡張なし、正規化のみ。

    Args:
        cfg: Hydra設定オブジェクト

    Returns:
        A.Compose: 変換パイプライン
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0
        ),
    ])


if __name__ == "__main__":
    # --- 簡易テスト（データがある場合のみ実行可能） ---
    print("=== dataset.py 簡易テスト ===\n")

    # テスト1: PatientSplitter
    print("【テスト1】PatientSplitter")
    try:
        annotation_csv = "../spine_data/slice_annotations.csv"
        splitter = PatientSplitter(annotation_csv, n_folds=5, test_size=6, random_seed=42)
        splits = splitter.split_patients()
        print(f"✓ テスト患者数: {len(splits['test'])}")
        print(f"✓ Fold数: {len(splits['folds'])}")
        print(f"✓ Fold0 訓練患者数: {len(splits['folds'][0]['train'])}")
        print(f"✓ Fold0 検証患者数: {len(splits['folds'][0]['val'])}")
        print()
    except Exception as e:
        print(f"✗ PatientSplitterのテストをスキップ: {e}\n")

    # テスト2: SpineSliceDataset（データローディング）
    print("【テスト2】SpineSliceDataset")
    try:
        df = pd.read_csv("../spine_data/slice_annotations.csv")
        dataset = SpineSliceDataset(
            df.head(10),  # 最初の10サンプルのみ
            data_dir="../spine_data",
            transform=None
        )
        print(f"✓ データセットサイズ: {len(dataset)}")

        # 1サンプル取得
        img, label = dataset[0]
        print(f"✓ 画像shape: {img.shape} (期待: (3, 128, 128))")
        print(f"✓ ラベル: {label.item()}")
        print(f"✓ マスク（Ch2）の範囲: [{img[2].min():.3f}, {img[2].max():.3f}] (期待: [0, 1])")
        print()
    except Exception as e:
        print(f"✗ SpineSliceDatasetのテストをスキップ: {e}\n")

    print("=== テスト完了 ===")
