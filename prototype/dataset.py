"""
Dataset定義: PatientSplitter, SpineSequenceDataset (スライディングウィンドウ方式)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import albumentations as A


class PatientSplitter:
    """患者レベルで5-fold CV + Test分割"""

    def __init__(self, annotation_csv, n_folds=5, test_size=6, random_seed=42):
        self.annotation_csv = annotation_csv
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_seed = random_seed

    def split_patients(self):
        """
        患者を分割

        Returns:
            dict: {
                'test': [patient_ids],
                'folds': [
                    {'train': [patient_ids], 'val': [patient_ids]},
                    ...
                ]
            }
        """
        # アノテーションCSVから患者IDリストを取得
        df = pd.read_csv(self.annotation_csv)
        patient_ids = sorted(df['sample_id'].unique())

        np.random.seed(self.random_seed)
        patients_shuffled = np.random.permutation(patient_ids)

        # 1. テスト用6患者を抽出
        test_patients = patients_shuffled[:self.test_size].tolist()
        train_val_patients = patients_shuffled[self.test_size:]

        # 2. 残り35患者を5-foldに分割
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        folds = []

        for train_idx, val_idx in kfold.split(train_val_patients):
            folds.append({
                'train': train_val_patients[train_idx].tolist(),
                'val': train_val_patients[val_idx].tolist()
            })

        return {
            'test': test_patients,
            'folds': folds
        }


class SpineSequenceDataset(Dataset):
    """スライス単位のDataset (スライディングウィンドウ方式)

    各スライスを中心とした window_size 枚のウィンドウを返す。
    端のスライスはゼロパディングで対応。
    """

    def __init__(self, annotation_df, data_dir, transform=None, sequence_length=15):
        """
        Args:
            annotation_df: フィルタ済みのDataFrame (各行=1スライス)
            data_dir: .npyファイルが配置されているベースディレクトリの絶対パス
            transform: Albumentations transform (optional)
            sequence_length: ウィンドウサイズ (デフォルト: 15)
        """
        self.annotation_df = annotation_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.half_window = sequence_length // 2  # 7

        # 各スライスが属する椎骨のnpy_pathをマッピング
        # {(sample_id, vertebra): npy_path}
        self.npy_cache = {}
        for _, row in annotation_df.iterrows():
            key = (row['sample_id'], row['vertebra'])
            self.npy_cache[key] = row['npy_path']

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        """
        Returns:
            window: (sequence_length, 3, 128, 128) - 中心スライスidxを含む15枚のウィンドウ
            label: スカラー - 中心スライスのラベル
        """
        row = self.annotation_df.iloc[idx]
        sample_id = row['sample_id']
        vertebra = row['vertebra']
        center_slice_idx = int(row['slice_index'])  # 中心スライスのインデックス (0-63)
        center_label = float(row['label'])

        # .npyファイルをロード: (3, 64, 128, 128)
        # CSVのパスは './spine_data/...' のようになっているため、ファイル名部分を抽出
        npy_path_in_csv = self.npy_cache[(sample_id, vertebra)]
        # ex: './spine_data/sample1/C1.npy' -> 'sample1/C1.npy'
        path_parts = npy_path_in_csv.replace('\\', '/').split('/')
        if len(path_parts) > 2 and path_parts[-3] == 'spine_data':
             npy_rel_path = os.path.join(path_parts[-2], path_parts[-1])
        else:
             # もし予期しない形式なら、とりあえず最後の2要素を使う
             npy_rel_path = os.path.join(*path_parts[-2:])

        npy_path = os.path.join(self.data_dir, npy_rel_path)
        volume = np.load(npy_path).astype(np.float32)  # (3, 64, 128, 128)

        # ウィンドウの開始・終了インデックスを計算
        start_idx = center_slice_idx - self.half_window
        end_idx = center_slice_idx + self.half_window + 1  # 15枚: [-7, ..., 0, ..., +7]

        # ウィンドウを抽出 (パディング処理込み)
        window_slices = []
        for i in range(start_idx, end_idx):
            if i < 0 or i >= volume.shape[1]:  # 範囲外 → ゼロパディング
                window_slices.append(np.zeros((3, 128, 128), dtype=np.float32))
            else:
                window_slices.append(volume[:, i, :, :])  # (3, 128, 128)

        window = np.stack(window_slices, axis=0)  # (15, 3, 128, 128)

        # データ拡張 (15枚すべてに同じ変換を適用)
        if self.transform:
            transformed_slices = []
            for i in range(window.shape[0]):
                # (3, 128, 128) -> (128, 128, 3)
                slice_img = window[i].transpose(1, 2, 0)
                transformed = self.transform(image=slice_img)
                # (128, 128, 3) -> (3, 128, 128)
                transformed_slices.append(transformed['image'].transpose(2, 0, 1))
            window = np.stack(transformed_slices, axis=0)  # (15, 3, 128, 128)

        return torch.from_numpy(window), torch.tensor(center_label, dtype=torch.float32)


def get_train_transforms(cfg):
    """Albumentationsでデータ拡張"""
    return A.Compose([
        A.HorizontalFlip(p=cfg.augmentation.horizontal_flip_p),
        A.Rotate(limit=cfg.augmentation.rotate_limit, p=cfg.augmentation.rotate_p),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=cfg.augmentation.brightness_contrast_p
        )
    ])
