import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import albumentations as A

class PatientSplitter:
    """(変更なし) 患者レベルで5-fold CV + Test分割"""
    def __init__(self, annotation_csv, n_folds=5, test_size=6, random_seed=42):
        self.annotation_csv = annotation_csv
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_seed = random_seed

    def split_patients(self):
        df = pd.read_csv(self.annotation_csv)
        patient_ids = sorted(df['sample_id'].unique())
        np.random.seed(self.random_seed)
        patients_shuffled = np.random.permutation(patient_ids)
        test_patients = patients_shuffled[:self.test_size].tolist()
        train_val_patients = patients_shuffled[self.test_size:]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        folds = []
        for train_idx, val_idx in kfold.split(train_val_patients):
            folds.append({
                'train': train_val_patients[train_idx].tolist(),
                'val': train_val_patients[val_idx].tolist()
            })
        return {'test': test_patients, 'folds': folds}

class SpineSequenceDataset(Dataset):
    def __init__(self, annotation_df, data_dir, transform=None, sequence_length=16):
        self.annotation_df = annotation_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.half_window = sequence_length // 2

        # 高速化のため、(sample_id, slice_index) -> label の辞書を作成
        # これにより、ウィンドウ内の隣接スライスのラベルを即座に引けるようにする
        self.label_map = {}
        for _, row in self.annotation_df.iterrows():
            self.label_map[(row['sample_id'], row['slice_index'])] = float(row['label'])

        # npyパスのキャッシュ
        self.npy_cache = {}
        for _, row in annotation_df.iterrows():
            # 重複を除去してキャッシュ
            key = (row['sample_id'], row['vertebra'])
            if key not in self.npy_cache:
                self.npy_cache[key] = row['npy_path']

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        row = self.annotation_df.iloc[idx]
        sample_id = row['sample_id']
        vertebra = row['vertebra']
        center_slice_idx = int(row['slice_index'])

        # --- 画像読み込みパート ---
        npy_path_in_csv = self.npy_cache[(sample_id, vertebra)]
        path_parts = npy_path_in_csv.replace('\\', '/').split('/')
        if len(path_parts) > 2 and path_parts[-3] == 'spine_data':
             npy_rel_path = os.path.join(path_parts[-2], path_parts[-1])
        else:
             npy_rel_path = os.path.join(*path_parts[-2:])
        
        npy_path = os.path.join(self.data_dir, npy_rel_path)
        
        # もしファイルがロードできなかった場合の安全策を入れると良いが、ここでは省略
        volume = np.load(npy_path).astype(np.float32)

        start_idx = center_slice_idx - self.half_window
        end_idx = center_slice_idx + self.half_window
        
        # バランスをとるため sequence_length 個取得 (end_idxは含まないrange)
        # sequence_length=16なら、前から8、後ろから8など
        
        window_slices = []
        window_labels = []

        for i in range(start_idx, end_idx):
            # 画像取得
            if i < 0 or i >= volume.shape[1]:
                window_slices.append(np.zeros((3, 128, 128), dtype=np.float32))
                window_labels.append(0.0) # パディング部分は正常扱い
            else:
                window_slices.append(volume[:, i, :, :])
                # ラベル取得: 辞書から引く。存在しなければ0(正常)とする
                label = self.label_map.get((sample_id, i), 0.0)
                window_labels.append(label)

        window = np.stack(window_slices, axis=0) # (T, 3, 128, 128)
        
        # --- データ拡張 ---
        if self.transform:
            T, C, H, W = window.shape
            combined_img = window.transpose(2, 3, 0, 1).reshape(H, W, T * C)
            transformed = self.transform(image=combined_img)
            combined_img = transformed['image']
            window = combined_img.reshape(H, W, T, C).transpose(2, 3, 0, 1)

        window_tensor = torch.from_numpy(window)
        
        # --- ラベル生成 ---
        # MIL学習用: ウィンドウ内に骨折(1)があれば、bag_label=1、なければ0
        window_labels = np.array(window_labels)
        bag_label = 1.0 if np.max(window_labels) > 0.5 else 0.0
        
        # 評価用に詳細ラベルも返すが、学習では bag_label を使う
        return window_tensor, torch.tensor(bag_label, dtype=torch.float32)

def get_train_transforms(cfg):
    return A.Compose([
        A.HorizontalFlip(p=cfg.augmentation.horizontal_flip_p),
        A.Rotate(limit=cfg.augmentation.rotate_limit, p=cfg.augmentation.rotate_p),
        A.RandomBrightnessContrast(p=cfg.augmentation.brightness_contrast_p),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
    ])

def get_val_transforms(cfg):
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
    ])