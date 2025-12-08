import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, affine_transform, label as nd_label
from scipy.spatial.transform import Rotation
import traceback

# --- 設定 ---
CSV_FILE = "nifti_list.csv"       # 入力CSV
NIFTI_DIR = "./nifti_output"      # CT画像 (nii.gz)
SEG_DIR = "./segmentations"       # 椎骨セグメンテーション (nii.gz)
LABEL_DIR = "./fracture_labels"   # 骨折ラベル (スライス全体マスク nii.gz)
OUTPUT_DIR = "./spine_data"   # 出力先

# 出力サイズ (X, Y, Z) mm
CROP_SIZE_MM = (128, 128, 64)
BOX_CENTER_MM = np.array([CROP_SIZE_MM[0]/2, CROP_SIZE_MM[1]/2, CROP_SIZE_MM[2]/2])

VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Windowing設定
BONE_WL, BONE_WW = 1000, 4000
SOFT_WL, SOFT_WW = 40, 400

# 骨折判定の閾値 (椎骨体積の何%が骨折スライスに含まれるか)
FRACTURE_OVERLAP_THRESHOLD = 0.10


class SpineDatasetGenerator:
    """頸椎データセット生成クラス (可視化NIfTI出力付き)"""

    def __init__(self, csv_file, nifti_dir, seg_dir, label_dir, output_dir):
        self.csv_file = csv_file
        self.nifti_dir = nifti_dir
        self.seg_dir = seg_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        self.csv_df = pd.read_csv(csv_file)
        os.makedirs(output_dir, exist_ok=True)

    def _load_nifti_data(self, sample_id):
        """CT画像と骨折ラベルの読み込み"""
        ct_path = os.path.join(self.nifti_dir, f"{sample_id}.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"CT画像なし: {ct_path}")
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()

        # 骨折ラベル (回転させず、重なり判定用に使用)
        fracture_path = os.path.join(self.label_dir, f"{sample_id}.nii.gz")
        if os.path.exists(fracture_path):
            fracture_img = nib.load(fracture_path)
            fracture_data = fracture_img.get_fdata()
        else:
            fracture_data = np.zeros_like(ct_data)

        return ct_img, ct_data, fracture_data

    def _load_vertebra_mask(self, sample_id, vertebra_id):
        """椎骨マスクの読み込み"""
        mask_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vertebra_id}.nii.gz")
        if not os.path.exists(mask_path):
            return None, None
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        return mask_img, mask_data

    def _get_largest_component(self, mask_data):
        """マスクの最大連結成分のみを抽出"""
        labeled, num_components = nd_label(mask_data > 0)
        if num_components <= 1:
            return mask_data
        
        component_sizes = [(labeled == i).sum() for i in range(1, num_components + 1)]
        largest_component = np.argmax(component_sizes) + 1
        return (labeled == largest_component).astype(mask_data.dtype)

    def _compute_physical_centroid(self, mask_img, mask_data):
        """物理座標空間(mm)での重心を計算"""
        idx_centroid = center_of_mass(mask_data)
        if np.any(np.isnan(idx_centroid)):
            raise ValueError("重心計算エラー")
        idx_homogeneous = np.array([*idx_centroid, 1.0])
        centroid_mm = mask_img.affine @ idx_homogeneous
        return centroid_mm[:3]

    def _compute_all_centroids(self, sample_id):
        """全椎骨の物理重心を計算"""
        centroids_mm = {}
        for vid in VERTEBRAE:
            try:
                mask_img, mask_data = self._load_vertebra_mask(sample_id, vid)
                if mask_img is None:
                    centroids_mm[vid] = None
                    continue
                if np.sum(mask_data) < 10:
                    centroids_mm[vid] = None
                    continue
                mask_data = self._get_largest_component(mask_data)
                centroids_mm[vid] = self._compute_physical_centroid(mask_img, mask_data)
            except Exception:
                centroids_mm[vid] = None
        return centroids_mm

    def _compute_spine_vector(self, vertebra_id, centroids):
        """脊椎の方向ベクトル（下から上）を計算"""
        idx = int(vertebra_id[1])
        if idx == 7:
            upper, lower = "C6", "C7"
        else:
            upper, lower = f"C{idx}", f"C{idx+1}"

        p_upper, p_lower = centroids.get(upper), centroids.get(lower)
        if p_upper is None or p_lower is None:
            return None

        vec = p_upper - p_lower
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-6 else None

    def _compute_rotation_matrix(self, vertebra_id, centroids):
        """回転行列計算"""
        if vertebra_id in ["C1", "C2"]:
            if centroids.get("C3") is not None:
                return self._compute_rotation_matrix("C3", centroids)
        
        spine_vec = self._compute_spine_vector(vertebra_id, centroids)
        if spine_vec is None: return None
            
        target_vec = np.array([0, 0, 1])
        rotation, _ = Rotation.align_vectors([target_vec], [spine_vec])
        return rotation.as_matrix()

    def _build_resampling_matrix(self, centroid_phys, R_phys, src_affine):
        """変換行列構築"""
        T_center = np.eye(4)
        T_center[:3, 3] = -BOX_CENTER_MM
        R_inv = np.eye(4)
        R_inv[:3, :3] = R_phys.T 
        T_loc = np.eye(4)
        T_loc[:3, 3] = centroid_phys
        src_affine_inv = np.linalg.inv(src_affine)
        return src_affine_inv @ T_loc @ R_inv @ T_center

    def _apply_windowing(self, data, wl, ww):
        lower, upper = wl - ww/2, wl + ww/2
        return np.clip((data - lower) / (upper - lower), 0, 1).astype(np.float32)

    def _extract_volume_channels(self, ct_data, mask_data, M_total):
        """ボリューム切り出し (3チャンネル)"""
        matrix = M_total[:3, :3]
        offset = M_total[:3, 3]
        out_shape = CROP_SIZE_MM

        volume = np.zeros((3, *out_shape), dtype=np.float32)

        # Ch 0: Bone
        bone_raw = affine_transform(ct_data, matrix, offset=offset, output_shape=out_shape, order=1, cval=-1024)
        volume[0] = self._apply_windowing(bone_raw, BONE_WL, BONE_WW)

        # Ch 1: Soft
        soft_raw = affine_transform(ct_data, matrix, offset=offset, output_shape=out_shape, order=1, cval=-1024)
        volume[1] = self._apply_windowing(soft_raw, SOFT_WL, SOFT_WW)

        # Ch 2: Mask (Vertebra)
        mask_resampled = affine_transform(mask_data, matrix, offset=offset, output_shape=out_shape, order=0, cval=0)
        volume[2] = (mask_resampled > 0.5).astype(np.float32)

        return volume

    def _determine_fracture_label(self, vertebra_mask, fracture_label_map):
        """骨折判定 (元画像空間での重なり率)"""
        vertebra_voxels = np.sum(vertebra_mask > 0)
        if vertebra_voxels == 0: return 0, 0.0

        overlap = np.logical_and(vertebra_mask > 0, fracture_label_map > 0)
        overlap_voxels = np.sum(overlap)
        overlap_ratio = overlap_voxels / vertebra_voxels
        
        is_fracture = 1 if overlap_ratio > FRACTURE_OVERLAP_THRESHOLD else 0
        return is_fracture, overlap_ratio

    def _save_debug_nifti(self, save_dir, vertebra_id, volume):
        """
        確認用のNIfTIファイルを保存
        Args:
            volume: (3, X, Y, Z) の配列。transpose前のもの。
        """
        # Affineは単位行列（1mm等方、正規化済みのため）
        affine = np.eye(4)

        # Channel 0: Bone Window
        # shapeは (X, Y, Z) である必要があるため、volume[0]をそのまま渡す
        nib.save(
            nib.Nifti1Image(volume[0], affine),
            os.path.join(save_dir, f"{vertebra_id}_bone.nii.gz")
        )

        # Channel 2: Mask
        nib.save(
            nib.Nifti1Image(volume[2], affine),
            os.path.join(save_dir, f"{vertebra_id}_mask.nii.gz")
        )

    def process_sample(self, sample_id):
        print(f"Processing ID: {sample_id} ...")
        metadata_list = []

        try:
            ct_img, ct_data, fracture_data = self._load_nifti_data(sample_id)
            centroids_mm = self._compute_all_centroids(sample_id)
            
            for vid in VERTEBRAE:
                if centroids_mm.get(vid) is None: continue

                # 1. 切り出し処理
                R_phys = self._compute_rotation_matrix(vid, centroids_mm)
                if R_phys is None: continue

                M_total = self._build_resampling_matrix(centroids_mm[vid], R_phys, ct_img.affine)
                _, mask_data = self._load_vertebra_mask(sample_id, vid)
                mask_data = self._get_largest_component(mask_data)

                # volume shape: (Channel, X, Y, Z)
                volume = self._extract_volume_channels(ct_data, mask_data, M_total)

                # 2. 骨折判定
                is_frac, overlap_ratio = self._determine_fracture_label(mask_data, fracture_data)

                # 3. 保存
                save_dir = os.path.join(self.output_dir, sample_id)
                os.makedirs(save_dir, exist_ok=True)

                # AI学習用 .npy (Channel, Depth, Height, Width) に変換
                volume_zyx = np.transpose(volume, (0, 3, 2, 1))
                np.save(os.path.join(save_dir, f"{vid}.npy"), volume_zyx)

                # 【追加】 可視化用 .nii.gz (Channel, X, Y, Z) のまま保存
                self._save_debug_nifti(save_dir, vid, volume)

                metadata_list.append({
                    'sample_id': sample_id,
                    'vertebra': vid,
                    'fracture_label': int(is_frac),
                    'fracture_overlap_ratio': float(overlap_ratio),
                    'file_path': f"{sample_id}/{vid}.npy"
                })
                
                status = "FRACTURE" if is_frac else "Normal"
                print(f"  {vid}: {status} (Overlap: {overlap_ratio:.1%})")

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            traceback.print_exc()

        return metadata_list

    def generate_all(self):
        target_ids = self.csv_df[self.csv_df['Exclude'] != True]['ID'].tolist()
        all_metadata = []
        print(f"Total samples: {len(target_ids)}")
        
        for sample_id in target_ids:
            meta = self.process_sample(str(sample_id))
            all_metadata.extend(meta)

        df = pd.DataFrame(all_metadata)
        df.to_csv(os.path.join(self.output_dir, "dataset_metadata.csv"), index=False)
        print("\nCompleted.")

if __name__ == "__main__":
    generator = SpineDatasetGenerator(CSV_FILE, NIFTI_DIR, SEG_DIR, LABEL_DIR, OUTPUT_DIR)
    generator.generate_all()