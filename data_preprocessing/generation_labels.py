import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass, affine_transform, label as nd_label
from scipy.spatial.transform import Rotation
import traceback
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # 進捗バー表示用

# =================================================================
# 設定
# =================================================================
# スクリプトの場所からプロジェクトルートを特定（実行場所に依存しない）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# プロジェクトルート基準の絶対パス
CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
LABEL_DIR = os.path.join(PROJECT_ROOT, "fracture_labels")
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "spine_data")
OUTPUT_CSV_NAME = "slice_annotations.csv"

# 並列プロセス数 (サーバーのCPUコア数に合わせて調整。Noneなら全コア使用)
NUM_WORKERS = None 

# 幾何パラメータ (生成時と同じ)
CROP_SIZE_MM = (128, 128, 64)
BOX_CENTER_MM = np.array([CROP_SIZE_MM[0]/2, CROP_SIZE_MM[1]/2, CROP_SIZE_MM[2]/2])
VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
FRACTURE_OVERLAP_THRESHOLD = 0.10
# =================================================================

class SliceLabelGeneratorFast:
    def __init__(self, seg_dir, label_dir, data_dir):
        self.seg_dir = seg_dir
        self.label_dir = label_dir
        self.data_dir = data_dir

    def _get_largest_component(self, mask_data):
        labeled, n = nd_label(mask_data > 0)
        if n <= 1: return mask_data
        sizes = [(labeled == i).sum() for i in range(1, n+1)]
        largest = np.argmax(sizes) + 1
        return (labeled == largest).astype(mask_data.dtype)

    def _compute_physical_centroid(self, affine, mask_data):
        c = center_of_mass(mask_data)
        if np.any(np.isnan(c)): return None
        return (affine @ np.array([*c, 1.0]))[:3]

    def _compute_rotation_matrix(self, vertebra_id, centroids):
        if vertebra_id in ["C1", "C2"] and centroids.get("C3") is not None:
            return self._compute_rotation_matrix("C3", centroids)
        idx = int(vertebra_id[1])
        upper, lower = ("C6", "C7") if idx == 7 else (f"C{idx}", f"C{idx+1}")
        if centroids.get(upper) is None or centroids.get(lower) is None: return None
        vec = centroids[upper] - centroids[lower]
        norm = np.linalg.norm(vec)
        if norm < 1e-6: return None
        R, _ = Rotation.align_vectors([np.array([0,0,1])], [vec / norm])
        return R.as_matrix()

    def _build_matrix(self, centroid, R, src_affine):
        T_center = np.eye(4); T_center[:3,3] = -BOX_CENTER_MM
        R_inv = np.eye(4); R_inv[:3,:3] = R.T
        T_loc = np.eye(4); T_loc[:3,3] = centroid
        return np.linalg.inv(src_affine) @ T_loc @ R_inv @ T_center

    def process_single_patient(self, sample_id):
        """1人の患者データを処理（並列実行される関数）"""
        slice_records = []
        
        try:
            # 1. 骨折ラベルを一括読み込み (ここで1回だけ読む！)
            fracture_path = os.path.join(self.label_dir, f"{sample_id}.nii.gz")
            if os.path.exists(fracture_path):
                f_img = nib.load(fracture_path)
                f_full_data = f_img.get_fdata()
                affine_ref = f_img.affine # 座標系参照用
            else:
                # 骨折ラベルがない場合は空データ
                # ただしAffine参照のためにC1等のマスクを読む必要がある
                # 簡易的に最初のマスクを探す
                first_seg = os.path.join(self.seg_dir, sample_id, "vertebrae_C1.nii.gz")
                if os.path.exists(first_seg):
                    f_img = nib.load(first_seg)
                    f_full_data = np.zeros(f_img.shape)
                    affine_ref = f_img.affine
                else:
                    return [] # データなし

            # 2. 全椎骨の重心計算 (マスク読み込みが必要)
            centroids = {}
            # メモリ節約のため、マスクは必要なときに読む方式に戻すが、
            # Affineは共通なので使い回す
            for vid in VERTEBRAE:
                v_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vid}.nii.gz")
                if not os.path.exists(v_path):
                    centroids[vid] = None
                    continue
                v_img = nib.load(v_path)
                v_data = v_img.get_fdata()
                if np.sum(v_data) < 10:
                    centroids[vid] = None; continue
                
                v_data = self._get_largest_component(v_data)
                centroids[vid] = self._compute_physical_centroid(v_img.affine, v_data)

            # 3. 各椎骨の処理
            for vid in VERTEBRAE:
                # 生成済みデータのパス確認
                npy_rel_path = f"{sample_id}/{vid}.npy"
                npy_full_path = os.path.join(self.data_dir, npy_rel_path)
                if not os.path.exists(npy_full_path): continue

                if centroids.get(vid) is None: continue
                R = self._compute_rotation_matrix(vid, centroids)
                if R is None: continue

                # 再度マスク読み込み (メモリ消費を抑えるためここで読む)
                v_path = os.path.join(self.seg_dir, sample_id, f"vertebrae_{vid}.nii.gz")
                v_img = nib.load(v_path)
                v_data = v_img.get_fdata()
                v_data = self._get_largest_component(v_data)

                # 変換行列
                M_total = self._build_matrix(centroids[vid], R, affine_ref)
                matrix, offset = M_total[:3, :3], M_total[:3, 3]

                # 切り出し (Nearest Neighbor)
                # マスク
                v_crop = affine_transform(v_data, matrix, offset=offset, output_shape=CROP_SIZE_MM, order=0)
                # 骨折ラベル (メモリ上のf_full_dataを使用)
                f_crop = affine_transform(f_full_data, matrix, offset=offset, output_shape=CROP_SIZE_MM, order=0)

                # スライス判定
                for z in range(CROP_SIZE_MM[2]):
                    v_slice = v_crop[:, :, z] > 0.5
                    
                    label = 0
                    if np.sum(v_slice) >= 10:
                        f_slice = f_crop[:, :, z] > 0.5
                        overlap = np.logical_and(v_slice, f_slice)
                        ratio = np.sum(overlap) / np.sum(v_slice)
                        if ratio > FRACTURE_OVERLAP_THRESHOLD:
                            label = 1
                    
                    slice_records.append({
                        "sample_id": sample_id,
                        "vertebra": vid,
                        "npy_path": npy_full_path,
                        "slice_index": z,
                        "label": label
                    })

        except Exception as e:
            # 並列処理中はprintが見づらくなるのでエラー文字列だけ返すなどの工夫も可
            print(f"\nError in {sample_id}: {e}")
            traceback.print_exc()

        return slice_records

def process_wrapper(args):
    """並列処理用のラッパー関数"""
    generator, sample_id = args
    return generator.process_single_patient(sample_id)

def main():
    # CSV読み込み
    if not os.path.exists(CSV_FILE):
        print("CSVファイルがありません")
        return
    df = pd.read_csv(CSV_FILE)
    target_ids = df[df['Exclude'] != True]['ID'].tolist()
    
    # 既存データの確認
    if not os.path.exists(GENERATED_DATA_DIR):
        print(f"Data directory not found: {GENERATED_DATA_DIR}")
        return

    print(f"Target Samples: {len(target_ids)}")
    print(f"Processing with {NUM_WORKERS if NUM_WORKERS else 'ALL'} CPU cores...")

    # インスタンス化 (データを持たせず、メソッドだけ提供する形)
    gen = SliceLabelGeneratorFast(SEG_DIR, LABEL_DIR, GENERATED_DATA_DIR)

    # 並列処理実行
    all_records = []
    
    # 引数リスト作成
    args_list = [(gen, str(pid)) for pid in target_ids]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # tqdmで進捗バーを表示
        results = list(tqdm(executor.map(process_wrapper, args_list), total=len(target_ids)))
    
    # 結果の結合
    for res in results:
        all_records.extend(res)

    # 保存
    out_df = pd.DataFrame(all_records)
    out_path = os.path.join(GENERATED_DATA_DIR, OUTPUT_CSV_NAME)
    out_df.to_csv(out_path, index=False)
    
    print("\n" + "="*50)
    print(f"Completed! Labels saved to: {out_path}")
    print(f"Total slices: {len(out_df)}")
    print(f"Fracture slices: {out_df['label'].sum()}")
    print("="*50)

if __name__ == "__main__":
    # Windows/Macの場合はfreeze_supportが必要な場合があるがLinuxなら不要
    main()