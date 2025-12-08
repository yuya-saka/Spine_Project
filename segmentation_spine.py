import os
import pandas as pd
import time
import traceback
from totalsegmentator.python_api import totalsegmentator

# --- 設定 ---
GPU_ID = 1                           # 使用するGPU番号 (0, 1, 2)
CSV_FILE = "nifti_list.csv"      # 転送したCSV
NIFTI_DIR = "./nifti_output"           # 転送したNIfTIフォルダ
OUTPUT_DIR = "./segmentations"      # セグメンテーション結果の保存先

# GPU設定（環境変数で上書き可能）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

def check_gpu_environment():
    """GPU環境をチェックして情報を表示"""
    try:
        import torch

        print("=== GPU環境情報 ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            print(f"Selected GPU ID: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")

            # 使用するGPU情報を表示
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  WARNING: CUDA is not available. Processing will use CPU (very slow).")

        print("=" * 60)
        return torch.cuda.is_available()

    except ImportError:
        print("⚠️  WARNING: PyTorch is not installed. GPU check skipped.")
        print("=" * 60)
        return False

def main():
    # 1. GPU環境チェック
    gpu_available = check_gpu_environment()

    # 2. 保存先フォルダの作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. CSV読み込み
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} が見つかりません。")
        return

    df = pd.read_csv(CSV_FILE)
    total_files = len(df)

    print(f"開始: 全 {total_files} 件のセグメンテーション処理")
    print("-" * 60)

    # 4. ループ処理
    processing_times = []  # 処理時間を記録
    failed_files = []      # 失敗したファイルを記録
    processed_count = 0    # 実際に処理した件数

    for idx, row in df.iterrows():
        sample_id = str(row['ID'])

        # ファイル名を ID から生成
        nifti_filename = f"{sample_id}.nii.gz"
        input_path = os.path.join(NIFTI_DIR, nifti_filename)
        output_subdir = os.path.join(OUTPUT_DIR, sample_id)

        # 進捗表示
        progress_pct = (idx + 1) / total_files * 100
        print(f"[{idx+1}/{total_files} ({progress_pct:.1f}%)] Processing: {sample_id} ...")

        # NIfTIファイルが存在するか確認
        if not os.path.exists(input_path):
            print(f"  [Skip] Input file not found: {input_path}")
            continue

        # 既に処理済み（フォルダがあり、かつ中身がある）ならスキップ
        # ※ 再実行したい場合はこのフォルダを削除してください
        if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
            print(f"  [Skip] Already segmented.")
            continue

        start_time = time.time()

        try:
            # TotalSegmentatorの実行
            # C1~C7のみを指定して高速化・容量節約
            totalsegmentator(
                input_path,
                output_subdir,
                roi_subset=[
                    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3",
                    "vertebrae_C4", "vertebrae_C5", "vertebrae_C6",
                    "vertebrae_C7"
                ],
                fast=False,  # 研究用なので高精度モード(False)推奨。A6000なら十分速い。
                quiet=True   # ログを少し静かにする
            )

            elapsed = time.time() - start_time
            processing_times.append(elapsed)
            processed_count += 1

            # 残り時間を推定
            if len(processing_times) > 0:
                avg_time = sum(processing_times) / len(processing_times)
                remaining_files = total_files - (idx + 1)
                estimated_remaining = avg_time * remaining_files

                print(f"  -> Done! ({elapsed:.1f} sec) [平均: {avg_time:.1f}s, 残り推定: {estimated_remaining/60:.1f}分]")
            else:
                print(f"  -> Done! ({elapsed:.1f} sec)")

        except Exception as e:
            print(f"  [Error] Failed processing {sample_id}: {e}")
            failed_files.append((sample_id, str(e)))
            # 詳細なトレースバック（デバッグ用）
            if os.environ.get("DEBUG", "0") == "1":
                traceback.print_exc()

    print("-" * 60)
    print(f"全処理完了しました。")
    print(f"  処理済み: {processed_count} 件")
    if failed_files:
        print(f"  失敗: {len(failed_files)} 件")
        print("\n失敗したファイル:")
        for sample_id, error in failed_files:
            print(f"  - {sample_id}: {error}")

if __name__ == "__main__":
    main()