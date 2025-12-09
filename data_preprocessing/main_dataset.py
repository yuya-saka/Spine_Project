import os
import sys
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# スクリプトの場所からプロジェクトルートをPYTHONPATHに追加
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing.generate_dataset import SpineDatasetGenerator
from data_preprocessing.generation_labels import SliceLabelGeneratorFast, process_wrapper

# ==========================================
# 設定セクション (環境に合わせて変更可能)
# ==========================================
# プロジェクトルート基準の絶対パス
CONFIG = {
    "CSV_FILE": os.path.join(PROJECT_ROOT, "nifti_list.csv"),       # 入力CSV
    "NIFTI_DIR": os.path.join(PROJECT_ROOT, "nifti_output"),      # CT画像ディレクトリ
    "SEG_DIR": os.path.join(PROJECT_ROOT, "segmentations"),       # 椎骨マスクディレクトリ
    "LABEL_DIR": os.path.join(PROJECT_ROOT, "fracture_labels"),   # 骨折ラベルディレクトリ
    "OUTPUT_DIR": os.path.join(PROJECT_ROOT, "spine_data"),       # 出力先ディレクトリ
    "NUM_WORKERS": None,                # 並列処理のワーカー数 (Noneの場合は全コア使用)
}

# 共通定数
CROP_SIZE_MM = (128, 128, 64)
VERTEBRAE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def check_directories():
    """必要なファイルやディレクトリの存在確認"""
    if not os.path.exists(CONFIG["CSV_FILE"]):
        print(f"エラー: CSVファイルが見つかりません -> {CONFIG['CSV_FILE']}")
        return False
    if not os.path.exists(CONFIG["NIFTI_DIR"]):
        print(f"エラー: NIfTIディレクトリが見つかりません -> {CONFIG['NIFTI_DIR']}")
        return False
    if not os.path.exists(CONFIG["SEG_DIR"]):
        print(f"エラー: セグメンテーションディレクトリが見つかりません -> {CONFIG['SEG_DIR']}")
        return False
    return True


def run_phase1_volume_generation():
    """
    Phase 1: ボリュームデータ生成
    CT画像から3Dボリュームデータ(.npy)を生成し、椎骨レベルの骨折判定を実行
    """
    print("\n" + "=" * 60)
    print("Phase 1: ボリュームデータ生成を開始")
    print("=" * 60)

    try:
        generator = SpineDatasetGenerator(
            csv_file=CONFIG["CSV_FILE"],
            nifti_dir=CONFIG["NIFTI_DIR"],
            seg_dir=CONFIG["SEG_DIR"],
            label_dir=CONFIG["LABEL_DIR"],
            output_dir=CONFIG["OUTPUT_DIR"]
        )

        # ボリュームデータ生成実行
        generator.generate_all()

        # 結果確認
        meta_csv = os.path.join(CONFIG["OUTPUT_DIR"], "dataset_metadata.csv")
        if os.path.exists(meta_csv):
            df = pd.read_csv(meta_csv)
            print("\n" + "-" * 60)
            print(f"Phase 1 完了: {len(df)} 椎骨データを生成")
            if 'fracture_label' in df.columns:
                pos_count = df['fracture_label'].sum()
                print(f"骨折陽性データ: {pos_count} / {len(df)}")
            print("-" * 60)
            return True
        else:
            print("\n警告: dataset_metadata.csv が生成されませんでした")
            return False

    except Exception as e:
        print(f"\nPhase 1 エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase2_slice_labeling():
    """
    Phase 2: スライスラベル生成
    生成済みの.npyファイルからスライスレベルの骨折ラベルを並列生成
    """
    print("\n" + "=" * 60)
    print("Phase 2: スライスラベル生成を開始 (並列処理)")
    print("=" * 60)

    try:
        # CSV読み込み
        if not os.path.exists(CONFIG["CSV_FILE"]):
            print("エラー: CSVファイルがありません")
            return False

        df = pd.read_csv(CONFIG["CSV_FILE"])
        target_ids = df[df['Exclude'] != True]['ID'].tolist()

        # 既存データの確認
        if not os.path.exists(CONFIG["OUTPUT_DIR"]):
            print(f"エラー: 出力ディレクトリが見つかりません: {CONFIG['OUTPUT_DIR']}")
            return False

        print(f"対象サンプル数: {len(target_ids)}")
        print(f"並列処理ワーカー数: {CONFIG['NUM_WORKERS'] if CONFIG['NUM_WORKERS'] else '全コア'}")

        # ラベル生成器のインスタンス化
        label_gen = SliceLabelGeneratorFast(
            seg_dir=CONFIG["SEG_DIR"],
            label_dir=CONFIG["LABEL_DIR"],
            data_dir=CONFIG["OUTPUT_DIR"]
        )

        # 並列処理実行
        all_records = []
        args_list = [(label_gen, str(pid)) for pid in target_ids]

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
            # tqdmで進捗バー表示
            results = list(tqdm(
                executor.map(process_wrapper, args_list),
                total=len(target_ids),
                desc="スライスラベル生成"
            ))

        # 結果の結合
        for res in results:
            all_records.extend(res)

        # 保存
        out_df = pd.DataFrame(all_records)
        out_path = os.path.join(CONFIG["OUTPUT_DIR"], "slice_annotations.csv")
        out_df.to_csv(out_path, index=False)

        print("\n" + "-" * 60)
        print(f"Phase 2 完了: {len(out_df)} スライスのラベルを生成")
        print(f"骨折スライス数: {out_df['label'].sum()}")
        print("-" * 60)

        return True

    except Exception as e:
        print(f"\nPhase 2 エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_report(start_time, phase1_success, phase2_success):
    """最終レポートの出力"""
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 60)
    print("データセット生成パイプライン 実行結果")
    print("=" * 60)

    # Phase 1 レポート
    if phase1_success:
        meta_csv = os.path.join(CONFIG["OUTPUT_DIR"], "dataset_metadata.csv")
        if os.path.exists(meta_csv):
            df = pd.read_csv(meta_csv)
            print("\n✓ Phase 1: ボリュームデータ生成 - 成功")
            print(f"  - 生成数: {len(df)} 椎骨データ")
            if 'fracture_label' in df.columns:
                pos_count = df['fracture_label'].sum()
                print(f"  - 骨折陽性: {pos_count} / {len(df)}")
            print(f"  - 出力: {meta_csv}")
    else:
        print("\n✗ Phase 1: ボリュームデータ生成 - 失敗")

    # Phase 2 レポート
    if phase2_success:
        slice_csv = os.path.join(CONFIG["OUTPUT_DIR"], "slice_annotations.csv")
        if os.path.exists(slice_csv):
            df = pd.read_csv(slice_csv)
            print("\n✓ Phase 2: スライスラベル生成 - 成功")
            print(f"  - 総スライス数: {len(df)}")
            print(f"  - 骨折スライス数: {df['label'].sum()}")
            print(f"  - 出力: {slice_csv}")
    else:
        print("\n✗ Phase 2: スライスラベル生成 - 失敗またはスキップ")

    print(f"\n総実行時間: {minutes}分 {seconds}秒")
    print("=" * 60)


def main():
    """メイン処理: 2段階パイプラインの実行"""
    print("=" * 60)
    print("頸椎データセット生成パイプライン")
    print("=" * 60)

    # 事前チェック
    if not check_directories():
        print("\n処理を中断します。パスを確認してください。")
        return

    start_time = time.time()
    phase1_success = False
    phase2_success = False

    try:
        # Phase 1: ボリュームデータ生成
        phase1_success = run_phase1_volume_generation()

        # Phase 1が成功した場合のみPhase 2を実行
        if phase1_success:
            # Phase 2: スライスラベル生成
            phase2_success = run_phase2_slice_labeling()
        else:
            print("\nPhase 1が失敗したため、Phase 2をスキップします")

    except KeyboardInterrupt:
        print("\n\n処理がユーザーによって中断されました")

    except Exception as e:
        print(f"\n\n予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 最終レポート出力
        print_final_report(start_time, phase1_success, phase2_success)


if __name__ == "__main__":
    main()
