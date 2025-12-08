import os
import time
import pandas as pd
from generate_dataset import SpineDatasetGenerator  # 先ほど保存したファイルからクラスをインポート

# ==========================================
# 設定セクション (環境に合わせて変更可能)
# ==========================================
CONFIG = {
    "CSV_FILE": "nifti_list.csv",       # 入力CSV
    "NIFTI_DIR": "./nifti_output",      # CT画像ディレクトリ
    "SEG_DIR": "./segmentations",       # 椎骨マスクディレクトリ
    "LABEL_DIR": "./fracture_labels",   # 骨折ラベルディレクトリ
    "OUTPUT_DIR": "./spine_data",   # 出力先ディレクトリ
}

def check_directories():
    """必要なファイルやディレクトリの存在確認"""
    if not os.path.exists(CONFIG["CSV_FILE"]):
        print(f"エラー: CSVファイルが見つかりません -> {CONFIG['CSV_FILE']}")
        return False
    if not os.path.exists(CONFIG["NIFTI_DIR"]):
        print(f"エラー: NIfTIディレクトリが見つかりません -> {CONFIG['NIFTI_DIR']}")
        return False
    return True

def main():
    print("=" * 60)
    print("頸椎データセット生成処理を開始します")
    print("=" * 60)

    # 1. 事前チェック
    if not check_directories():
        print("処理を中断します。パスを確認してください。")
        return

    # 2. ジェネレータの初期化
    try:
        generator = SpineDatasetGenerator(
            csv_file=CONFIG["CSV_FILE"],
            nifti_dir=CONFIG["NIFTI_DIR"],
            seg_dir=CONFIG["SEG_DIR"],
            label_dir=CONFIG["LABEL_DIR"],
            output_dir=CONFIG["OUTPUT_DIR"]
        )
    except Exception as e:
        print(f"初期化エラー: {e}")
        return

    # 3. 処理実行 (時間計測付き)
    start_time = time.time()
    
    try:
        # クラス内の generate_all() が全ループ処理を行います
        generator.generate_all()
        
    except KeyboardInterrupt:
        print("\n\n処理がユーザーによって中断されました。")
    except Exception as e:
        print(f"\n\n予期せぬエラーが発生しました: {e}")
    finally:
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "=" * 60)
        print(f"処理終了 - 経過時間: {minutes}分 {seconds}秒")
        
        # 簡易レポート
        meta_csv = os.path.join(CONFIG["OUTPUT_DIR"], "dataset_metadata.csv")
        if os.path.exists(meta_csv):
            df = pd.read_csv(meta_csv)
            print(f"生成数: {len(df)} 椎骨データ")
            print(f"保存先: {CONFIG['OUTPUT_DIR']}")
            
            # 骨折陽性数の確認
            if 'fracture_label' in df.columns:
                pos_count = df['fracture_label'].sum()
                print(f"骨折陽性 (Fracture): {pos_count} / {len(df)}")
        else:
            print("メタデータCSVが生成されませんでした。エラーログを確認してください。")
        print("=" * 60)

if __name__ == "__main__":
    main()