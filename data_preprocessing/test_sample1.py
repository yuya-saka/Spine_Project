#!/usr/bin/env python
"""sample1のみをテスト実行（修正版）"""

import os
import sys
import pandas as pd

# 設定（メインコードのパスに合わせてください）
# スクリプトの場所からプロジェクトルートを特定（実行場所に依存しない）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# プロジェクトルートをPYTHONPATHに追加
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# メインのクラスが書かれたファイルを generate_dataset.py として保存していると仮定
from data_preprocessing.generate_dataset import SpineDatasetGenerator

# プロジェクトルート基準の絶対パス
CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_output")
SEG_DIR = os.path.join(PROJECT_ROOT, "segmentations")
LABEL_DIR = os.path.join(PROJECT_ROOT, "fracture_labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "spine_data")  # 最新のメインコードの出力先

def main():
    generator = SpineDatasetGenerator(
        csv_file=CSV_FILE,
        nifti_dir=NIFTI_DIR,
        seg_dir=SEG_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR
    )

    # sample1のみ処理
    target_id = "sample1"
    
    print("=" * 60)
    print(f"{target_id} のテスト処理")
    print("=" * 60)

    metadata_list = generator.process_sample(target_id)

    if metadata_list:
        df = pd.DataFrame(metadata_list)
        print("\n生成されたメタデータ:")
        
        # 【修正点】 fracture_voxel_count -> fracture_overlap_ratio に変更
        display_cols = ['sample_id', 'vertebra', 'fracture_label', 'fracture_overlap_ratio']
        
        # 見やすく表示
        print(df[display_cols].to_string(index=False))
        
        print("-" * 30)
        print(f"Total: {len(df)} vertebrae")
        print(f"Fracture: {df['fracture_label'].sum()} vertebrae")
        
        # 保存先の案内
        print(f"\n確認用ファイルは {OUTPUT_DIR}/{target_id}/ に保存されました。")
        print("ITK-SNAP等で _bone.nii.gz と _mask.nii.gz を確認してください。")
    else:
        print("\n処理失敗、またはデータが見つかりませんでした。")

if __name__ == "__main__":
    main()