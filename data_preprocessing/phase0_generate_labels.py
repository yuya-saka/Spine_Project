import os
import numpy as np
import pandas as pd
import nibabel as nib

# --- 設定 ---
# スクリプトの場所からプロジェクトルートを特定（実行場所に依存しない）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# プロジェクトルート基準の絶対パス
CSV_FILE = os.path.join(PROJECT_ROOT, "nifti_list.csv")
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_output")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "fracture_labels")

def parse_z_value(value):
    """
    CSV の Z_Start/Z_End の値をパースする。
    単一値（float）または複数値（文字列 "a/b/c"）に対応。

    Args:
        value: CSVから読み込んだ値（float, str, または NaN）

    Returns:
        list: 数値のリスト、またはNone（欠損値の場合）

    Raises:
        ValueError: パースに失敗した場合
    """
    # NaN チェック
    if pd.isna(value):
        return None

    # float型の場合（単一区間）
    if isinstance(value, (int, float)):
        return [int(value)]

    # 文字列の場合（複数区間の可能性）
    if isinstance(value, str):
        # スラッシュで分割
        parts = value.split('/')
        try:
            # 小数点を含む文字列に対応: float経由でintに変換
            return [int(float(p.strip())) for p in parts]
        except ValueError as e:
            raise ValueError(f"数値への変換に失敗しました: '{value}'") from e

    raise ValueError(f"未対応の型です: {type(value)}")

def main():
    # 1. 保存先フォルダの作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. CSV読み込み
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} が見つかりません。")
        return

    df = pd.read_csv(CSV_FILE)
    total_files = len(df)

    print("=" * 60)
    print("Phase 0: 正解ラベルの3Dマスク化")
    print("=" * 60)
    print(f"開始: 全 {total_files} 件の処理")
    print("-" * 60)

    # 3. 統計用カウンター
    success_count = 0
    skip_count = 0
    error_count = 0
    skipped_files = []
    error_files = []

    # 4. ループ処理
    for idx, row in df.iterrows():
        sample_id = str(row['ID'])

        # 進捗表示
        progress_pct = (idx + 1) / total_files * 100
        print(f"[{idx+1}/{total_files} ({progress_pct:.1f}%)] Processing: {sample_id} ...")

        # Exclude処理
        if row['Exclude']:
            print(f"  [Skip] Excluded in CSV.")
            skip_count += 1
            skipped_files.append((sample_id, "Excluded"))
            continue

        # パス設定
        input_path = os.path.join(NIFTI_DIR, f"{sample_id}.nii.gz")
        output_path = os.path.join(OUTPUT_DIR, f"{sample_id}.nii.gz")

        # 入力ファイル存在確認
        if not os.path.exists(input_path):
            print(f"  [Skip] Input file not found: {input_path}")
            skip_count += 1
            skipped_files.append((sample_id, "Input file not found"))
            continue

        # 既に処理済みならスキップ
        if os.path.exists(output_path):
            print(f"  [Skip] Already processed.")
            skip_count += 1
            continue

        try:
            # Z値のパース
            z_starts = parse_z_value(row['Z_Start'])
            z_ends = parse_z_value(row['Z_End'])

            # NaN/欠損値チェック
            if z_starts is None or z_ends is None:
                print(f"  [Skip] Z_Start or Z_End is NaN.")
                skip_count += 1
                skipped_files.append((sample_id, "Missing Z range"))
                continue

            # 区間数の一致チェック
            if len(z_starts) != len(z_ends):
                print(f"  [Error] Z_Start と Z_End の区間数が一致しません: {len(z_starts)} vs {len(z_ends)}")
                error_count += 1
                error_files.append((sample_id, f"区間数の不一致: {len(z_starts)} vs {len(z_ends)}"))
                continue

            # 区間リストの作成
            ranges = list(zip(z_starts, z_ends))

            # NIfTI読み込み
            img = nib.load(input_path)
            shape = img.shape

            # ゼロマスク作成
            mask = np.zeros(shape, dtype=np.uint8)

            # 複数区間の処理
            all_ranges_info = []
            for z_start, z_end in ranges:
                # 範囲チェック
                original_start = z_start
                original_end = z_end

                if z_start < 0 or z_end >= shape[2]:
                    # 範囲外の場合はクリップして警告
                    z_start = max(0, z_start)
                    z_end = min(shape[2] - 1, z_end)
                    print(f"  ⚠️  Warning: Z range clipped from [{original_start}, {original_end}] to [{z_start}, {z_end}] (image Z size: {shape[2]})")

                # マスク設定
                mask[:, :, z_start:z_end+1] = 1
                all_ranges_info.append(f"[{z_start}, {z_end}]")

            # NIfTI保存 (affineとheaderは元画像から継承)
            mask_img = nib.Nifti1Image(mask, img.affine, img.header)
            nib.save(mask_img, output_path)

            # 統計情報
            num_voxels = np.sum(mask)
            ranges_str = ", ".join(all_ranges_info)
            print(f"  -> Done! Shape: {shape}, Z ranges: {ranges_str}, Voxels: {num_voxels}")
            success_count += 1

        except Exception as e:
            print(f"  [Error] Failed: {e}")
            error_count += 1
            error_files.append((sample_id, str(e)))

    # 5. サマリー表示
    print("-" * 60)
    print("全処理完了しました。")
    print(f"  成功: {success_count} 件")
    print(f"  スキップ: {skip_count} 件")
    print(f"  エラー: {error_count} 件")

    if skipped_files:
        print("\nスキップしたファイル:")
        for sample_id, reason in skipped_files[:10]:  # 最初の10件のみ表示
            print(f"  - {sample_id}: {reason}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    if error_files:
        print("\nエラーが発生したファイル:")
        for sample_id, error in error_files:
            print(f"  - {sample_id}: {error}")

    print("=" * 60)

if __name__ == "__main__":
    main()
