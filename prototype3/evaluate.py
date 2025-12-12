import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応

from model import FractureMILModel


def visualize_heatmaps(model, dataset, device, save_dir, num_samples=10, seed=42):
    """
    ヒートマップを可視化して保存

    Args:
        model: FractureMILModel
        dataset: SpineSliceDataset
        device: デバイス
        save_dir: 保存ディレクトリ
        num_samples: 可視化サンプル数
        seed: ランダムシード
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # ランダムにサンプル選択
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"ヒートマップ可視化中... ({num_samples}サンプル)")

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)  # (1, 3, 128, 128)

        with torch.no_grad():
            slice_prob, heatmap = model.predict_with_heatmap(img_batch)

        # NumPy変換
        bone_window = img[0].cpu().numpy()  # Ch0: Bone Window
        soft_window = img[1].cpu().numpy()  # Ch1: Soft Tissue Window
        mask = img[2].cpu().numpy()         # Ch2: Mask
        heatmap_np = heatmap[0].cpu().numpy()  # (128, 128)
        prob = slice_prob.item()

        # --- 可視化 ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Bone Window
        axes[0].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"Bone Window\n(Label: {'Fracture' if label == 1 else 'Normal'})", fontsize=12)
        axes[0].axis('off')

        # 2. Soft Tissue Window
        axes[1].imshow(soft_window, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Soft Tissue Window", fontsize=12)
        axes[1].axis('off')

        # 3. Mask
        axes[2].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title("Vertebra Mask", fontsize=12)
        axes[2].axis('off')

        # 4. Heatmap Overlay
        axes[3].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        im = axes[3].imshow(heatmap_np, alpha=0.6, cmap='jet', vmin=0, vmax=1)
        axes[3].set_title(f"Heatmap\n(Prob: {prob:.3f})", fontsize=12)
        axes[3].axis('off')

        # カラーバー
        cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label('Fracture Probability', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"heatmap_{i:03d}_idx{idx}_label{int(label)}_prob{prob:.3f}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ ヒートマップ保存完了: {save_dir}")


def visualize_predictions_by_label(model, dataset, device, save_dir, num_per_class=5, seed=42):
    """
    正常・骨折それぞれからサンプルを選んでヒートマップ可視化

    医療AIの定性評価に有用。
    True Positive: 骨折を正しく検出 → ヒートマップが骨折部位を指しているか
    False Positive: 正常を骨折と誤検出 → ヒートマップがどこを指しているか
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # ラベルごとにインデックス取得
    normal_indices = []
    fracture_indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label == 0:
            normal_indices.append(idx)
        else:
            fracture_indices.append(idx)

    # ランダムサンプリング
    np.random.seed(seed)
    normal_samples = np.random.choice(normal_indices, min(num_per_class, len(normal_indices)), replace=False)
    fracture_samples = np.random.choice(fracture_indices, min(num_per_class, len(fracture_indices)), replace=False)

    print(f"\nラベル別ヒートマップ可視化:")
    print(f"  正常サンプル: {len(normal_samples)}個")
    print(f"  骨折サンプル: {len(fracture_samples)}個")

    # 正常サンプル可視化
    for i, idx in enumerate(normal_samples):
        img, label = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            slice_prob, heatmap = model.predict_with_heatmap(img_batch)

        bone_window = img[0].cpu().numpy()
        heatmap_np = heatmap[0].cpu().numpy()
        prob = slice_prob.item()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Bone Window", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        im = axes[1].imshow(heatmap_np, alpha=0.6, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Heatmap (Prob: {prob:.3f})", fontsize=12)
        axes[1].axis('off')

        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.suptitle(f"Normal Sample #{i+1} (idx={idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"normal_{i:02d}_idx{idx}_prob{prob:.3f}.png"), dpi=150)
        plt.close()

    # 骨折サンプル可視化
    for i, idx in enumerate(fracture_samples):
        img, label = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            slice_prob, heatmap = model.predict_with_heatmap(img_batch)

        bone_window = img[0].cpu().numpy()
        heatmap_np = heatmap[0].cpu().numpy()
        prob = slice_prob.item()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Bone Window", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(bone_window, cmap='gray', vmin=0, vmax=1)
        im = axes[1].imshow(heatmap_np, alpha=0.6, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Heatmap (Prob: {prob:.3f})", fontsize=12)
        axes[1].axis('off')

        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.suptitle(f"Fracture Sample #{i+1} (idx={idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fracture_{i:02d}_idx{idx}_prob{prob:.3f}.png"), dpi=150)
        plt.close()

    print(f"✓ ラベル別ヒートマップ保存完了: {save_dir}")


if __name__ == "__main__":
    """
    使用例:
    python evaluate.py --model_path best_model_fold0.pth --data_dir ../spine_data --save_dir ./heatmaps
    """
    import argparse
    import pandas as pd
    from dataset import SpineSliceDataset, get_val_transforms
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="ヒートマップ生成・可視化スクリプト")
    parser.add_argument('--model_path', type=str, required=True, help='学習済みモデルのパス')
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリ')
    parser.add_argument('--annotation_csv', type=str, default='../spine_data/slice_annotations.csv', help='アノテーションCSV')
    parser.add_argument('--save_dir', type=str, default='./heatmaps', help='保存先ディレクトリ')
    parser.add_argument('--num_samples', type=int, default=20, help='可視化サンプル数')
    parser.add_argument('--device', type=str, default='cuda:1', help='デバイス')
    args = parser.parse_args()

    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # モデル読み込み
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['config']['model']

    model = FractureMILModel(
        mask_threshold=model_config['mask_threshold'],
        pretrained=False  # 学習済み重みを読み込むのでFalse
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ モデル読み込み完了: {args.model_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Best PR-AUC: {checkpoint['best_pr_auc']:.4f}")

    # データセット準備（検証データ全体を使用）
    df = pd.read_csv(args.annotation_csv)
    # 設定ファイルから変換パイプライン作成（簡易版）
    cfg_simple = OmegaConf.create({
        'augmentation': {
            'horizontal_flip_p': 0.0,
            'rotate_limit': 0,
            'rotate_p': 0.0,
            'brightness_contrast_p': 0.0
        }
    })
    dataset = SpineSliceDataset(
        df.head(100),  # 最初の100サンプル（全体を使う場合は df をそのまま渡す）
        args.data_dir,
        transform=get_val_transforms(cfg_simple)
    )
    print(f"✓ データセット準備完了: {len(dataset)}サンプル\n")

    # ヒートマップ可視化
    visualize_heatmaps(model, dataset, device, args.save_dir, num_samples=args.num_samples)

    # ラベル別可視化
    visualize_predictions_by_label(model, dataset, device, args.save_dir, num_per_class=5)

    print("\n=== 評価完了 ===")
