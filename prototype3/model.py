import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class BackboneFeatureExtractor(nn.Module):
    """
    ResNet18の特徴マップ抽出器（avgpool・fc削除版）

    入力: (B, 3, 128, 128)
    出力: (B, 512, 4, 4) - 特徴マップ
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # ImageNet事前学習済みResNet18をロード
        resnet = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # 最終2層（avgpool, fc）を削除 → 特徴マップを保持
        # resnet.children(): [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # [:-2]でavgpoolとfcを除外
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        """
        Args:
            x: (B, 3, 128, 128) - 入力画像
        Returns:
            features: (B, 512, 4, 4) - 特徴マップ
        """
        return self.features(x)


class MILHead(nn.Module):
    """
    MILヘッド: パッチ化 + Instance分類 + マスク処理

    特徴マップをパッチ化し、各パッチの骨折確率を予測。
    同時にマスク情報から骨/背景を判定。
    """
    def __init__(self, in_channels=512, mask_threshold=0.1):
        """
        Args:
            in_channels: 入力特徴マップのチャンネル数（ResNet18の場合512）
            mask_threshold: 骨パッチ判定の閾値（デフォルト: 0.1 = 10%）
        """
        super().__init__()
        self.mask_threshold = mask_threshold

        # Instance分類器: 各パッチを512次元 → 1次元（骨折確率）
        self.instance_classifier = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

        # マスクを4×4にダウンサンプリングするためのプーリング層
        self.mask_pooling = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, features, mask):
        """
        Args:
            features: (B, 512, 4, 4) - バックボーンの特徴マップ
            mask: (B, 1, 128, 128) - 椎体マスク（Ch2）

        Returns:
            patch_probs: (B, 16) - 各パッチの骨折確率 [0, 1]
            bone_mask: (B, 16) - 各パッチが骨か背景か（Boolean）
        """
        B = features.shape[0]

        # --- パッチ化: (B, 512, 4, 4) → (B, 16, 512) ---
        # (B, 512, 4, 4) → (B, 4, 4, 512) に並び替え
        patches = features.permute(0, 2, 3, 1)  # (B, 4, 4, 512)
        # 空間次元をフラット化: (B, 4, 4, 512) → (B, 16, 512)
        patches = patches.reshape(B, 16, 512)

        # --- Instance分類 ---
        # 各パッチについて骨折確率を予測: (B, 16, 512) → (B, 16, 1) → (B, 16)
        patch_probs = self.instance_classifier(patches).squeeze(-1)

        # --- マスク処理: 4×4に縮小 → 閾値処理 ---
        # (B, 1, 128, 128) → (B, 1, 4, 4)
        mask_pooled = self.mask_pooling(mask)
        # (B, 1, 4, 4) → (B, 16)
        mask_flat = mask_pooled.view(B, 16)
        # 閾値処理: パッチの10%以上が椎体なら「骨」と判定
        bone_mask = (mask_flat > self.mask_threshold)  # (B, 16) Boolean

        return patch_probs, bone_mask


class FractureMILModel(nn.Module):
    """
    統合MILモデル: バックボーン + MILヘッド

    入力画像から骨折部位を検出し、ヒートマップを生成可能。
    """
    def __init__(self, mask_threshold=0.1, pretrained=True):
        """
        Args:
            mask_threshold: 骨パッチ判定の閾値
            pretrained: ImageNet事前学習済み重みを使用するか
        """
        super().__init__()
        self.backbone = BackboneFeatureExtractor(pretrained)
        self.mil_head = MILHead(in_channels=512, mask_threshold=mask_threshold)

    def forward(self, x):
        """
        学習時のForwardパス

        Args:
            x: (B, 3, 128, 128) - 入力画像（Ch0: Bone, Ch1: Soft, Ch2: Mask）

        Returns:
            patch_probs: (B, 16) - 各パッチの骨折確率
            bone_mask: (B, 16) - 各パッチが骨か背景か
        """
        # マスク（Ch2）を抽出
        mask = x[:, 2:3, :, :]  # (B, 1, 128, 128)

        # 特徴抽出
        features = self.backbone(x)  # (B, 512, 4, 4)

        # MILヘッド
        patch_probs, bone_mask = self.mil_head(features, mask)

        return patch_probs, bone_mask

    def predict_with_heatmap(self, x):
        """
        推論時: スライス確率 + ヒートマップ生成

        Args:
            x: (B, 3, 128, 128) - 入力画像

        Returns:
            slice_probs: (B,) - スライスレベルの骨折確率（骨パッチの最大値）
            heatmap: (B, 128, 128) - ヒートマップ（4×4を128×128にUpsampling）
        """
        # Forwardパス
        patch_probs, bone_mask = self.forward(x)

        # --- スライス確率: 骨パッチの最大値 ---
        slice_probs = []
        for b in range(x.shape[0]):
            bone_idx = bone_mask[b]
            if bone_idx.sum() > 0:
                # 骨パッチが存在する場合: 最大値を取得
                slice_probs.append(patch_probs[b, bone_idx].max())
            else:
                # 全パッチが背景の場合: 0.0（正常）
                slice_probs.append(torch.tensor(0.0, device=x.device))
        slice_probs = torch.stack(slice_probs)

        # --- ヒートマップ: 4×4 → 128×128にBilinear Upsampling ---
        # (B, 16) → (B, 1, 4, 4)
        heatmap_4x4 = patch_probs.view(-1, 1, 4, 4)
        # Bilinear補間で128×128に拡大
        heatmap = F.interpolate(
            heatmap_4x4,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # (B, 128, 128)

        return slice_probs, heatmap


if __name__ == "__main__":
    # --- 単体テスト ---
    print("=== FractureMILModel 単体テスト ===")

    # ダミー入力作成
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 128, 128)
    # Ch2をマスクとして設定（0.5以上がランダムに配置）
    dummy_input[:, 2, :, :] = (torch.rand(batch_size, 128, 128) > 0.5).float()

    # モデル作成
    model = FractureMILModel(mask_threshold=0.1, pretrained=False)
    model.eval()

    # Forward
    with torch.no_grad():
        patch_probs, bone_mask = model(dummy_input)
        print(f"✓ patch_probs shape: {patch_probs.shape} (期待: {(batch_size, 16)})")
        print(f"✓ bone_mask shape: {bone_mask.shape} (期待: {(batch_size, 16)})")
        print(f"✓ patch_probs range: [{patch_probs.min():.3f}, {patch_probs.max():.3f}] (期待: [0, 1])")
        print(f"✓ 骨パッチ数（サンプルごと）: {bone_mask.sum(dim=1).tolist()}")

        # ヒートマップ生成
        slice_probs, heatmap = model.predict_with_heatmap(dummy_input)
        print(f"✓ slice_probs shape: {slice_probs.shape} (期待: {(batch_size,)})")
        print(f"✓ heatmap shape: {heatmap.shape} (期待: {(batch_size, 128, 128)})")
        print(f"✓ slice_probs values: {slice_probs.tolist()}")

    print("\n=== テスト完了 ===")
