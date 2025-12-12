import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def fracture_mil_loss(
    patch_probs: torch.Tensor,  # (B, 16)
    bone_mask: torch.Tensor,    # (B, 16) Boolean
    slice_labels: torch.Tensor, # (B,) 0 or 1
    w_bg: float = 1.5,
    w_max: float = 1.0,
    w_mean: float = 0.5,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Fracture MIL Loss（3項構成）

    prototype3の独自Loss関数。マスク情報を活用してパッチレベルで骨折検出を学習。

    3つの項:
    1. L_bg (背景抑制): 背景パッチの予測を0に抑制 → False Positive削減
    2. L_max (最大値): 骨パッチの最大値とスライスラベルのBCE → MIL主タスク
    3. L_mean (平均値): 骨パッチの平均値とスライスラベルのBCE → 学習安定化

    Args:
        patch_probs: (B, 16) - 各パッチの骨折確率 [0, 1]
        bone_mask: (B, 16) - 各パッチが骨か背景か（Boolean）
        slice_labels: (B,) - スライスレベルの骨折ラベル（0 or 1）
        w_bg: 背景抑制Lossの重み（推奨: 1.5）
        w_max: 最大値Lossの重み（推奨: 1.0）
        w_mean: 平均値Lossの重み（推奨: 0.5）
        eps: 数値安定性のための微小値（log(0)回避）

    Returns:
        total_loss: スカラー - 3項の加重和
        components: Dict[str, float] - 各項の値（ロギング用）
    """
    device = patch_probs.device
    B = patch_probs.shape[0]

    # ========================================
    # 項1: 背景抑制 (L_bg)
    # ========================================
    # 背景パッチ（M_i=0）の予測確率を0に近づける
    # → 骨のない場所での誤検知（False Positive）を削減
    bg_mask = ~bone_mask  # 背景パッチのマスク
    if bg_mask.sum() > 0:
        bg_probs = patch_probs[bg_mask]  # 背景パッチの確率のみ抽出
        # BCEで0に近づける（数値安定性のためclamp）
        L_bg = F.binary_cross_entropy(
            bg_probs.clamp(eps, 1 - eps),
            torch.zeros_like(bg_probs),
            reduction='mean'
        )
    else:
        # バッチ内に背景パッチがない場合（稀）
        L_bg = torch.tensor(0.0, device=device)

    # ========================================
    # 項2: 骨領域・最大値 (L_max)
    # ========================================
    # 骨パッチ（M_i=1）の最大値とスライスラベルのBCE
    # → 「どこか一箇所でも骨折があればスライス全体が陽性」というMILの核心
    slice_max_probs = []
    for b in range(B):
        bone_idx = bone_mask[b]
        if bone_idx.sum() > 0:
            # 骨パッチが存在する場合: 最大値を取得
            slice_max_probs.append(patch_probs[b, bone_idx].max())
        else:
            # 全パッチが背景の場合: 0.0（正常扱い）
            slice_max_probs.append(torch.tensor(0.0, device=device))

    slice_max = torch.stack(slice_max_probs)  # (B,)
    L_max = F.binary_cross_entropy(
        slice_max.clamp(eps, 1 - eps),
        slice_labels,
        reduction='mean'
    )

    # ========================================
    # 項3: 骨領域・平均値 (L_mean)
    # ========================================
    # 骨パッチ（M_i=1）の平均値とスライスラベルのBCE
    # → 学習の安定化（正常スライスは全パッチ低確率、骨折スライスは複数パッチ高確率）
    # → 重みは弱め（強すぎると局在化精度が低下）
    slice_mean_probs = []
    for b in range(B):
        bone_idx = bone_mask[b]
        if bone_idx.sum() > 0:
            # 骨パッチが存在する場合: 平均値を計算
            slice_mean_probs.append(patch_probs[b, bone_idx].mean())
        else:
            # 全パッチが背景の場合: 0.0
            slice_mean_probs.append(torch.tensor(0.0, device=device))

    slice_mean = torch.stack(slice_mean_probs)  # (B,)
    L_mean = F.binary_cross_entropy(
        slice_mean.clamp(eps, 1 - eps),
        slice_labels,
        reduction='mean'
    )

    # ========================================
    # 合計Loss
    # ========================================
    total_loss = w_bg * L_bg + w_max * L_max + w_mean * L_mean

    # ロギング用に各項を記録
    components = {
        'L_bg': L_bg.item(),
        'L_max': L_max.item(),
        'L_mean': L_mean.item()
    }

    return total_loss, components


if __name__ == "__main__":
    # --- 単体テスト ---
    print("=== fracture_mil_loss 単体テスト ===\n")

    # テストケース1: 通常ケース
    print("【テストケース1】通常ケース")
    batch_size = 4
    patch_probs = torch.rand(batch_size, 16)  # (B, 16) ランダム確率
    bone_mask = torch.rand(batch_size, 16) > 0.5  # (B, 16) ランダムに骨/背景
    slice_labels = torch.tensor([0.0, 1.0, 0.0, 1.0])  # (B,) 正常・骨折

    loss, components = fracture_mil_loss(patch_probs, bone_mask, slice_labels)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Components: L_bg={components['L_bg']:.4f}, L_max={components['L_max']:.4f}, L_mean={components['L_mean']:.4f}")
    print()

    # テストケース2: エッジケース - 全パッチが背景
    print("【テストケース2】エッジケース - 全パッチが背景")
    bone_mask_all_bg = torch.zeros(batch_size, 16, dtype=torch.bool)
    slice_labels_normal = torch.zeros(batch_size)

    loss, components = fracture_mil_loss(patch_probs, bone_mask_all_bg, slice_labels_normal)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Components: L_bg={components['L_bg']:.4f}, L_max={components['L_max']:.4f}, L_mean={components['L_mean']:.4f}")
    print("（期待: L_bgのみ有効、L_max/L_meanは0に近い）")
    print()

    # テストケース3: エッジケース - 全パッチが骨
    print("【テストケース3】エッジケース - 全パッチが骨")
    bone_mask_all_bone = torch.ones(batch_size, 16, dtype=torch.bool)
    slice_labels_fracture = torch.ones(batch_size)

    loss, components = fracture_mil_loss(patch_probs, bone_mask_all_bone, slice_labels_fracture)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Components: L_bg={components['L_bg']:.4f}, L_max={components['L_max']:.4f}, L_mean={components['L_mean']:.4f}")
    print("（期待: L_bgは0、L_max/L_meanが有効）")
    print()

    # テストケース4: 勾配伝播確認
    print("【テストケース4】勾配伝播確認")
    patch_probs_grad = torch.rand(batch_size, 16, requires_grad=True)
    bone_mask = torch.rand(batch_size, 16) > 0.5
    slice_labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    loss, components = fracture_mil_loss(patch_probs_grad, bone_mask, slice_labels)
    loss.backward()
    print(f"Total Loss: {loss.item():.4f}")
    print(f"patch_probs.grad: min={patch_probs_grad.grad.min():.4f}, max={patch_probs_grad.grad.max():.4f}")
    print("✓ 勾配が正常に計算されています")
    print()

    print("=== テスト完了 ===")
