import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MILResNet(nn.Module):
    """
    MIL用 ResNet Encoder
    時系列(LSTM)としては扱わず、バッチ内の各スライスを独立して評価する。
    """

    def __init__(
        self,
        encoder_name='resnet18',
        dropout=0.5
    ):
        super().__init__()

        # 1. 2D CNN Encoder
        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # 最終層(fc)を置き換え
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1) # 各スライスに対して1つのスコア(logit)を出力
            )
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, T) - 各スライスの異常度スコア(logit)
        """
        B, T, C, H, W = x.shape

        # バッチと時間をマージしてCNNに入力 (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # (B*T, 1)
        out = self.encoder(x)
        
        # 形を戻す (B, T)
        logits = out.view(B, T)

        return logits