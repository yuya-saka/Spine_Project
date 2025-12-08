"""
モデル定義: CNN_LSTM_Fracture
ResNet18 Encoder + Bidirectional LSTM
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CNN_LSTM_Fracture(nn.Module):
    """2D CNN (ResNet18) Encoder + Bidirectional LSTM"""

    def __init__(
        self,
        encoder_name='resnet18',
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3
    ):
        super().__init__()

        # 1. 2D CNN Encoder (ResNet18, ImageNet pretrained)
        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Identity()  # FC層削除
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # 2. LSTM Layer
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # 3. Classification Head
        fc_input = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary classification (logits)
        )

    def forward(self, x):
        """
        Forward pass (スライディングウィンドウ方式)

        Args:
            x: (B, T=15, C=3, H=128, W=128) - バッチのウィンドウ
               T: ウィンドウサイズ (15枚)
               中心スライス (T//2=7番目) を予測

        Returns:
            logits: (B,) - 中心スライスのlogits (スカラー)
        """
        B, T, C, H, W = x.shape

        # CNN Encoding (各スライスを独立に処理)
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        features = self.encoder(x)  # (B*T, 512)
        features = features.view(B, T, -1)  # (B, T, 512)

        # LSTM (シーケンス処理)
        lstm_out, _ = self.lstm(features)  # (B, T, lstm_hidden*2)

        # 中心スライス (T//2) の特徴量のみを取得
        center_idx = T // 2  # 7
        center_features = lstm_out[:, center_idx, :]  # (B, lstm_hidden*2)

        # Classification (中心スライスのみ)
        logits = self.fc(center_features).squeeze(-1)  # (B,)

        return logits
