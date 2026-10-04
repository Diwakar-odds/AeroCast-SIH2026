"""
AeroCast Multi-Task Learning (MTL) Branching Heads
1. Severe Thunderstorm Probability Map (0-100%)
2. Cloudburst Inception & QPE Rainfall Rate (mm/hr)
3. Flash Flood Catchment Inundation Risk Map
"""
import torch
import torch.nn as nn
from models.convlstm import AeroCastBackbone

class AeroCastMultiTaskModel(nn.Module):
    def __init__(self, in_channels: int = 10):
        super().__init__()
        self.backbone = AeroCastBackbone(in_channels=in_channels)

        # Head 1: Thunderstorm
        self.head_thunderstorm = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 2: Cloudburst (Classification Prob + Regression Intensity)
        self.head_cloudburst_prob = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_qpe_intensity = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU() # Rain intensity >= 0 mm/hr
        )

        # Head 3: Flash Flood Runoff Risk
        self.head_flash_flood = nn.Sequential(
            nn.Conv2d(64 + 1, 32, kernel_size=3, padding=1), # Conditioned on QPE rate
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        shared_feat = self.backbone(x_seq)
        p_ts = self.head_thunderstorm(shared_feat)
        p_cb = self.head_cloudburst_prob(shared_feat)
        qpe_rate = self.head_qpe_intensity(shared_feat)
        
        # Flash flood conditioned on predicted precipitation
        flood_input = torch.cat([shared_feat, qpe_rate], dim=1)
        p_ff = self.head_flash_flood(flood_input)

        return {
            "thunderstorm_prob": p_ts,
            "cloudburst_prob": p_cb,
            "qpe_rainfall_rate": qpe_rate,
            "flash_flood_prob": p_ff
        }

if __name__ == "__main__":
    model = AeroCastMultiTaskModel(in_channels=10)
    dummy_input = torch.randn(2, 12, 10, 256, 256)
    out = model(dummy_input)
    print("Inference Test:")
    for k, v in out.items():
        print(f" - {k}: shape {v.shape}, range [{v.min().item():.3f}, {v.max().item():.3f}]")
