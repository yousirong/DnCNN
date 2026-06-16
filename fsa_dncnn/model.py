"""PyTorch DnCNN model for grayscale residual restoration."""

from __future__ import annotations

import torch
from torch import nn


class DnCNN(nn.Module):
    """DnCNN residual predictor.

    The network predicts residual = input - target. The restored image is
    produced by ``restore(input) = clip(input - residual, 0, 1)``.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        depth: int = 17,
        features: int = 64,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("DnCNN depth must be at least 3")
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    features,
                    features,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(features, out_ch, kernel_size=3, padding=1, bias=True)
        )
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.forward(x)
        return torch.clamp(x - residual, 0.0, 1.0)
