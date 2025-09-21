# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-09-20
# Purpose: FSRCNN model definition (Accelerating the Super-Resolution CNN)

from __future__ import annotations
import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """
    FSRCNN model.

    Paper: "Accelerating the Super-Resolution Convolutional Neural Network"
    Authors: Chao Dong, Chen Change Loy, Xiaoou Tang

    Args:
        scale: Upscale factor (2, 3, 4).
        d: Number of feature maps for the feature extraction/expanding layers.
        s: Number of feature maps for shrinking/mapping layers.
        m: Number of mapping layers (3x3 convs) operating on s channels.
        in_ch: Input channels (default 1 for Y channel).
        out_ch: Output channels (default 1 for Y channel).
    """

    # CNN layers
    feature: nn.Sequential
    """Feature extraction layer (5x5 conv)."""

    shrink: nn.Sequential
    """Shrinking layer (1x1 conv)."""

    map: nn.Sequential
    """Mapping layers (3x3 convs)."""

    expand: nn.Sequential
    """Expanding layer (1x1 conv)."""

    deconv: nn.ConvTranspose2d
    """Deconvolution (transpose conv) for upsampling."""

    def __init__(
        self,
        scale: int = 4,
        d: int = 56,
        s: int = 12,
        m: int = 4,
        in_ch: int = 1,
        out_ch: int = 1,
    ) -> None:
        """
        Initialize the FSRCNN model.
        Args:
            scale: Upscale factor (2, 3, 4). Default is 4.
            d: Number of feature maps for the feature extraction/expanding layers. Default is 56.
            s: Number of feature maps for shrinking/mapping layers. Default is 12.
            m: Number of mapping layers (3x3 convs) operating on s channels. Default is 4.
            in_ch: Input channels (default 1 for Y channel).
            out_ch: Output channels (default 1 for Y channel).
        """
        super().__init__()
        self.scale = scale

        # Feature extraction (5x5)
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=5, stride=1, padding=2),
            nn.PReLU(num_parameters=d),
        )

        # Shrinking (1x1)
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=s),
        )

        # Mapping (m x 3x3)
        mapping_layers = []
        for _ in range(m):
            mapping_layers += [
                nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1),
                nn.PReLU(num_parameters=s),
            ]
        self.map = nn.Sequential(*mapping_layers)

        # Expanding (1x1)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0),
            nn.PReLU(num_parameters=d),
        )

        # Deconvolution (transpose conv) for upsampling (9x9, stride=scale)
        self.deconv = nn.ConvTranspose2d(
            d, out_ch, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

        self._initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: LR->HR in a single shot."""
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x

    def _initialize(self) -> None:
        """
        Weight initialization following FSRCNN heuristics.
        - Conv layers: Kaiming normal
        - Deconv: init to approximate bilinear upsampling for stability
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.25, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                # Initialize deconv as bilinear upsampler
                self._init_deconv_bilinear(m)

    @staticmethod
    def _init_deconv_bilinear(deconv: nn.ConvTranspose2d) -> None:
        """Initialize transposed conv to behave like bilinear upsampling."""
        f = deconv.kernel_size[0]
        factor = (f + 1) // 2

        if f % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = torch.arange(f, dtype=torch.float32)
        filt = 1 - torch.abs(og - center) / factor
        weight = filt[:, None] * filt[None, :]
        w = torch.zeros_like(deconv.weight.data)

        for c in range(deconv.out_channels):
            w[c, 0, :, :] = weight

        deconv.weight.data.copy_(w)

        if deconv.bias is not None:
            nn.init.zeros_(deconv.bias)
