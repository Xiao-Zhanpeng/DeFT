"""DeFT backbone: 31M U-Net without batch normalization, used as the denoising
backbone in Descriptor-Forked Test-Time Adaptation (DeFT)."""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """(Conv3x3 -> ReLU) * 2, no batch normalization."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class _DownEncoder(nn.Module):
    """MaxPool2d -> DoubleConv, encoder down-sampling block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class _UpDecoder(nn.Module):
    """ConvTranspose2d -> Concat(skip) -> DoubleConv, decoder up-sampling block.

    Assumes input spatial dimensions are powers of two (e.g. 512x512), so skip
    connections are naturally aligned with the upsampled feature maps.  Dynamic
    padding is removed for torch.compile compatibility.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # With power-of-two input (e.g. 512x512), spatial dimensions are
        # naturally aligned -- no dynamic padding needed.
        # To support arbitrary sizes, uncomment the following:
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # if diffY != 0 or diffX != 0:
        #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DeFTBackbone(nn.Module):
    """31M U-Net encoder-decoder with skip connections, following the original
    U-Net architecture (Ronneberger et al., MICCAI 2015).  No batch
    normalization — uses ReLU activations throughout.

    Channel progression: (64, 128, 256, 512, 1024).
    """

    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.input_conv = _DoubleConv(in_channels, 64)
        self.encoder1 = _DownEncoder(64, 128)
        self.encoder2 = _DownEncoder(128, 256)
        self.encoder3 = _DownEncoder(256, 512)
        self.encoder4 = _DownEncoder(512, 1024)
        self.decoder1 = _UpDecoder(1024, 512)
        self.decoder2 = _UpDecoder(512, 256)
        self.decoder3 = _UpDecoder(256, 128)
        self.decoder4 = _UpDecoder(128, 64)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, return_bottleneck=False):
        x1 = self.input_conv(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        out = self.output_conv(x)
        if return_bottleneck:
            return out, x5
        return out
