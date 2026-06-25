"""DeFT backbone: 31M U-Net without batch normalization, used as the denoising
backbone in Descriptor-Forked Test-Time Adaptation (DeFT)."""

import torch
import torch.nn as nn
from pathlib import Path


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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Mapping from original DANRFUNet / JointUNetMIM module names to
# DeFTBackbone names.  Checkpoints produced by the source pretraining
# pipeline store weights under these original names.
#
# body_outc (64→64) is the U-Net body's internal output projection;
# head_denoise (64→1) is the final denoising head.  In DeFTBackbone
# these are collapsed into a single output_conv (64→1).
_KEY_MAP = {
    'inc.':           'input_conv.',
    'down1.':         'encoder1.',
    'down2.':         'encoder2.',
    'down3.':         'encoder3.',
    'down4.':         'encoder4.',
    'up1.':           'decoder1.',
    'up2.':           'decoder2.',
    'up3.':           'decoder3.',
    'up4.':           'decoder4.',
    'head_denoise.':  'output_conv.',
}
_SKIP_PREFIXES = ('head_mim', 'unet_body.outc')


class DeFTBackbone(nn.Module):
    """31M U-Net encoder-decoder with skip connections, following the original
    U-Net architecture (Ronneberger et al., MICCAI 2015).  No batch
    normalization — uses ReLU activations throughout.

    Channel progression: (64, 128, 256, 512, 1024).
    """

    def __init__(self, in_channels=1, out_channels=1):
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

    @classmethod
    def from_pretrained(cls, checkpoint_path):
        """Load pretrained weights from a source-domain checkpoint.

        The source pretraining pipeline saves weights under the original
        DANRFUNet / JointUNetMIM module names (``down4``, ``up1``,
        ``outc``, etc.).  This method handles the key mapping transparently
        so callers never need to know the original naming.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint produced by the
                source pretraining pipeline.

        Returns:
            DeFTBackbone instance with pretrained weights loaded.
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu',
                          weights_only=False)
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

        # Strip common wrapper prefixes, then map old → new names.
        mapped = {}
        skipped = 0
        for key, value in state.items():
            k = key.replace('_orig_mod.', '').replace('unet_body.', '').replace('denoiser.', '')
            # Discard auxiliary heads (not used in TTA).
            if any(k.startswith(p) for p in _SKIP_PREFIXES):
                skipped += 1
                continue
            for old_prefix, new_prefix in _KEY_MAP.items():
                if k.startswith(old_prefix):
                    k = new_prefix + k[len(old_prefix):]
                    break
            mapped[k] = value

        model = cls(in_channels=1, out_channels=1)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if skipped:
            print(f"  (skipped {skipped} MIM-head keys)")
        if missing:
            print(f"  Note: {len(missing)} key(s) not found in checkpoint (OK for partial load)")
        if unexpected:
            print(f"  Note: {len(unexpected)} unexpected key(s) (ignored)")
        print(f"  Loaded {len(mapped) - skipped} of {len(state)} keys")
        return model

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
