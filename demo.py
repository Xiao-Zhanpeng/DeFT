#!/usr/bin/env python3
"""DeFT single-image inference demo.

Usage:
    # Q1 Mayo abdomen CT
    python demo.py --input examples/q1_mayo_ct_noisy.npy --checkpoint checkpoints/unet_source_checkpoint.pt --output denoised.npy

    # Q2 fastMRI knee MRI
    python demo.py --input examples/q2_fastmri_knee_noisy.npy --checkpoint checkpoints/unet_source_checkpoint.pt --output denoised.npy

    # Q3 Chest X-ray
    python demo.py --input examples/q3_chest_xray_noisy.npy --checkpoint checkpoints/unet_source_checkpoint.pt --output denoised.npy
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deft import DeFT, DeFTBackbone


def main():
    parser = argparse.ArgumentParser(
        description="DeFT: Single-image test-time denoising adaptation."
    )
    parser.add_argument("--input", required=True,
                        help="Input noisy image (.npy, shape (H,W) or (1,H,W), float32/float16).")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the pretrained backbone checkpoint (.pt). "
                             "Defaults to checkpoints/unet_source_checkpoint.pt.")
    parser.add_argument("--output", default="denoised.npy",
                        help="Output path for the denoised image (.npy).")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda, cpu, or mps (default: cuda).")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt = args.checkpoint or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints", "unet_source_checkpoint.pt"
    )
    if not os.path.isfile(ckpt):
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Download it with:")
        print("  gdown --id 372932474 -O checkpoints/unet_source_checkpoint.pt")
        sys.exit(1)

    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Load input image
    img = np.load(args.input).astype(np.float32)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3 and img.shape[0] != 1:
        img = img[0:1, ...]  # take first channel if multi-slice

    noisy = torch.from_numpy(img).float()
    noisy = noisy.unsqueeze(0) if noisy.ndim == 3 else noisy  # (1, C, H, W)

    # Build model (use the DeFTBackbone for a clean standalone architecture)
    print(f"Loading backbone from {ckpt} ...")
    backbone = DeFTBackbone(in_channels=1, out_channels=64)
    model = DeFT(denoiser=backbone)
    model.to(args.device)
    noisy = noisy.to(args.device)

    print(f"Adapting to {args.input} (shape {list(noisy.shape)}, device={args.device}) ...")
    with torch.no_grad():
        denoised = model.adapt(noisy, steps=5)

    # Save output
    out = denoised.squeeze().cpu().numpy()
    np.save(args.output, out)
    print(f"Done. Denoised image saved to {args.output}")


if __name__ == "__main__":
    main()
