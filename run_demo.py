"""Run DeFT on three example medical images and display results.

Usage:
    python run_demo.py [--checkpoint PATH] [--device cuda|cpu]

This script is designed to be called from the Colab notebook
(demo.ipynb) to avoid notebook-cell indentation caching issues.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deft import DeFT, DeFTBackbone


SAMPLES = [
    ("Q1 Mayo CT",        "examples/q1_mayo_ct_noisy.npy",      5),
    ("Q2 fastMRI Knee",   "examples/q2_fastmri_knee_noisy.npy", 5),
    ("Q3 Chest X-ray",    "examples/q3_chest_xray_noisy.npy",   5),
]


def main():
    parser = argparse.ArgumentParser(description="DeFT demo runner")
    parser.add_argument("--checkpoint", default="checkpoints/unet_source_checkpoint.pt",
                        help="Path to pretrained source-domain checkpoint")
    parser.add_argument("--device", default=None,
                        help="Device to use (cuda or cpu; auto-detect if not set)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_demo] Device: {args.device}")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[run_demo] ERROR: checkpoint not found: {ckpt}")
        print("[run_demo] Make sure you ran the download cell first.")
        sys.exit(1)

    print(f"[run_demo] Loading backbone from {ckpt} ...")
    backbone = DeFTBackbone.from_pretrained(str(ckpt))
    model = DeFT(denoiser=backbone)
    model.to(args.device)

    all_exist = True
    for _, path, _ in SAMPLES:
        if not Path(path).exists():
            print(f"[run_demo] ERROR: sample not found: {path}")
            all_exist = False
    if not all_exist:
        print("[run_demo] Make sure example .npy files are downloaded (they come with the repo).")
        sys.exit(1)

    fig, axes = plt.subplots(len(SAMPLES), 2, figsize=(10, 12))

    for i, (name, path, steps) in enumerate(SAMPLES):
        img = np.load(path).astype(np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        noisy = torch.from_numpy(img).float().to(args.device).unsqueeze(0)

        print(f"[run_demo] Adapting {name} ({steps} steps) ...", end=" ", flush=True)
        denoised = model.adapt(noisy, steps=steps)
        print("done")

        axes[i][0].imshow(noisy.squeeze().cpu().numpy(), cmap="gray")
        axes[i][0].set_title(f"{name} — Noisy", fontsize=12)
        axes[i][0].axis("off")
        axes[i][1].imshow(denoised.squeeze().cpu().numpy(), cmap="gray")
        axes[i][1].set_title(f"{name} — DeFT Denoised", fontsize=12)
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.suptitle("DeFT: Source-Free Single-Image Test-Time Adaptation",
                 fontsize=14, y=1.02)
    plt.savefig("/content/deft_demo_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n[run_demo] All three domains processed. Output saved to deft_demo_output.png")


if __name__ == "__main__":
    main()
