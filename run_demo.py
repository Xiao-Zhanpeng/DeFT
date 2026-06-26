"""Run DeFT on three locked evaluation cases and display GT / Noisy / Denoised / Residual.

These are the same slices used in the paper's Fig.4 (Q1 Mayo CT),
Fig.5 (Q2 fastMRI Knee), and Fig.6 (Q3 Chest X-ray) qualitative figures.

Usage (from Colab notebook — %run shares the IPython backend):
    %run run_demo.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from deft import DeFT, DeFTBackbone


# Locked paper cases with known ground truth (Fig.4/5/6)
SAMPLES = [
    {
        "label": "Q1 Mayo CT ($\\sigma$=0.10)",
        "gt":    "examples/q1_gt.npy",
        "noisy": "examples/q1_noisy.npy",
        "vmin": 0, "vmax": 1,
    },
    {
        "label": "Q2 fastMRI Knee ($\\sigma$=0.07)",
        "gt":    "examples/q2_gt.npy",
        "noisy": "examples/q2_noisy.npy",
        "vmin": 0, "vmax": 1,
    },
    {
        "label": "Q3 Chest X-ray ($\\sigma$=0.10)",
        "gt":    "examples/q3_gt.npy",
        "noisy": "examples/q3_noisy.npy",
        "vmin": 0, "vmax": 1,
    },
]

CHECKPOINT = "checkpoints/unet_source_checkpoint.pt"
OUTPUT_PNG = "deft_demo_output.png"


def main():
    ckpt = Path(CHECKPOINT)
    if not ckpt.exists():
        print(f"[run_demo] ERROR: checkpoint not found: {ckpt}")
        print("[run_demo] Make sure you ran the download cell first.")
        sys.exit(1)

    for s in SAMPLES:
        for key in ("gt", "noisy"):
            if not Path(s[key]).exists():
                print(f"[run_demo] ERROR: {s[key]} not found.")
                sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_demo] Device: {device}")

    print(f"[run_demo] Loading backbone from {ckpt} ...")
    backbone = DeFTBackbone.from_pretrained(str(ckpt))
    model = DeFT(denoiser=backbone)
    model.to(device)

    N = len(SAMPLES)
    fig, axes = plt.subplots(N, 4, figsize=(16, 10))

    col_titles = ["GT", "Noisy", "DeFT Denoised", "Residual |GT − Denoised|"]

    for i in range(N):
        s = SAMPLES[i]
        gt = np.load(s["gt"])
        noisy = np.load(s["noisy"])

        noisy_t = torch.from_numpy(noisy.copy()).float().to(device)
        if noisy_t.ndim == 2:
            noisy_t = noisy_t.unsqueeze(0).unsqueeze(0)
        elif noisy_t.ndim == 3:
            noisy_t = noisy_t.unsqueeze(0)

        print(f"[run_demo] Adapting {s['label']} (5 steps) ...", end=" ", flush=True)
        denoised_t = model.adapt(noisy_t, steps=5)
        print("done")

        denoised = denoised_t.squeeze().cpu().numpy()

        # Q1 CT: paper uses [-160,240] HU window, here normalized to [0,1]
        vmin, vmax = s["vmin"], s["vmax"]

        # Residual (absolute error)
        residual = np.abs(gt.astype(np.float32) - denoised.astype(np.float32))

        # --- Column 1: GT ---
        ax = axes[i][0]
        ax.imshow(gt, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("GT" if i == 0 else "", fontsize=11)
        ax.axis("off")

        # --- Column 2: Noisy ---
        ax = axes[i][1]
        ax.imshow(noisy, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("Noisy" if i == 0 else "", fontsize=11)
        ax.axis("off")

        # --- Column 3: DeFT Denoised ---
        ax = axes[i][2]
        ax.imshow(denoised, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title("DeFT Denoised" if i == 0 else "", fontsize=11)
        ax.axis("off")

        # --- Column 4: Residual difference map ---
        ax = axes[i][3]
        im = ax.imshow(residual, cmap="inferno", vmin=0, vmax=0.15)
        ax.set_title("Residual |GT − Denoised|" if i == 0 else "", fontsize=11)
        ax.axis("off")

        # Domain label on the left
        axes[i][0].set_ylabel(s["label"], fontsize=12, rotation=90,
                              labelpad=10, va="center")

    # Colorbar for residual column
    cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.78])
    fig.colorbar(im, cax=cbar_ax, label="Absolute error")

    plt.suptitle("DeFT: Source-Free Single-Image Test-Time Adaptation",
                 fontsize=14, y=0.99)
    plt.subplots_adjust(left=0.08, right=0.91, top=0.92, bottom=0.06,
                        wspace=0.05, hspace=0.15)

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[run_demo] Output saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
