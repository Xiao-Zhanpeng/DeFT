"""Run DeFT on three locked evaluation cases and display GT / Noisy / Denoised / Residual.

These are the same slices used in the paper's Fig.4 (Q1 Mayo CT),
Fig.5 (Q2 fastMRI Knee), and Fig.6 (Q3 Chest X-ray) qualitative figures.

Visual conventions mirror the paper:
- Q1 CT: windowed to [-160, 240] HU (vmin=0.6, vmax=0.886 in normalized space)
- Q2/Q3: normalized [0, 1] display; Q3 has a 12 px display crop
- Diff: signed prediction−GT, RdBu_r colormap, fixed range [-0.10, +0.10]

Usage (from Colab notebook — %run shares the IPython backend):
    %run run_demo.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm.auto import tqdm

from deft import DeFT, DeFTBackbone


SAMPLES = [
    {
        "label": "Q1 Mayo CT ($\\sigma$=0.10)",
        "gt":    "examples/q1_gt.npy",
        "noisy": "examples/q1_noisy.npy",
        "vmin": 0.600, "vmax": 0.886,
        "crop":  None,
    },
    {
        "label": "Q2 fastMRI Knee ($\\sigma$=0.07)",
        "gt":    "examples/q2_gt.npy",
        "noisy": "examples/q2_noisy.npy",
        "vmin": 0.0, "vmax": 1.0,
        "crop":  None,
    },
    {
        "label": "Q3 Chest X-ray ($\\sigma$=0.10)",
        "gt":    "examples/q3_gt.npy",
        "noisy": "examples/q3_noisy.npy",
        "vmin": 0.0, "vmax": 1.0,
        "crop":  (0, -12, 12, 0),  # top, bottom, left, right (Fig.6 annotation)
    },
]

CHECKPOINT = "checkpoints/unet_source_checkpoint.pt"
OUTPUT_PNG = "deft_demo_output.png"
DIFF_RANGE = 0.10
DIFF_CMAP = "RdBu_r"


def apply_crop(img, crop):
    if crop is None:
        return img
    top, bottom, left, right = crop
    bottom = bottom if bottom else None
    right = right if right else None
    return img[top:bottom, left:right]


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
    fig, axes = plt.subplots(N, 4, figsize=(15, 10))

    col_labels = ["GT", "Noisy", "DeFT Denoised", "Signed Diff (DeFT − GT)"]

    for i in range(N):
        s = SAMPLES[i]
        gt = np.load(s["gt"])
        noisy = np.load(s["noisy"])

        noisy_t = torch.from_numpy(noisy.copy()).float().to(device)
        if noisy_t.ndim == 2:
            noisy_t = noisy_t.unsqueeze(0).unsqueeze(0)
        elif noisy_t.ndim == 3:
            noisy_t = noisy_t.unsqueeze(0)

        print(f"[run_demo] Adapting {s['label']} ...")
        pbar = tqdm(total=10, desc=f"  {s['label'].split('(')[0].strip()}", unit="step")
        def _tick(step, bar=pbar):
            if step == 0 and bar.total != model._actual_steps:
                bar.reset(total=model._actual_steps)
            bar.update(1)
        denoised_t = model.adapt(noisy_t, steps=5, callback=_tick)
        pbar.close()

        denoised = denoised_t.squeeze().cpu().numpy().astype(np.float64)
        gt64 = gt.astype(np.float64)

        signed_diff = denoised - gt64

        vmin, vmax = s["vmin"], s["vmax"]

        gt_disp = apply_crop(gt, s["crop"])
        noisy_disp = apply_crop(noisy, s["crop"])
        denoised_disp = apply_crop(denoised, s["crop"])
        diff_disp = apply_crop(signed_diff, s["crop"])

        # Column 1 — GT
        ax = axes[i][0]
        ax.imshow(gt_disp, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        if i == 0:
            ax.set_title(col_labels[0], fontsize=11, pad=3)
        ax.axis("off")

        # Column 2 — Noisy
        ax = axes[i][1]
        ax.imshow(noisy_disp, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        if i == 0:
            ax.set_title(col_labels[1], fontsize=11, pad=3)
        ax.axis("off")

        # Column 3 — DeFT Denoised
        ax = axes[i][2]
        ax.imshow(denoised_disp, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        if i == 0:
            ax.set_title(col_labels[2], fontsize=11, pad=3)
        ax.axis("off")

        # Column 4 — Signed diff
        ax = axes[i][3]
        im = ax.imshow(diff_disp, cmap=DIFF_CMAP, vmin=-DIFF_RANGE, vmax=DIFF_RANGE,
                       aspect="equal", interpolation="nearest")
        if i == 0:
            ax.set_title(col_labels[3], fontsize=11, pad=3)
        ax.axis("off")

        axes[i][0].set_ylabel(s["label"], fontsize=12, rotation=90,
                              labelpad=10, va="center")

    cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.78])
    fig.colorbar(im, cax=cbar_ax, label="")

    plt.suptitle("DeFT: Source-Free Single-Image Test-Time Adaptation",
                 fontsize=14, y=0.99)
    plt.subplots_adjust(left=0.08, right=0.91, top=0.94, bottom=0.04,
                        wspace=0.05, hspace=0.06)

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[run_demo] Output saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
