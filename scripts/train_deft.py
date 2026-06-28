#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFT source-domain pretraining script.

Trains the U-Net backbone (DeFTBackbone) on the source domain P using
supervised Charbonnier loss on noisy-clean pairs.

Usage:
    python scripts/train_deft.py \
        --data-root /path/to/P_train_db \
        --output-dir /path/to/checkpoints \
        --epochs 100 --batch-size 16 --lr 2e-4

The trained checkpoint is then used by eval_deft.py for test-time adaptation.

----------------------------------------------------------------------------
NOTE: This script is a reference template. For the full training pipeline
used in the paper, see the project training scripts.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
DEFT_ROOT = Path(os.environ.get("DEFT_ROOT", SCRIPT_DIR.parent))
sys.path.insert(0, str(DEFT_ROOT))

from deft.backbone import DeFTBackbone


# ============================================================
# Data loading
# ============================================================


class PairedNumpyDataset(Dataset):
    """Simple paired dataset from noisy/ and clean/ subdirectories of .npy files.

    Expects:
        data_root/
            noisy/   — *.npy noisy images
            clean/   — *.npy clean images (matched by sorted filename order)
    """

    def __init__(self, data_root: str, limit: Optional[int] = None):
        self.data_root = Path(data_root)
        self.noisy_dir = self.data_root / "noisy"
        self.clean_dir = self.data_root / "clean"

        if not self.noisy_dir.is_dir() or not self.clean_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {data_root}/ with noisy/ and clean/ subdirectories"
            )

        self.noisy_files = sorted(p for p in self.noisy_dir.glob("*.npy"))
        self.clean_files = sorted(p for p in self.clean_dir.glob("*.npy"))

        if len(self.noisy_files) != len(self.clean_files):
            raise RuntimeError(
                f"Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean files"
            )

        if limit is not None:
            self.noisy_files = self.noisy_files[:limit]
            self.clean_files = self.clean_files[:limit]

        # Probe first file for shape info
        sample = np.load(self.noisy_files[0])
        if sample.ndim == 2:
            self._img_shape = (1, *sample.shape)
        else:
            self._img_shape = sample.shape
        self.channels = sample.shape[0] if sample.ndim == 3 else 1

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = np.load(self.noisy_files[idx]).astype(np.float32)
        clean = np.load(self.clean_files[idx]).astype(np.float32)
        if noisy.ndim == 2:
            noisy = noisy[np.newaxis, ...]
        if clean.ndim == 2:
            clean = clean[np.newaxis, ...]
        return torch.from_numpy(noisy), torch.from_numpy(clean)


# ============================================================
# Loss helpers
# ============================================================


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return float(10.0 * torch.log10(data_range ** 2 / mse))


def save_checkpoint(
    state: dict,
    output_dir: Path,
    filename: str = "checkpoint_latest.pth",
    is_best: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(state, path)
    if is_best:
        best_path = output_dir / "checkpoint_best.pth"
        shutil.copyfile(path, best_path)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training
# ============================================================


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_dataset = PairedNumpyDataset(args.data_root, limit=args.max_samples)
    print(f"[Data] {len(train_dataset)} noisy-clean pairs from {args.data_root}")
    print(f"       Image shape: {train_dataset._img_shape}, channels={train_dataset.channels}")

    # Optional validation set
    val_dataset = None
    val_dir = Path(args.data_root).parent / "val"
    if val_dir.exists():
        try:
            val_dataset = PairedNumpyDataset(str(val_dir), limit=args.max_val_samples)
            print(f"[Data] Validation set: {len(val_dataset)} pairs from {val_dir}")
        except FileNotFoundError:
            print("[Data] No validation set found")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

    # --- Model ---
    model = DeFTBackbone(
        in_channels=train_dataset.channels,
        out_channels=train_dataset.channels,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] DeFTBackbone: {total_params / 1e6:.2f}M params "
          f"({trainable_params / 1e6:.2f}M trainable)")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training state ---
    start_epoch = 0
    best_val_psnr = -float("inf")
    epochs_no_improve = 0
    train_status_path = output_dir / "train_status.json"

    if args.resume and (output_dir / "checkpoint_latest.pth").exists():
        ckpt = torch.load(output_dir / "checkpoint_latest.pth", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_psnr = ckpt.get("best_val_psnr", -float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"[Resume] Epoch {start_epoch}, best PSNR {best_val_psnr:.4f}")

    # --- Training loop ---
    print(f"[Train] Starting {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = AverageMeter("Loss")

        t_epoch = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for noisy, clean in pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            output = model(noisy)
            if isinstance(output, tuple):
                output = output[0]

            # Residual prediction: if output mean is small, treat as residual
            if output.abs().mean() < 0.5:
                denoised = torch.clamp(noisy + output, 0, 1)
            else:
                denoised = torch.clamp(output, 0, 1)

            loss = charbonnier_loss(denoised, clean)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_meter.update(loss.item(), noisy.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4e}"})

        scheduler.step()
        epoch_time = time.time() - t_epoch
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {loss_meter.avg:.4e} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_psnr = 0.0
            val_count = 0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy = noisy.to(device, non_blocking=True)
                    clean = clean.to(device, non_blocking=True)
                    output = model(noisy)
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.abs().mean() < 0.5:
                        denoised = torch.clamp(noisy + output, 0, 1)
                    else:
                        denoised = torch.clamp(output, 0, 1)
                    val_psnr += compute_psnr(denoised, clean) * noisy.size(0)
                    val_count += noisy.size(0)

            val_psnr /= max(val_count, 1)
            is_best = val_psnr > best_val_psnr + args.min_delta

            if is_best:
                epochs_no_improve = 0
                best_val_psnr = val_psnr
            else:
                epochs_no_improve += 1

            print(f"   Val PSNR: {val_psnr:.4f} | "
                  f"Best: {best_val_psnr:.4f} | "
                  f"No-improve: {epochs_no_improve}/{args.patience}")
        else:
            is_best = True  # save every epoch when no val set

        # --- Save checkpoint ---
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_psnr": best_val_psnr,
                "epochs_no_improve": epochs_no_improve,
                "args": vars(args),
            },
            output_dir,
            is_best=is_best,
        )

        train_status = {
            "epoch": epoch + 1,
            "best_val_psnr": best_val_psnr,
            "epochs_no_improve": epochs_no_improve,
            "channels": train_dataset.channels,
        }
        with open(train_status_path, "w") as f:
            json.dump(train_status, f, indent=2)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[Early Stop] Patience {args.patience} reached. Best PSNR: {best_val_psnr:.4f}")
            break

    print(f"[Done] Best val PSNR: {best_val_psnr:.4f}. Checkpoints saved to {output_dir}")


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeFT source-domain backbone pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_deft.py --data-root data/P_train_db --output-dir checkpoints/source

  # Customize training
  python scripts/train_deft.py --data-root data/P_train_db --output-dir checkpoints/source \\
      --epochs 100 --batch-size 16 --lr 2e-4

  # Resume training
  python scripts/train_deft.py --data-root data/P_train_db --output-dir checkpoints/source --resume
        """,
    )

    # Data
    parser.add_argument("--data-root", type=str, required=True,
                        help="Training data directory with noisy/ and clean/ subdirectories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples (for quick tests)")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Limit validation samples")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (<=0 to disable)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (<=0 to disable)")
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Minimum validation PSNR improvement to count as best")

    # Hardware
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Resume
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint_latest.pth in output-dir")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 72)
    print("DeFT Source Backbone Pretraining")
    print("=" * 72)
    print()
    print("NOTE: This script is a reference template. For the full training")
    print("pipeline used in the paper, see the project training scripts.")
    print()

    torch.backends.cudnn.benchmark = True

    try:
        train(args)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        sys.exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
