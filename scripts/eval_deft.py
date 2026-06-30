#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFT evaluation script: per-image test-time adaptation with PSNR reporting.

Loads a source-pretrained backbone, wraps it with DeFT, and adapts each
noisy image individually. If clean ground truth is available, PSNR/SSIM
are computed and saved to a CSV.

Canonical defaults from the paper evaluation pipeline.

Usage:
    # Q1 Mayo abdomen CT (Gaussian sigma=0.10)
    python scripts/eval_deft.py \\
        --dataset Q0_mayo_eval_S10 \\
        --q-dataset-dir /path/to/Q1_mayo_eval \\
        --checkpoint checkpoints/source/checkpoint_best.pth \\
        --output results/Q1_mayo_S10.csv

    # Q2 fastMRI knee MRI (Rician sigma=0.07)
    python scripts/eval_deft.py \\
        --dataset Q3_fastmri_eval_R07 \\
        --q-dataset-dir /path/to/Q2_fastmri_eval \\
        --checkpoint checkpoints/source/checkpoint_best.pth \\
        --output results/Q2_fastmri_R07.csv

    # Q3 Chest X-ray (Gaussian sigma=0.10)
    python scripts/eval_deft.py \\
        --dataset Q2_xray_eval_S10 \\
        --q-dataset-dir /path/to/Q3_xray_eval \\
        --checkpoint checkpoints/source/checkpoint_best.pth \\
        --output results/Q3_xray_S10.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
DEFT_ROOT = Path(os.environ.get("DEFT_ROOT", SCRIPT_DIR.parent))
sys.path.insert(0, str(DEFT_ROOT))

from deft import DeFT, DeFTBackbone


# ============================================================
# Data loading
# ============================================================


class SingleImageEvalDataset(Dataset):
    """Loads a single-image evaluation dataset from noisy/ and clean/ subdirectories.

    Expects:
        dataset_dir/
            noisy/   — *.npy or *.png noisy images
            clean/   — *.npy or *.png clean images (optional, for PSNR)
    """

    def __init__(self, dataset_dir: str):
        self.root = Path(dataset_dir)
        self.noisy_dir = self.root / "noisy"
        self.clean_dir = self.root / "clean"

        if not self.noisy_dir.is_dir():
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_dir}")

        self.noisy_files = sorted(
            list(self.noisy_dir.glob("*.npy")) + list(self.noisy_dir.glob("*.png"))
        )
        if not self.noisy_files:
            raise FileNotFoundError(f"No .npy or .png files in {self.noisy_dir}")

        if self.clean_dir.is_dir():
            self.clean_files = sorted(
                list(self.clean_dir.glob("*.npy")) + list(self.clean_dir.glob("*.png"))
            )
            self.has_clean = len(self.clean_files) > 0
        else:
            self.clean_files = []
            self.has_clean = False

        if self.has_clean and len(self.clean_files) != len(self.noisy_files):
            print(f"[WARN] Mismatched noisy ({len(self.noisy_files)}) vs "
                  f"clean ({len(self.clean_files)}) counts. PSNR may be unreliable.")

        # Probe first file
        sample = self._load_npy_or_png(self.noisy_files[0])
        if sample.ndim == 2:
            self._img_shape = (1, *sample.shape)
        else:
            self._img_shape = sample.shape
        self.channels = sample.shape[0] if sample.ndim == 3 else 1

    @staticmethod
    def _load_npy_or_png(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".png":
            import imageio.v2 as imageio
            img = imageio.imread(str(path))
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            elif img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC -> CHW
            return img.astype(np.float32) / 255.0
        else:
            return np.load(path).astype(np.float32)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = self._load_npy_or_png(self.noisy_files[idx])
        if noisy.ndim == 2:
            noisy = noisy[np.newaxis, ...]
        if noisy.shape[0] > 3:  # multi-channel, take first 3
            noisy = noisy[:3, ...]

        clean = None
        if self.has_clean and idx < len(self.clean_files):
            clean = self._load_npy_or_png(self.clean_files[idx])
            if clean.ndim == 2:
                clean = clean[np.newaxis, ...]
            if clean.shape[0] > 3:
                clean = clean[:3, ...]

        filename = self.noisy_files[idx].name
        return torch.from_numpy(noisy), clean, filename


# ============================================================
# Metrics
# ============================================================


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 100.0
    return float(10.0 * torch.log10(data_range ** 2 / (mse + 1e-12)))


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """SSIM for a single image pair [1, C, H, W]."""
    C = pred.shape[1]
    device = pred.device

    gauss = _gaussian_window(window_size, 1.5, device)
    window = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(ssim_map.mean())


# ============================================================
# Evaluation core
# ============================================================


@dataclass
class EvalResult:
    filename: str
    psnr: float = float("nan")
    ssim: float = float("nan")
    sec_per_image: float = float("nan")
    estimated_noise: float = float("nan")
    actual_steps: int = 0
    actual_lr: float = 0.0


DEFAULT_CSV_FIELDS = [
    "filename", "psnr", "ssim", "sec_per_image",
    "estimated_noise", "actual_steps", "actual_lr",
    "dataset", "checkpoint",
]


def evaluate(
    model: DeFT,
    dataset: SingleImageEvalDataset,
    device: torch.device,
    steps: int = 5,
    dataset_name: str = "",
    checkpoint_path: str = "",
) -> List[EvalResult]:
    """Run per-image adaptation and compute PSNR/SSIM."""
    results: List[EvalResult] = []
    total = len(dataset)

    for idx in tqdm(range(total), desc="Adapting"):
        noisy_tensor, clean, filename = dataset[idx]
        noisy = noisy_tensor.unsqueeze(0).to(device)

        t_start = time.perf_counter()
        denoised = model.adapt(noisy, steps=steps, episodic=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        result = EvalResult(
            filename=filename,
            sec_per_image=t_end - t_start,
            estimated_noise=getattr(model, "_estimated_noise", float("nan")) or float("nan"),
            actual_steps=getattr(model, "_actual_steps", steps) or steps,
            actual_lr=getattr(model, "_actual_lr", model.lr) or model.lr,
        )

        if clean is not None and dataset.has_clean:
            clean_tensor = clean.unsqueeze(0).to(device)
            # Match shapes
            if clean_tensor.shape[1] != denoised.shape[1]:
                if denoised.shape[1] == 1:
                    clean_tensor = clean_tensor.mean(dim=1, keepdim=True)
                elif clean_tensor.shape[1] == 1:
                    denoised = denoised.mean(dim=1, keepdim=True)
            min_h = min(clean_tensor.shape[2], denoised.shape[2])
            min_w = min(clean_tensor.shape[3], denoised.shape[3])
            clean_tensor = clean_tensor[:, :, :min_h, :min_w]
            denoised_crop = denoised[:, :, :min_h, :min_w]

            result.psnr = compute_psnr(denoised_crop, clean_tensor)
            result.ssim = compute_ssim(denoised_crop, clean_tensor)

        results.append(result)

    return results


def save_results(results: List[EvalResult], output_path: Path, extra_info: Dict[str, str]):
    """Save results to CSV with per-sample rows and a summary footer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r.filename,
                "psnr": f"{r.psnr:.4f}" if not np.isnan(r.psnr) else "nan",
                "ssim": f"{r.ssim:.4f}" if not np.isnan(r.ssim) else "nan",
                "sec_per_image": f"{r.sec_per_image:.2f}",
                "estimated_noise": f"{r.estimated_noise:.4f}" if not np.isnan(r.estimated_noise) else "nan",
                "actual_steps": r.actual_steps,
                "actual_lr": f"{r.actual_lr:.2e}",
                "dataset": extra_info.get("dataset", ""),
                "checkpoint": extra_info.get("checkpoint", ""),
            })

    # Print summary
    valid_psnr = [r.psnr for r in results if not np.isnan(r.psnr)]
    valid_ssim = [r.ssim for r in results if not np.isnan(r.ssim)]
    valid_time = [r.sec_per_image for r in results]

    print(f"\nResults saved to {output_path}")
    print(f"  Samples: {len(results)}")
    if valid_psnr:
        print(f"  PSNR: {np.mean(valid_psnr):.4f} ± {np.std(valid_psnr):.4f}")
    if valid_ssim:
        print(f"  SSIM: {np.mean(valid_ssim):.4f} ± {np.std(valid_ssim):.4f}")
    if valid_time:
        print(f"  Time/image: {np.mean(valid_time):.2f} ± {np.std(valid_time):.2f}s")


# ============================================================
# Model construction
# ============================================================


def load_model(
    checkpoint_path: str,
    device: torch.device,
    steps: int = 5,
    lr: float = 2e-4,
    adapter_type: str = "prm_prompt",
    loss_type: str = "nbr2nbr",
    loss_fn: str = "charbonnier",
    prm_mode: str = "hybrid",
    prm_spatial_gate: bool = True,
    prm_preserve_variant: str = "struct",
    prm_aggressive_variant: str = "hetero",
    prm_struct_source: bool = True,
    prm_hard_route: bool = True,
    prm_hard_grid: int = 4,
    prm_hard_topk: int = 2,
    prompt_dim: int = 32,
    prompt_len: int = 8,
    use_reliable_filter: bool = True,
    reliable_mad_k: float = 1.0,
    use_adaptive_schedule: bool = True,
    schedule_low_steps: int = 3,
    schedule_high_steps: int = 10,
    schedule_low_lr_scale: float = 0.5,
    schedule_noise_threshold: float = 0.05,
    adaptive_very_low_threshold: float = 0.025,
) -> DeFT:
    """Build a DeFT model from a pretrained backbone checkpoint.

    Canonical default arguments from the paper.
    """

    # Detect input channels from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Infer in_channels from first conv weight
    in_channels = 1
    for key in ["input_conv.double_conv.0.weight", "inc.double_conv.0.weight"]:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    out_channels = in_channels  # residual prediction
    backbone = DeFTBackbone(in_channels=in_channels, out_channels=out_channels)
    backbone.load_state_dict(state_dict, strict=False)
    print(f"[Model] Loaded backbone from {checkpoint_path} "
          f"(in={in_channels}, out={out_channels})")

    model = DeFT(
        denoiser=backbone,
        adapter_type=adapter_type,
        lr=lr,
        loss_type=loss_type,
        loss_fn=loss_fn,
        use_prompt_bank=True,
        prompt_dim=prompt_dim,
        prompt_len=prompt_len,
        prm_mode=prm_mode,
        prm_spatial_gate=prm_spatial_gate,
        prm_preserve_variant=prm_preserve_variant,
        prm_aggressive_variant=prm_aggressive_variant,
        prm_struct_source=prm_struct_source,
        prm_hard_route=prm_hard_route,
        prm_hard_grid=prm_hard_grid,
        prm_hard_topk=prm_hard_topk,
        use_reliable_filter=use_reliable_filter,
        reliable_mad_k=reliable_mad_k,
        use_adaptive_schedule=use_adaptive_schedule,
        schedule_low_steps=schedule_low_steps,
        schedule_high_steps=schedule_high_steps,
        schedule_low_lr_scale=schedule_low_lr_scale,
        schedule_noise_threshold=schedule_noise_threshold,
        adaptive_very_low_threshold=adaptive_very_low_threshold,
    )
    model.to(device)
    return model


# ============================================================
# Dataset alias lookup
# Internal directory names (Q0/Q1/Q2/Q3) follow legacy experiment order.
# The paper re-maps these for narrative clarity:
#   internal Q0/Q1 → paper Q1 (Mayo CT)
#   internal Q3     → paper Q2 (fastMRI knee)
#   internal Q2     → paper Q3 (Chest X-ray)
# ============================================================


DATASET_ALIASES = {
    "Q0_mayo_eval_S10":  "Q1 Mayo abdomen CT, Gaussian sigma=0.10",
    "Q1_mayo_eval_S06":  "Q1 Mayo abdomen CT, Gaussian sigma=0.06",
    "Q1_mayo_eval_S20":  "Q1 Mayo abdomen CT, Gaussian sigma=0.20",
    "Q3_fastmri_eval_R03": "Q2 fastMRI knee MRI, Rician sigma=0.03",
    "Q3_fastmri_eval_R07": "Q2 fastMRI knee MRI, Rician sigma=0.07",
    "Q3_fastmri_eval_R15": "Q2 fastMRI knee MRI, Rician sigma=0.15",
    "Q2_xray_eval_S06":   "Q3 Chest X-ray, Gaussian sigma=0.06",
    "Q2_xray_eval_S10":   "Q3 Chest X-ray, Gaussian sigma=0.10",
    "Q2_xray_eval_S20":   "Q3 Chest X-ray, Gaussian sigma=0.20",
}


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeFT: Descriptor-Forked Test-Time Adaptation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset aliases (used for logging only; --q-dataset-dir determines actual data):

  Q1 (Mayo abdomen CT):    Q0_mayo_eval_S10  Q1_mayo_eval_S06  Q1_mayo_eval_S20
  Q2 (fastMRI knee MRI):   Q3_fastmri_eval_R03 Q3_fastmri_eval_R07 Q3_fastmri_eval_R15
  Q3 (Chest X-ray):        Q2_xray_eval_S06   Q2_xray_eval_S10   Q2_xray_eval_S20

Canonical defaults:
  --steps 5 --lr 2e-4 --adapter prm_prompt --prm-mode hybrid
  --spatial-gate --preserve struct --aggressive hetero --struct-source
  --hard-route --hard-grid 4 --hard-topk 2
  --use-reliable --mad-k 1.0
  --adaptive-schedule --low-steps 3 --high-steps 10
        """,
    )

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to source-pretrained backbone checkpoint (.pt)")

    # Data
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset alias for logging (e.g. Q0_mayo_eval_S10)")
    parser.add_argument("--q-dataset-dir", type=str, required=True,
                        help="Path to evaluation dataset (must contain noisy/ subdirectory)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path for results")

    # Adaptation budget (overrides DCS)
    parser.add_argument("--steps", type=int, default=5,
                        help="Default adaptation steps per image (DCS may override)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Base learning rate")

    # Adapter
    parser.add_argument("--adapter", type=str, default="prm_prompt",
                        choices=["prm_prompt", "prompt_film", "film", "lora"],
                        help="Adapter type: prm_prompt (canonical DeFT), prompt_film, film, lora")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (only used with --adapter lora)")

    # PRM (dual-route)
    parser.add_argument("--prm-mode", type=str, default="hybrid",
                        choices=["hybrid", "film", "amir", "anyir"],
                        help="PRM routing mode")
    parser.add_argument("--spatial-gate", dest="prm_spatial_gate", action="store_true",
                        default=True, help="Enable spatial gate (default: on)")
    parser.add_argument("--no-spatial-gate", dest="prm_spatial_gate", action="store_false",
                        help="Disable spatial gate")
    parser.add_argument("--preserve", type=str, default="struct",
                        choices=["struct", "auto"],
                        help="Preservation route variant")
    parser.add_argument("--aggressive", type=str, default="hetero",
                        choices=["hetero", "base"],
                        help="Aggressive route variant")
    parser.add_argument("--struct-source", dest="prm_struct_source", action="store_true",
                        default=True)
    parser.add_argument("--no-struct-source", dest="prm_struct_source", action="store_false")
    parser.add_argument("--hard-route", dest="prm_hard_route", action="store_true",
                        default=True)
    parser.add_argument("--no-hard-route", dest="prm_hard_route", action="store_false")
    parser.add_argument("--hard-grid", type=int, default=4)
    parser.add_argument("--hard-topk", type=int, default=2)

    # Prompt bank (DCI)
    parser.add_argument("--prompt-dim", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=8)

    # Reliability filter
    parser.add_argument("--use-reliable", dest="use_reliable_filter", action="store_true",
                        default=True)
    parser.add_argument("--no-reliable", dest="use_reliable_filter", action="store_false")
    parser.add_argument("--mad-k", type=float, default=1.0,
                        help="MAD multiplier for reliable filter")

    # Adaptive schedule (DCS)
    parser.add_argument("--adaptive-schedule", dest="use_adaptive_schedule",
                        action="store_true", default=True)
    parser.add_argument("--no-adaptive-schedule", dest="use_adaptive_schedule",
                        action="store_false")
    parser.add_argument("--low-steps", type=int, default=3,
                        help="Steps for low-noise images")
    parser.add_argument("--high-steps", type=int, default=10,
                        help="Steps for high-noise images")
    parser.add_argument("--low-lr-scale", type=float, default=0.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05,
                        help="DCS noise threshold")
    parser.add_argument("--very-low-threshold", type=float, default=0.025)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to first N samples")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available()
                          else args.device)
    output_path = Path(args.output)

    print("=" * 72)
    print("DeFT Evaluation — Canonical defaults")
    print("=" * 72)
    print(f"  Dataset:   {args.dataset or args.q_dataset_dir}")
    print(f"  Checkpoint:{args.checkpoint}")
    print(f"  Output:    {output_path}")
    print(f"  Device:    {device}")
    print(f"  Steps/LR:  {args.steps} / {args.lr:.1e}")
    print(f"  Adapter:   {args.adapter}")
    if args.adapter == "prm_prompt":
        print(f"  PRM:       mode={args.prm_mode}, spatial_gate={args.prm_spatial_gate}, "
              f"preserve={args.preserve}, aggressive={args.aggressive}")
        print(f"             struct_source={args.prm_struct_source}, "
              f"hard_route={args.prm_hard_route}, "
              f"hard_grid={args.hard_grid}, hard_topk={args.hard_topk}")
    print(f"  Reliable:  {args.use_reliable_filter} (mad_k={args.mad_k})")
    print(f"  DCS:       adaptive={args.use_adaptive_schedule}, "
          f"low={args.low_steps}, high={args.high_steps}")
    print()

    # --- Dataset ---
    dataset = SingleImageEvalDataset(args.q_dataset_dir)
    if args.max_samples:
        dataset.noisy_files = dataset.noisy_files[:args.max_samples]
        dataset.clean_files = dataset.clean_files[:args.max_samples] if dataset.clean_files else []
        print(f"[Data] Limited to {args.max_samples} samples")
    print(f"[Data] {len(dataset)} noisy images from {args.q_dataset_dir}")
    print(f"[Data] Clean ground truth: {'YES' if dataset.has_clean else 'NO'} "
          f"({len(dataset.clean_files)} files)")
    print()

    # --- Model ---
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        steps=args.steps,
        lr=args.lr,
        adapter_type=args.adapter,
        prm_mode=args.prm_mode,
        prm_spatial_gate=args.prm_spatial_gate,
        prm_preserve_variant=args.preserve,
        prm_aggressive_variant=args.aggressive,
        prm_struct_source=args.prm_struct_source,
        prm_hard_route=args.prm_hard_route,
        prm_hard_grid=args.hard_grid,
        prm_hard_topk=args.hard_topk,
        prompt_dim=args.prompt_dim,
        prompt_len=args.prompt_len,
        use_reliable_filter=args.use_reliable_filter,
        reliable_mad_k=args.mad_k,
        use_adaptive_schedule=args.use_adaptive_schedule,
        schedule_low_steps=args.low_steps,
        schedule_high_steps=args.high_steps,
        schedule_low_lr_scale=args.low_lr_scale,
        schedule_noise_threshold=args.noise_threshold,
        adaptive_very_low_threshold=args.very_low_threshold,
    )

    # --- Evaluate ---
    torch.backends.cudnn.benchmark = False

    results = evaluate(
        model=model,
        dataset=dataset,
        device=device,
        steps=args.steps,
        dataset_name=args.dataset or Path(args.q_dataset_dir).name,
        checkpoint_path=args.checkpoint,
    )

    # --- Save ---
    save_results(results, output_path, {
        "dataset": args.dataset or Path(args.q_dataset_dir).name,
        "checkpoint": str(Path(args.checkpoint).name),
    })

    # --- Cleanup ---
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
