"""
DeFT: Descriptor-Forked Test-Time Adaptation for multi-domain medical image denoising.

One descriptor state, three projections:
  DCI (Descriptor Conditioning Interface) — conditions adapter behavior,
  PRM (Polarized Route Mixture)      — dual-route spatial adaptation space,
  DCS (Descriptor-Conditioned Scheduler) — adaptive steps/lr from descriptor.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deft.dci import (
    FiLMLayer,
    FiLMWrapper,
    LoRALayer,
    LoRAWrapper,
    PromptFiLMLayer,
    PromptFiLMWrapper,
)
from deft.descriptor import (
    DescriptorState,
    build_descriptor_tensor,
    estimate_noise_level,
)
from deft.dcs import compute_adaptive_budget
from deft.loss import Neighbor2NeighborLoss
from deft.prm import PolarizedRouteMixture, PromptBank, SpatialGate


# ============================================================================
# Utility functions
# ============================================================================


def _compute_gradient(x: torch.Tensor,
                      sobel_x: torch.Tensor,
                      sobel_y: torch.Tensor) -> torch.Tensor:
    """Compute image gradient magnitude using Sobel filters."""
    if x.size(1) > 1:
        x = x.mean(dim=1, keepdim=True)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)


# ============================================================================
# SAMOptimizer
# ============================================================================


class SAMOptimizer(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    Two-step optimization to find flatter minima:
    1. first_step: compute epsilon = rho * grad / ||grad|| and move to theta + epsilon
    2. second_step: compute gradient at theta + epsilon, restore theta, then update

    Reference: https://github.com/mr-eggplant/SAR/blob/main/sam.py
    """

    def __init__(self, params, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group.setdefault('rho', rho)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ============================================================================
# DeFT — main class
# ============================================================================


class DeFT(nn.Module):
    """DeFT: Descriptor-Forked Test-Time Adaptation.

    One descriptor state, three projections:
    DCI (conditioning interface), PRM (polarized route mixture),
    DCS (descriptor-conditioned scheduler).
    """

    def __init__(  # Canonical DeFT defaults
        self,
        denoiser: nn.Module,
        # --- Adapter architecture ---
        adapter_type: str = 'prm_prompt',
        lora_rank: int = 8,
        lora_rezero: bool = False,
        adapter_layers: Optional[List[str]] = None,
        # --- Loss ---
        loss_type: str = 'nbr2nbr',
        loss_fn: str = 'charbonnier',
        lr: float = 2e-4,
        # --- Prompt Bank (DCI) ---
        use_prompt_bank: bool = True,
        prompt_dim: int = 32,
        prompt_len: int = 8,
        # --- Adaptive Schedule (DCS) ---
        use_adaptive_schedule: bool = True,
        schedule_low_steps: int = 3,
        schedule_high_steps: int = 10,
        schedule_low_lr_scale: float = 0.5,
        schedule_noise_threshold: float = 0.05,
        adaptive_very_low_threshold: float = 0.025,
        adaptive_very_low_steps: Optional[int] = None,
        adaptive_very_low_lr_scale: float = 1.0,
        # --- Reliability filter ---
        use_reliable_filter: bool = True,
        reliable_patch_size: int = 64,
        reliable_mad_k: float = 1.0,
        # --- PRM: dual-route adapter ---
        prm_mode: str = "hybrid",
        prm_policy: str = "auto",
        prm_res_scale: float = 0.25,
        prm_freq_strength: float = 0.1,
        prm_spatial_gate: bool = True,
        prm_preserve_variant: str = "struct",
        prm_aggressive_variant: str = "hetero",
        prm_gate_temperature: float = 1.0,
        prm_gate_entropy_weight: float = 0.0,
        prm_gate_sparse_weight: float = 0.0,
        prm_struct_weight: float = 0.0,
        prm_struct_gate_weight: float = 0.0,
        prm_struct_source: bool = True,
        prm_rank_weight: float = 0.0,
        prm_rank_margin: float = 0.15,
        prm_rank_grid: int = 4,
        prm_rank_topk: int = 2,
        prm_hard_route: bool = True,
        prm_hard_grid: int = 4,
        prm_hard_topk: int = 2,
        prm_adaptive_freq_mod: bool = False,
        prm_adaptive_freq_strength: float = 0.1,
        prm_mask_guided_preserve: bool = False,
        prm_mask_keep_ratio: float = 0.25,
        prm_gated_degprop: bool = False,
        prm_degprop_strength: float = 1.0,
        prm_rezero: bool = False,
        # --- Ablation / diagnostic ---
        desc_drop: Optional[str] = None,
        desc_override: Optional[str] = None,
        desc_persist_fixed: bool = False,
        single_route: Optional[str] = None,
        same_init: bool = False,
        descriptor_init_scale: bool = False,
        # --- External bypass (eval CLI) ---
        force_bypass: bool = False,
        force_bypass_film: bool = False,
        force_bypass_eval_mode: bool = False,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.adapter_type = adapter_type.lower()
        self.adapter_layers = adapter_layers
        self.lora_rank = lora_rank
        self.lora_rezero = lora_rezero
        self.loss_type = loss_type.lower()
        self.loss_fn = loss_fn.lower()
        self.lr = lr

        # --- Prompt bank (DCI) ---
        self.use_prm_prompt = self.adapter_type == "prm_prompt"
        self.use_prompt_bank = use_prompt_bank or self.adapter_type in {"prm_prompt"}
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len

        # --- Dual-route (PRM) ---
        self.prm_mode = prm_mode.lower()
        self.prm_policy = prm_policy.lower()
        self.prm_res_scale = prm_res_scale
        self.prm_freq_strength = prm_freq_strength
        self.prm_spatial_gate = prm_spatial_gate
        self.prm_preserve_variant = prm_preserve_variant.lower()
        self.prm_aggressive_variant = prm_aggressive_variant.lower()
        self.prm_struct_use_source = bool(prm_struct_source)
        self.prm_gate_temperature = prm_gate_temperature
        self.prm_gate_entropy_weight = prm_gate_entropy_weight
        self.prm_gate_sparse_weight = prm_gate_sparse_weight
        self.prm_struct_weight = prm_struct_weight
        self.prm_struct_gate_weight = prm_struct_gate_weight
        self.prm_rank_weight = prm_rank_weight
        self.prm_rank_margin = prm_rank_margin
        self.prm_rank_grid = prm_rank_grid
        self.prm_rank_topk = prm_rank_topk
        self.prm_hard_route = prm_hard_route
        self.prm_hard_grid = prm_hard_grid
        self.prm_hard_topk = prm_hard_topk
        self.prm_adaptive_freq_mod = prm_adaptive_freq_mod
        self.prm_adaptive_freq_strength = prm_adaptive_freq_strength
        self.prm_mask_guided_preserve = prm_mask_guided_preserve
        self.prm_mask_keep_ratio = prm_mask_keep_ratio
        self.prm_gated_degprop = prm_gated_degprop
        self.prm_degprop_strength = prm_degprop_strength

        if self.use_prm_prompt:
            self._validate_prm_params()

        # --- Adaptive schedule (DCS) ---
        self.use_adaptive_schedule = use_adaptive_schedule
        self.schedule_low_steps = schedule_low_steps
        self.schedule_high_steps = schedule_high_steps
        self.schedule_low_lr_scale = schedule_low_lr_scale
        self.schedule_noise_threshold = schedule_noise_threshold
        self.adaptive_very_low_threshold = adaptive_very_low_threshold
        self.adaptive_very_low_steps = adaptive_very_low_steps
        self.adaptive_very_low_lr_scale = adaptive_very_low_lr_scale

        # --- Reliability filter ---
        self.use_reliable_filter = use_reliable_filter
        self.reliable_patch_size = reliable_patch_size
        self.reliable_mad_k = reliable_mad_k

        # --- Descriptor ---
        self.descriptor_dim = 6
        self.prompt_prior_dim = 1
        self.prompt_bank = None

        # --- Extra modules ---
        self.extra_trainable_modules: Dict[str, nn.Module] = {}
        self.extra_initial_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.trainable_param_anchor: Dict[str, torch.Tensor] = {}

        if self.use_prompt_bank:
            self.prompt_prior_dim = prompt_dim
            self.prompt_bank = PromptBank(
                descriptor_dim=self.descriptor_dim,
                prompt_dim=prompt_dim,
                prompt_len=prompt_len,
            )
            self.extra_trainable_modules["prompt_bank"] = self.prompt_bank
            print(f"[DeFT] Prompt bank enabled (dim={prompt_dim}, len={prompt_len})")

        if self.use_prm_prompt:
            print(
                f"[DeFT] PRM dual-route adapter enabled "
                f"(mode={self.prm_mode}, policy={self.prm_policy}, "
                f"spatial_gate={self.prm_spatial_gate}, "
                f"preserve={self.prm_preserve_variant}, "
                f"aggressive={self.prm_aggressive_variant})"
            )
            if (
                self.prm_gate_temperature != 1.0
                or self.prm_gate_entropy_weight > 0
                or self.prm_gate_sparse_weight > 0
                or self.prm_struct_weight > 0
                or self.prm_struct_gate_weight > 0
                or self.prm_rank_weight > 0
                or self.prm_hard_route
                or self.prm_adaptive_freq_mod
                or self.prm_mask_guided_preserve
                or self.prm_gated_degprop
            ):
                print(
                    "[DeFT] PRM route regularization enabled "
                    f"(gtemp={self.prm_gate_temperature}, "
                    f"gent={self.prm_gate_entropy_weight}, "
                    f"gsparse={self.prm_gate_sparse_weight}, "
                    f"sthm={self.prm_struct_weight}, "
                    f"sthg={self.prm_struct_gate_weight}, "
                    f"rkw={self.prm_rank_weight}, "
                    f"rkm={self.prm_rank_margin}, "
                    f"rkg={self.prm_rank_grid}, "
                    f"rkk={self.prm_rank_topk}, "
                    f"hard={int(self.prm_hard_route)}, "
                    f"hardg={self.prm_hard_grid}, "
                    f"hardk={self.prm_hard_topk}, "
                    f"afm={int(self.prm_adaptive_freq_mod)}, "
                    f"maskp={int(self.prm_mask_guided_preserve)}, "
                    f"degprop={int(self.prm_gated_degprop)})"
                )

        # --- State ---
        self.desc_drop = desc_drop
        self.desc_override = desc_override
        self.desc_persist_fixed = bool(desc_persist_fixed)
        self.single_route = single_route
        self.same_init = same_init
        self.descriptor_init_scale = bool(descriptor_init_scale)
        self.force_bypass = bool(force_bypass)
        self.force_bypass_film = bool(force_bypass_film)
        self.force_bypass_eval_mode = bool(force_bypass_eval_mode)
        self.prm_rezero = bool(prm_rezero)

        # --- Runtime state ---
        self.blur_kernel = None
        self.blur_kernel_size = 7
        self.sobel_x = None
        self.sobel_y = None
        self.initial_state = None
        self.ema_teacher = None
        self.source_snapshot = None
        self.ss_loss = None
        self.use_film = False
        self.use_lora = False
        self.use_prompt_film = False

        self._last_prm_aux_losses = {
            "gate_entropy": float("nan"),
            "gate_sparse": float("nan"),
            "preserve_struct": float("nan"),
            "aggressive_smooth": float("nan"),
            "gate_align": float("nan"),
            "route_rank": float("nan"),
        }
        self._last_prm_diag: Dict[str, float] = {}
        self._keep_fractions: List[float] = []
        self._grad_norms: List[float] = []
        self._delta_theta = 0.0
        self._last_keep_fraction = 1.0
        self._estimated_noise = None
        self._is_low_noise = None
        self._noise_category = None
        self._last_route_gates = None
        self._cached_descriptor = None
        self._actual_steps = None
        self._actual_lr = None
        self._fixed_desc = None

        # --- Init N2N loss ---
        if self.loss_type == 'nbr2nbr':
            self.ss_loss = Neighbor2NeighborLoss(loss_fn=self.loss_fn)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        print(f"[DeFT] Loss engine: N2N ({self.loss_type})")

        # --- Init blur kernel ---
        if self.use_reliable_filter:
            self._init_blur_kernel()

        # --- Configure model ---
        self._configure_model()
        self._save_initial_state()

    def _validate_prm_params(self):
        valid_modes = {"film", "amir", "anyir", "hybrid"}
        valid_policies = {"auto", "aggressive", "preserve"}
        valid_preserve = {"auto", "struct"}
        valid_aggressive = {"base", "hetero"}
        if self.prm_mode not in valid_modes:
            raise ValueError(f"Unknown prm_mode: {self.prm_mode}")
        if self.prm_policy not in valid_policies:
            raise ValueError(f"Unknown prm_policy: {self.prm_policy}")
        if self.prm_preserve_variant not in valid_preserve:
            raise ValueError(f"Unknown prm_preserve_variant: {self.prm_preserve_variant}")
        if self.prm_aggressive_variant not in valid_aggressive:
            raise ValueError(f"Unknown prm_aggressive_variant: {self.prm_aggressive_variant}")
        if self.prm_gate_temperature <= 0:
            raise ValueError("prm_gate_temperature must be > 0")
        for name, value in {
            "prm_gate_entropy_weight": self.prm_gate_entropy_weight,
            "prm_gate_sparse_weight": self.prm_gate_sparse_weight,
            "prm_struct_weight": self.prm_struct_weight,
            "prm_struct_gate_weight": self.prm_struct_gate_weight,
            "prm_rank_weight": self.prm_rank_weight,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name, value in {
            "prm_adaptive_freq_strength": self.prm_adaptive_freq_strength,
            "prm_degprop_strength": self.prm_degprop_strength,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.prm_struct_use_source and self.prm_preserve_variant != "struct":
            raise ValueError("prm_struct_source requires prm_preserve_variant='struct'")
        if self.prm_rank_weight > 0 and not self.prm_spatial_gate:
            raise ValueError("prm_rank_weight requires prm_spatial_gate=True")
        if self.prm_rank_weight > 0 and self.prm_policy != "auto":
            raise ValueError("prm_rank_weight requires prm_policy='auto'")
        if self.prm_rank_margin < 0:
            raise ValueError("prm_rank_margin must be >= 0")
        if self.prm_rank_grid < 2:
            raise ValueError("prm_rank_grid must be >= 2")
        if self.prm_rank_topk < 1:
            raise ValueError("prm_rank_topk must be >= 1")
        if self.prm_hard_route and not self.prm_spatial_gate:
            raise ValueError("prm_hard_route requires prm_spatial_gate=True")
        if self.prm_hard_route and self.prm_policy != "auto":
            raise ValueError("prm_hard_route requires prm_policy='auto'")
        if self.prm_hard_grid < 2:
            raise ValueError("prm_hard_grid must be >= 2")
        if self.prm_hard_topk < 1:
            raise ValueError("prm_hard_topk must be >= 1")
        if self.prm_mask_guided_preserve and not self.prm_spatial_gate and not self.prm_hard_route:
            raise ValueError("prm_mask_guided_preserve requires spatial or hard routing")
        if not (0.0 < self.prm_mask_keep_ratio <= 1.0):
            raise ValueError("prm_mask_keep_ratio must be in (0, 1]")

    # ================================================================
    # Model configuration
    # ================================================================

    def _configure_model(self):
        """Configure the model: enable Norm layers or insert adapter layers.

        Adapter insertion strategy:
        - Only insert at Bottleneck + Decoder first two blocks (B + D4 + D3)
        - Do not touch Encoder, protecting edge/detail extraction.
        """
        self.denoiser.train()
        self.denoiser.requires_grad_(False)

        trainable_count = 0
        has_norm = False
        for name, m in self.denoiser.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                trainable_count += sum(p.numel() for p in m.parameters())
                has_norm = True
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
                m.requires_grad_(True)
                trainable_count += sum(p.numel() for p in m.parameters())
                has_norm = True

        self.use_film = False
        self.use_lora = False
        self.use_prompt_film = False
        adapter_count = 0

        if not has_norm:
            adapter_name = self.adapter_type.upper()
            print(f"[DeFT] No Norm layers found, inserting {adapter_name} layers (strategy: B+D4+D3)...")

            if self.adapter_type == 'film':
                self.use_film = True
            elif self.adapter_type == 'lora':
                self.use_lora = True
            elif self.adapter_type in ('prompt_film', 'prm_prompt'):
                self.use_prompt_film = True
            else:
                raise ValueError(
                    f"Unknown adapter_type: {self.adapter_type}. "
                    "Choose 'film', 'lora', 'prompt_film', or 'prm_prompt'"
                )

            base_targets = ['down4', 'up1', 'up2']
            available_modules = [n for n, _ in self.denoiser.named_modules() if n]

            prefix = ""
            if hasattr(self.denoiser, 'unet_body'):
                prefix = "unet_body."
                print(f"  Detected wrapper: using prefix '{prefix}'")

            if self.adapter_layers is not None:
                base_targets = self.adapter_layers
            else:
                is_shallow_unet = any('downs.2' in m for m in available_modules)
                is_restormer = any(m.startswith('model.latent') for m in available_modules)
                is_deft_backbone = any('encoder4' in m for m in available_modules)
                if is_restormer:
                    print("  Detected Restormer-like architecture.")
                    base_targets = ['model.latent', 'model.decoder_level3', 'model.decoder_level2']
                elif is_shallow_unet:
                    print("  Detected ShallowUNet architecture.")
                    base_targets = ['downs.2', 'ups.1', 'ups.3']
                elif is_deft_backbone:
                    print("  Detected DeFTBackbone architecture (encoder4+decoder1+decoder2).")
                    base_targets = ['encoder4', 'decoder1', 'decoder2']
                else:
                    base_targets = ['down4', 'up1', 'up2']

            adapter_targets = [f"{prefix}{t}" for t in base_targets]

            print(f"  Available top-level modules: {[m for m in available_modules if '.' not in m][:10]}...")

            for target_path in adapter_targets:
                try:
                    original_module = self.denoiser.get_submodule(target_path)
                except AttributeError:
                    print(f"  [WARN] Module '{target_path}' not found, skipping...")
                    continue

                out_channels = None
                for m in original_module.modules():
                    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                        out_channels = m.out_channels

                if out_channels is None:
                    print(f"  [WARN] No Conv2d/ConvTranspose2d found in '{target_path}', skipping...")
                    continue

                parts = target_path.rsplit('.', 1)
                if len(parts) == 1:
                    parent = self.denoiser
                    attr_name = parts[0]
                else:
                    parent = self.denoiser.get_submodule(parts[0])
                    attr_name = parts[1]

                if self.use_prompt_film:
                    if self.use_prm_prompt:
                        wrapped = PolarizedRouteMixture(
                            original_module,
                            out_channels,
                            prompt_dim=self.prompt_prior_dim,
                            descriptor_dim=self.descriptor_dim,
                            route_mode=self.prm_mode,
                            route_policy=self.prm_policy,
                            res_scale=self.prm_res_scale,
                            freq_strength=self.prm_freq_strength,
                            spatial_gate=self.prm_spatial_gate,
                            preserve_variant=self.prm_preserve_variant,
                            aggressive_variant=self.prm_aggressive_variant,
                            gate_temperature=self.prm_gate_temperature,
                            hard_route=self.prm_hard_route,
                            hard_grid=self.prm_hard_grid,
                            hard_topk=self.prm_hard_topk,
                            adaptive_freq_mod=self.prm_adaptive_freq_mod,
                            adaptive_freq_strength=self.prm_adaptive_freq_strength,
                            mask_guided_preserve=self.prm_mask_guided_preserve,
                            mask_keep_ratio=self.prm_mask_keep_ratio,
                            gated_degprop=self.prm_gated_degprop,
                            degprop_strength=self.prm_degprop_strength,
                            prm_disabled=False,
                            single_route=self.single_route,
                        )
                    else:
                        wrapped = PromptFiLMWrapper(original_module, out_channels, prior_dim=self.prompt_prior_dim)
                elif self.use_film:
                    wrapped = FiLMWrapper(original_module, out_channels)
                else:
                    wrapped = LoRAWrapper(original_module, out_channels, rank=self.lora_rank, use_rezero=self.lora_rezero)

                param_count = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
                setattr(parent, attr_name, wrapped)
                trainable_count += param_count
                adapter_count += 1
                print(f"  + Inserted {adapter_name} at '{target_path}' (channels={out_channels}, params={param_count})")

            for name, m in self.denoiser.named_modules():
                if isinstance(m, (FiLMLayer, LoRALayer, PromptFiLMLayer, PolarizedRouteMixture)):
                    m.requires_grad_(True)

        total_params = sum(p.numel() for p in self.denoiser.parameters())
        if self.use_prm_prompt:
            mode = (
                f"PRM({self.prm_mode},{self.prm_policy},"
                f"spg={int(self.prm_spatial_gate)},"
                f"pres={self.prm_preserve_variant},"
                f"agg={self.prm_aggressive_variant},"
                f"hard={int(self.prm_hard_route)},"
                f"afm={int(self.prm_adaptive_freq_mod)},"
                f"maskp={int(self.prm_mask_guided_preserve)},"
                f"degprop={int(self.prm_gated_degprop)})"
            )
        elif self.use_prompt_film:
            mode = "PromptFiLM"
        elif self.use_film:
            mode = "FiLM"
        elif self.use_lora:
            mode = f"LoRA (rank={self.lora_rank})"
        else:
            mode = "Norm"
        print(f"[DeFT] Mode: {mode}, Trainable: {trainable_count:,} / {total_params:,} "
              f"({100*trainable_count/total_params:.2f}%)")

        if not has_norm:
            assert adapter_count > 0, (
                f"[DeFT] FATAL: Adapter insertion failed! No modules matched. "
                f"Expected targets: {adapter_targets}. "
                f"Available modules: {[n for n, _ in self.denoiser.named_modules() if 'down' in n or 'up' in n][:20]}"
            )

        assert trainable_count > 0, (
            f"[DeFT] FATAL: No trainable parameters! "
            f"Mode={mode}, has_norm={has_norm}, adapter_count={adapter_count}. "
            f"This means adaptation will have NO effect. Check backbone compatibility."
        )

    # ================================================================
    # Lifecycle
    # ================================================================

    def _init_blur_kernel(self):
        k = self.blur_kernel_size
        sigma = k / 6.0
        x = torch.arange(k).float() - k // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()
        self.blur_kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

    def _apply_blur(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur_kernel is None:
            return x
        b, c, h, w = x.shape
        kernel = self.blur_kernel.to(x.device, x.dtype)
        pad = self.blur_kernel_size // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')
        kernel_expanded = kernel.expand(c, 1, -1, -1)
        return F.conv2d(x_padded, kernel_expanded, groups=c)

    def _init_sobel(self, device, dtype):
        if self.sobel_x is None:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=dtype, device=device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=dtype, device=device).view(1, 1, 3, 3)
            self.sobel_x = sobel_x
            self.sobel_y = sobel_y

    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        self._init_sobel(x.device, x.dtype)
        return _compute_gradient(x, self.sobel_x, self.sobel_y)

    def _compute_structure_map(self, x: torch.Tensor) -> torch.Tensor:
        edge = self._compute_gradient(x)
        b = edge.size(0)
        flat = edge.view(b, -1)
        e_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
        e_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (edge - e_min) / (e_max - e_min + 1e-8)

    def _save_initial_state(self):
        self.initial_state = deepcopy(self.denoiser.state_dict())
        self.extra_initial_state = {
            name: deepcopy(module.state_dict())
            for name, module in self.extra_trainable_modules.items()
        }
        if self.prm_struct_use_source:
            self.source_snapshot = deepcopy(self.denoiser)
            self.source_snapshot.eval()
            for p in self.source_snapshot.parameters():
                p.requires_grad = False

        self._refresh_trainable_param_anchor()

    def reset(self):
        """Reset model to initial state."""
        if self.initial_state is not None:
            self.denoiser.load_state_dict(self.initial_state)
            self.denoiser.train()
            self.denoiser.requires_grad_(False)

            for m in self.denoiser.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d,
                                     FiLMLayer, LoRALayer, PromptFiLMLayer, PolarizedRouteMixture)):
                    m.requires_grad_(True)

            self._set_adapter_condition(None, None)
            self._set_structure_target(None)
            self._clear_prm_runtime_cache()

        for name, module in self.extra_trainable_modules.items():
            if name in self.extra_initial_state:
                module.load_state_dict(self.extra_initial_state[name])
            module.train()

        self._keep_fractions = []
        self._grad_norms = []
        self._delta_theta = 0.0
        self._last_keep_fraction = 1.0
        self._estimated_noise = None
        self._is_low_noise = None
        self._noise_category = None
        self._last_route_gates = None
        self._cached_descriptor = None
        if not (self.desc_persist_fixed and self.desc_override == 'fixed_first_image'):
            self._fixed_desc = None
        self._last_prm_diag = {
            "route_gate_mean": float("nan"),
            "route_soft_gate_mean": float("nan"),
            "route_gate_std": float("nan"),
            "route_gate_lowfrac": float("nan"),
            "route_gate_highfrac": float("nan"),
            "route_aggr_delta_norm": float("nan"),
            "route_prsv_delta_norm": float("nan"),
            "route_preserve_mask_frac": float("nan"),
            "route_degprop_gate_mean": float("nan"),
            "route_freq_response": float("nan"),
        }
        self._last_prm_aux_losses = {
            "gate_entropy": float("nan"),
            "gate_sparse": float("nan"),
            "preserve_struct": float("nan"),
            "aggressive_smooth": float("nan"),
            "gate_align": float("nan"),
            "route_rank": float("nan"),
        }
        self._refresh_trainable_param_anchor()

    # ================================================================
    # DCI: Descriptor Conditioning Interface
    # ================================================================

    def _set_adapter_condition(
        self,
        prompt_tensor: Optional[torch.Tensor],
        descriptor: Optional[torch.Tensor] = None,
    ) -> None:
        for module in self.denoiser.modules():
            if hasattr(module, "set_condition"):
                module.set_condition(prompt_tensor, descriptor)
            elif isinstance(module, PromptFiLMWrapper):
                module.current_prompt = prompt_tensor

    def _set_structure_target(
        self,
        structure_tensor: Optional[torch.Tensor],
    ) -> None:
        for module in self.denoiser.modules():
            if hasattr(module, "set_structure_target"):
                module.set_structure_target(structure_tensor)

    def _build_structure_anchor(self, noisy: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.prm_struct_use_source:
            return None
        if self.source_snapshot is None:
            self.source_snapshot = deepcopy(self.denoiser)
            self.source_snapshot.eval()
            for p in self.source_snapshot.parameters():
                p.requires_grad = False
        with torch.no_grad():
            anchor_raw = self.source_snapshot(noisy)
            if isinstance(anchor_raw, tuple):
                anchor_raw = anchor_raw[0]
            if anchor_raw.abs().mean() < 0.5:
                anchor_img = torch.clamp(noisy + anchor_raw, 0, 1)
            else:
                anchor_img = torch.clamp(anchor_raw, 0, 1)
        self._clear_model_runtime_cache(self.source_snapshot)
        return self._compute_structure_map(anchor_img).detach()

    # ================================================================
    # DCS: Descriptor-Conditioned Scheduler
    # ================================================================

    def _estimate_noise_level(self, noisy: torch.Tensor) -> float:
        return estimate_noise_level(noisy)

    @torch.no_grad()
    def _build_degradation_descriptor(self, noisy: torch.Tensor) -> torch.Tensor:
        """Build a compact diagnostic descriptor for prompting / routing.

        Features:
        0. estimated noise level
        1. image std
        2. mean gradient magnitude
        3. mean high-frequency energy
        4. source residual magnitude
        5. mean intensity
        """
        noise_level = self._estimate_noise_level(noisy)
        source_out = self._denoise(noisy)
        descriptor = build_descriptor_tensor(noisy, noise_level, source_out)

        if self.desc_drop:
            for comp in self.desc_drop.split(','):
                comp = comp.strip().lower()
                if comp == 'mad':
                    descriptor[:, 0] = 0.0
                elif comp == 'intensity':
                    descriptor[:, 1] = 0.0
                    descriptor[:, 5] = 0.0
                elif comp == 'struct':
                    descriptor[:, 2] = 0.0
                    descriptor[:, 3] = 0.0
                elif comp == 'src':
                    descriptor[:, 4] = 0.0

        if self.desc_override is not None and self.desc_override != '':
            if self.desc_override == 'random':
                descriptor = torch.rand_like(descriptor)
            elif self.desc_override == 'fixed_first_image':
                if not hasattr(self, '_fixed_desc') or self._fixed_desc is None:
                    self._fixed_desc = descriptor[0:1].clone().detach()
                descriptor = self._fixed_desc.expand_as(descriptor)

        return descriptor

    # ================================================================
    # PRM: Polarized Route Mixture
    # ================================================================

    def _iter_prm_wrappers(self) -> List[PolarizedRouteMixture]:
        return [
            module for module in self.denoiser.modules()
            if isinstance(module, PolarizedRouteMixture)
        ]

    def _rezero_gate_params(self) -> list:
        gate_params = []
        for m in self.denoiser.modules():
            if hasattr(m, 'rezero_g') and m.rezero_g is not None:
                gate_params.append(m.rezero_g)
        return gate_params

    def _rezero_max_gate(self) -> float:
        gates = self._rezero_gate_params()
        if not gates:
            return 0.0
        return max(g.item() for g in gates)

    def _clear_prm_runtime_cache(self) -> None:
        for module in self._iter_prm_wrappers():
            if hasattr(module, "clear_runtime_cache"):
                module.clear_runtime_cache()

    @staticmethod
    def _clear_model_runtime_cache(model: Optional[nn.Module]) -> None:
        if model is None:
            return
        for module in model.modules():
            if hasattr(module, "clear_runtime_cache"):
                module.clear_runtime_cache()

    def _prime_prm_state(self, noisy: torch.Tensor):
        if not self.use_prm_prompt:
            return
        with torch.no_grad():
            _ = self._denoise(noisy)

    def _collect_prm_route_mask(
        self, kind: str, noisy: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        modules = self._iter_prm_wrappers()
        if not modules:
            return None
        route_masks: List[torch.Tensor] = []
        ref = noisy[:, :1]
        for module in modules:
            route_mask = module.build_route_mask(kind, target_shape=ref)
            if route_mask is None:
                continue
            if route_mask.size(1) != 1:
                route_mask = route_mask.mean(dim=1, keepdim=True)
            route_masks.append(route_mask)
        if not route_masks:
            return None
        return torch.stack(route_masks, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def _collect_prm_diagnostics(self) -> Dict[str, float]:
        stats = {
            "route_gate_mean": float("nan"),
            "route_soft_gate_mean": float("nan"),
            "route_gate_std": float("nan"),
            "route_gate_lowfrac": float("nan"),
            "route_gate_highfrac": float("nan"),
            "route_aggr_delta_norm": float("nan"),
            "route_prsv_delta_norm": float("nan"),
            "route_preserve_mask_frac": float("nan"),
            "route_degprop_gate_mean": float("nan"),
            "route_freq_response": float("nan"),
        }
        modules = self._iter_prm_wrappers()
        if not modules:
            return stats

        metric_map = {
            "route_gate_mean": "last_gate_mean",
            "route_soft_gate_mean": "last_soft_gate_mean",
            "route_gate_std": "last_gate_std",
            "route_gate_lowfrac": "last_gate_lowfrac",
            "route_gate_highfrac": "last_gate_highfrac",
            "route_aggr_delta_norm": "last_aggressive_delta_norm",
            "route_prsv_delta_norm": "last_preserve_delta_norm",
            "route_preserve_mask_frac": "last_preserve_mask_frac",
            "route_degprop_gate_mean": "last_degprop_gate_mean",
            "route_freq_response": "last_freq_response",
        }
        for out_key, attr_name in metric_map.items():
            vals = []
            for module in modules:
                val = getattr(module, attr_name, None)
                if val is None:
                    continue
                if torch.is_tensor(val):
                    if val.numel() != 1:
                        continue
                    if torch.isnan(val).item():
                        continue
                    vals.append(float(val.detach().float().cpu().item()))
                    continue
                if isinstance(val, float) and math.isnan(val):
                    continue
                vals.append(float(val))
            if vals:
                stats[out_key] = float(sum(vals) / len(vals))
        return stats

    @staticmethod
    def _weighted_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_map = (pred - target).abs()
        if weight is None:
            return loss_map.mean()
        if weight.size(1) == 1 and loss_map.size(1) > 1:
            weight = weight.expand_as(loss_map)
        weighted = loss_map * weight
        denom = weight.sum().clamp(min=1e-6)
        return weighted.sum() / denom

    @staticmethod
    def _zero_prm_aux_losses(device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        return {
            "gate_entropy": zero.clone(),
            "gate_sparse": zero.clone(),
            "preserve_struct": zero.clone(),
            "aggressive_smooth": zero.clone(),
            "gate_align": zero.clone(),
            "route_rank": zero.clone(),
        }

    @staticmethod
    def _normalize_prm_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        x_min = flat.min(dim=1)[0].view(-1, 1, 1, 1)
        x_max = flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _compute_prm_rank_loss(
        self, module: PolarizedRouteMixture, gate: torch.Tensor,
        x: torch.Tensor, structure_map: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if x.dim() != 4 or min(x.shape[-2:]) < 2:
            return None
        gate_map = module._match_target_shape(gate, x[:, :1]).mean(dim=1, keepdim=True)
        structure_map = module._match_target_shape(structure_map.detach(), x[:, :1]).mean(dim=1, keepdim=True)

        feat_gray = x.detach().mean(dim=1, keepdim=True)
        feat_smooth = F.avg_pool2d(feat_gray, kernel_size=3, stride=1, padding=1)
        noise_proxy = self._normalize_prm_map((feat_gray - feat_smooth).abs())

        aggressive_priority = self._normalize_prm_map((1.0 - structure_map) * (0.5 + 0.5 * noise_proxy))
        preserve_priority = self._normalize_prm_map(structure_map)

        grid = int(min(self.prm_rank_grid, gate_map.shape[-2], gate_map.shape[-1]))
        if grid < 2:
            return None

        gate_grid = F.adaptive_avg_pool2d(gate_map, (grid, grid)).flatten(1)
        aggressive_grid = F.adaptive_avg_pool2d(aggressive_priority, (grid, grid)).flatten(1)
        preserve_grid = F.adaptive_avg_pool2d(preserve_priority, (grid, grid)).flatten(1)

        num_cells = gate_grid.size(1)
        topk = min(int(self.prm_rank_topk), max(1, num_cells // 2))
        if topk < 1:
            return None

        aggr_vals, aggr_idx = aggressive_grid.topk(topk, dim=1, largest=True)
        preserve_masked = preserve_grid.clone()
        preserve_masked.scatter_(1, aggr_idx, -1.0)
        prsv_vals, prsv_idx = preserve_masked.topk(topk, dim=1, largest=True)

        gate_aggr = gate_grid.gather(1, aggr_idx)
        gate_prsv = gate_grid.gather(1, prsv_idx)

        valid = (aggr_vals - prsv_vals) > 1e-3
        if not torch.any(valid):
            return None

        gate_aggr_valid = gate_aggr[valid]
        gate_prsv_valid = gate_prsv[valid]
        rank_target = torch.ones_like(gate_aggr_valid)
        return F.margin_ranking_loss(
            gate_aggr_valid, gate_prsv_valid, rank_target,
            margin=self.prm_rank_margin,
        )

    def _compute_prm_aux_losses(self, device: torch.device) -> Dict[str, torch.Tensor]:
        losses = self._zero_prm_aux_losses(device)
        wrappers = [
            module for module in self.denoiser.modules()
            if isinstance(module, PolarizedRouteMixture)
        ]
        if not wrappers:
            return losses

        gate_entropy_terms: List[torch.Tensor] = []
        gate_sparse_terms: List[torch.Tensor] = []
        preserve_terms: List[torch.Tensor] = []
        aggressive_terms: List[torch.Tensor] = []
        gate_align_terms: List[torch.Tensor] = []
        route_rank_terms: List[torch.Tensor] = []

        for module in wrappers:
            gate = getattr(module, "last_soft_gate", None)
            if gate is None:
                gate = getattr(module, "last_gate", None)
            if gate is not None:
                gate_clamped = gate.clamp(1e-6, 1.0 - 1e-6)
                gate_entropy_terms.append(
                    -(gate_clamped * torch.log(gate_clamped) + (1.0 - gate_clamped) * torch.log(1.0 - gate_clamped)).mean()
                )
                gate_sparse_terms.append(gate_clamped.mean())

            if not (
                self.prm_struct_weight > 0
                or self.prm_struct_gate_weight > 0
                or self.prm_rank_weight > 0
            ):
                continue

            x = getattr(module, "last_input", None)
            aggressive = getattr(module, "last_aggressive", None)
            preserve = getattr(module, "last_preserve", None)
            structure_map = getattr(module, "last_structure_map", None)
            if x is None or structure_map is None:
                continue

            if self.prm_rank_weight > 0 and gate is not None:
                route_rank = self._compute_prm_rank_loss(module, gate, x, structure_map)
                if route_rank is not None:
                    route_rank_terms.append(route_rank)

            if not (self.prm_struct_weight > 0 or self.prm_struct_gate_weight > 0):
                continue

            if aggressive is None or preserve is None:
                continue

            x_target = x.detach()
            structure_target = structure_map.detach()
            smooth_target = F.avg_pool2d(x_target, kernel_size=3, stride=1, padding=1)
            detail_target = x_target + structure_target * (x_target - smooth_target)
            preserve_terms.append((structure_target * (preserve - detail_target).abs()).mean())
            aggressive_terms.append(((1.0 - structure_target) * (aggressive - smooth_target).abs()).mean())

            if gate is not None:
                gate_target = getattr(module, "last_gate_target", None)
                if gate_target is None:
                    gate_target = module._match_target_shape(1.0 - structure_target, gate)
                gate_align_terms.append(
                    F.binary_cross_entropy(
                        gate.clamp(1e-6, 1.0 - 1e-6),
                        gate_target.detach().clamp(0.0, 1.0),
                    )
                )

        if gate_entropy_terms:
            losses["gate_entropy"] = sum(gate_entropy_terms) / len(gate_entropy_terms)
        if gate_sparse_terms:
            losses["gate_sparse"] = sum(gate_sparse_terms) / len(gate_sparse_terms)
        if preserve_terms:
            losses["preserve_struct"] = sum(preserve_terms) / len(preserve_terms)
        if aggressive_terms:
            losses["aggressive_smooth"] = sum(aggressive_terms) / len(aggressive_terms)
        if gate_align_terms:
            losses["gate_align"] = sum(gate_align_terms) / len(gate_align_terms)
        if route_rank_terms:
            losses["route_rank"] = sum(route_rank_terms) / len(route_rank_terms)
        return losses

    # ================================================================
    # Parameter management
    # ================================================================

    def _iter_named_trainable_params(self) -> List[Tuple[str, torch.Tensor]]:
        named_params: List[Tuple[str, torch.Tensor]] = []
        for name, p in self.denoiser.named_parameters():
            if p.requires_grad:
                named_params.append((f"denoiser.{name}", p))
        for module_name, module in self.extra_trainable_modules.items():
            for name, p in module.named_parameters():
                if p.requires_grad:
                    named_params.append((f"{module_name}.{name}", p))
        return named_params

    def _refresh_trainable_param_anchor(self):
        self.trainable_param_anchor = {
            name: param.detach().clone()
            for name, param in self._iter_named_trainable_params()
        }

    def _collect_params(self) -> list:
        return [param for _, param in self._iter_named_trainable_params()]

    # ================================================================
    # Core forward / denoise
    # ================================================================

    def _denoise(self, noisy: torch.Tensor) -> torch.Tensor:
        """Denoising forward pass (handles residual output)."""
        output = self.denoiser(noisy)
        if isinstance(output, tuple):
            output = output[0]
        output = output.float()
        if output.abs().mean() < 0.5:
            return torch.clamp(noisy + output, 0, 1)
        return torch.clamp(output, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode."""
        output = self.denoiser(x)
        if isinstance(output, tuple):
            return output[0]
        return output

    # ================================================================
    # Adaptation
    # ================================================================

    def adapt(
        self,
        noisy: torch.Tensor,
        steps: int = 1,
        episodic: bool = True,
        sanity_check: bool = False,
        collect_per_step: bool = False,
        callback=None,
    ) -> torch.Tensor:
        """Test-time adaptation.

        Args:
            noisy: noisy image [B, C, H, W]
            steps: adaptation steps (overridden if use_adaptive_schedule is True)
            episodic: whether to reset per image
            sanity_check: verify adaptation produces output change
            collect_per_step: save per-step inference for convergence diagnostics.
                Results stored in self._per_step_outputs.
            callback: optional callable(step) invoked after each adaptation step.
        """
        if episodic:
            self.reset()

        params = self._collect_params()
        if len(params) == 0:
            print("[DeFT] WARNING: No trainable params collected! Adaptation will have NO effect.")
            with torch.no_grad():
                return self._denoise(noisy)

        actual_steps = steps
        actual_lr = self.lr
        self._estimated_noise = None
        self._is_low_noise = None
        self._noise_category = None

        noise_level = self._estimate_noise_level(noisy)
        self._estimated_noise = noise_level

        # DCI: FiLM confidence suppression on far-OOD domains
        cf = float(torch.sigmoid(torch.tensor((noise_level - 0.005) / 0.003)))
        for m in self.denoiser.modules():
            if hasattr(m, 'film_confidence'):
                m.film_confidence = cf

        if self.descriptor_init_scale:
            g = torch.sigmoid(torch.tensor((noise_level - 0.015) / 0.005)).item()
            for w in self._iter_prm_wrappers():
                if w.aggressive_route is not None:
                    w.aggressive_route.depthwise.weight.data.mul_(g)

        # Build descriptor
        self._cached_descriptor = self._build_degradation_descriptor(noisy).detach()

        # Single-scalar PromptFiLM: cache once
        if self.use_prompt_film and not self.use_prompt_bank:
            prompt_tensor = torch.tensor([[noise_level]], dtype=torch.float32, device=noisy.device)
            self._set_adapter_condition(prompt_tensor, self._cached_descriptor)

        # DCS: adaptive schedule
        if self.use_adaptive_schedule:
            if noise_level < self.adaptive_very_low_threshold:
                if self.adaptive_very_low_steps is not None:
                    actual_steps = self.adaptive_very_low_steps
                    actual_lr = self.lr * self.adaptive_very_low_lr_scale
                else:
                    actual_steps = self.schedule_high_steps
                    actual_lr = self.lr
                self._is_low_noise = False
                self._noise_category = 'very_low'
            else:
                actual_steps, actual_lr = compute_adaptive_budget(
                    noise_level=noise_level,
                    base_lr=self.lr,
                    low_steps=self.schedule_low_steps,
                    high_steps=self.schedule_high_steps,
                    low_lr_scale=self.schedule_low_lr_scale,
                    noise_threshold=self.schedule_noise_threshold,
                )
                if noise_level < self.schedule_noise_threshold:
                    self._is_low_noise = True
                    self._noise_category = 'mid_low'
                else:
                    self._is_low_noise = False
                    self._noise_category = 'high'

        # Force bypass (eval CLI)
        if self.force_bypass:
            for m in self.denoiser.modules():
                if isinstance(m, PolarizedRouteMixture):
                    m.bypass_routing = True
                if self.force_bypass_film and isinstance(m, (FiLMLayer, PromptFiLMLayer)):
                    m.bypass_film = True
            self._set_adapter_condition(None, None)
            self._cached_descriptor = None
            was_training = self.denoiser.training
            if self.force_bypass_eval_mode:
                self.denoiser.eval()
            with torch.no_grad():
                result = self._denoise(noisy)
            if self.force_bypass_eval_mode and was_training:
                self.denoiser.train()
            for m in self.denoiser.modules():
                if isinstance(m, PolarizedRouteMixture):
                    m.bypass_routing = False
                if self.force_bypass_film and isinstance(m, (FiLMLayer, PromptFiLMLayer)):
                    m.bypass_film = False
            self._delta_theta = 0.0
            self._grad_norms = []
            self._actual_steps = 0
            self._actual_lr = 0.0
            self._last_prm_diag = {}
            self._last_prm_aux_losses = {}
            return result

        if sanity_check:
            with torch.no_grad():
                y_before = self._denoise(noisy).clone()

        optimizer = torch.optim.Adam(params, lr=actual_lr)

        self._last_keep_fraction = 1.0
        self._keep_fractions = []
        self._grad_norms = []
        self._actual_steps = actual_steps
        self._actual_lr = actual_lr
        self._per_step_outputs = []

        theta_before = {name: p.clone() for name, p in self.denoiser.named_parameters() if p.requires_grad}

        for step in range(actual_steps):
            self._adapt_step(noisy, optimizer, step, actual_steps)
            if callback is not None:
                callback(step)

            if hasattr(self, '_last_keep_fraction'):
                self._keep_fractions.append(self._last_keep_fraction)

            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            self._grad_norms.append(grad_norm ** 0.5)

            if collect_per_step:
                with torch.no_grad():
                    _step_out = self._denoise(noisy)
                self._per_step_outputs.append(_step_out.cpu())

            if sanity_check and step == 0:
                with torch.no_grad():
                    y_after = self._denoise(noisy)
                    diff = (y_after - y_before).abs().mean().item()
                    print(f"[DeFT Sanity Check] After 1 step: mean|y_after - y_before| = {diff:.6f}")
                    if diff < 1e-8:
                        print("[DeFT Sanity Check] WARNING: Output unchanged! Check if params are being updated.")

        delta_theta = 0.0
        for name, p in self.denoiser.named_parameters():
            if p.requires_grad and name in theta_before:
                delta_theta += (p - theta_before[name]).norm().item() ** 2
        self._delta_theta = delta_theta ** 0.5
        del theta_before

        optimizer.zero_grad(set_to_none=True)
        del optimizer

        with torch.no_grad():
            x_adapt = self._denoise(noisy)
        self._last_prm_diag = self._collect_prm_diagnostics()
        self._last_prm_diag.update(self._last_prm_aux_losses)
        self._clear_prm_runtime_cache()

        return x_adapt

    def _adapt_step(self, noisy: torch.Tensor, optimizer, step: int, total_steps: int):
        """Single adaptation step."""
        optimizer.zero_grad()

        if self.prm_rezero:
            gate_params_set = set(self._rezero_gate_params())
            body_params = [p for p in optimizer.param_groups[0]['params'] if p not in gate_params_set]
        else:
            body_params = None

        total_loss = self._compute_adaptation_loss(noisy, step, total_steps)
        total_loss.backward()

        if self.prm_rezero and body_params:
            g = self._rezero_max_gate()
            for p in body_params:
                if p.grad is not None:
                    p.grad.mul_(g)

        optimizer.step()

    def _compute_adaptation_loss(self, noisy: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        """Compute the full adaptation loss."""
        # DCI: generate prompt from descriptor
        if self.use_prompt_bank and self.prompt_bank is not None and self._cached_descriptor is not None:
            prompt_tensor = self.prompt_bank(self._cached_descriptor)
            self._set_adapter_condition(prompt_tensor, self._cached_descriptor)

        # PRM: structure target for dual-route
        structure_target = None
        if self.prm_struct_use_source:
            structure_target = self._build_structure_anchor(noisy)
        self._set_structure_target(structure_target)

        # Core N2N loss (canonical loss engine)
        loss_ss = self._compute_n2n_loss(noisy)
        total_loss = loss_ss

        # PRM: dual-route auxiliary losses
        prm_aux = self._compute_prm_aux_losses(noisy.device)
        self._last_prm_aux_losses = {
            key: float(val.detach().cpu().item())
            for key, val in prm_aux.items()
        }
        if self.prm_gate_entropy_weight > 0:
            total_loss = total_loss + self.prm_gate_entropy_weight * prm_aux["gate_entropy"]
        if self.prm_gate_sparse_weight > 0:
            total_loss = total_loss + self.prm_gate_sparse_weight * prm_aux["gate_sparse"]
        if self.prm_struct_weight > 0:
            total_loss = total_loss + self.prm_struct_weight * (
                prm_aux["preserve_struct"] + prm_aux["aggressive_smooth"]
            )
        if self.prm_struct_gate_weight > 0:
            total_loss = total_loss + self.prm_struct_gate_weight * prm_aux["gate_align"]
        if self.prm_rank_weight > 0:
            total_loss = total_loss + self.prm_rank_weight * prm_aux["route_rank"]

        # DCS: reliable filter
        if self.use_reliable_filter:
            patch_mask = self._compute_patch_mask(noisy)
            if patch_mask.sum() > 0:
                total_loss = total_loss * patch_mask.mean()
                self._last_keep_fraction = patch_mask.mean().item()
            else:
                total_loss = total_loss * 0.1
                self._last_keep_fraction = 0.0
        else:
            self._last_keep_fraction = 1.0

        return total_loss

    def _compute_n2n_loss(self, noisy: torch.Tensor, patch_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute Neighbor2Neighbor loss."""
        if self.ss_loss is not None:
            return self.ss_loss(noisy, self.denoiser, patch_mask=patch_mask)
        else:
            B, C, H, W = noisy.shape
            mask = torch.zeros_like(noisy)
            mask[:, :, ::2, ::2] = 1
            noisy_sampled = noisy * mask
            noisy_neighbor = noisy * (1 - mask)
            denoised = self._denoise(noisy_sampled + noisy_neighbor)
            loss = self._pixel_loss(denoised * mask, noisy_neighbor * mask)
            return loss

    def _pixel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pixel-wise loss according to loss_fn."""
        diff = pred - target
        if self.loss_fn == 'l2':
            return (diff ** 2).mean()
        elif self.loss_fn == 'l1':
            return diff.abs().mean()
        elif self.loss_fn == 'charbonnier':
            return torch.sqrt(diff ** 2 + 1e-6).mean()
        elif self.loss_fn == 'huber':
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    def _compute_patch_mask(self, noisy: torch.Tensor) -> torch.Tensor:
        """Compute patch-level reliability mask via MAD adaptive outlier detection.

        Returns:
            patch_mask: [B, 1, H, W], 1 = reliable, 0 = unreliable
        """
        B, C, H, W = noisy.shape
        ps = self.reliable_patch_size
        MAD_SCALE = 1.4826

        with torch.no_grad():
            denoised = self._denoise(noisy)

            if self.blur_kernel is not None:
                noisy_high = noisy - self._apply_blur(noisy)
                denoised_high = denoised - self._apply_blur(denoised)
                residual = (denoised_high - noisy_high).abs()
            else:
                residual = (denoised - noisy).abs()

            if C > 1:
                residual = residual.mean(dim=1, keepdim=True)

            num_patches_h = H // ps
            num_patches_w = W // ps

            if num_patches_h == 0 or num_patches_w == 0:
                return torch.ones(B, 1, H, W, device=noisy.device, dtype=noisy.dtype)

            residual_cropped = residual[:, :, :num_patches_h * ps, :num_patches_w * ps]
            patch_residuals = F.avg_pool2d(residual_cropped, kernel_size=ps, stride=ps)
            patch_residuals_flat = patch_residuals.view(B, -1)
            num_patches = patch_residuals_flat.size(1)

            patch_mask_flat = torch.zeros_like(patch_residuals_flat)

            for b in range(B):
                residuals_b = patch_residuals_flat[b]
                med = residuals_b.median()
                abs_deviation = (residuals_b - med).abs()
                mad = abs_deviation.median()
                scaled_mad = max(MAD_SCALE * mad.item(), 1e-8)
                threshold = med + self.reliable_mad_k * scaled_mad
                patch_mask_flat[b] = (residuals_b <= threshold).float()

                min_keep = max(1, int(num_patches * 0.1))
                if patch_mask_flat[b].sum() < min_keep:
                    _, top_k_indices = residuals_b.topk(min_keep, largest=False)
                    patch_mask_flat[b] = 0.0
                    patch_mask_flat[b, top_k_indices] = 1.0

            patch_mask = patch_mask_flat.view(B, 1, num_patches_h, num_patches_w)
            patch_mask = F.interpolate(patch_mask, size=(H, W), mode='nearest')

        return patch_mask
