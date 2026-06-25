"""
Neighbor2Neighbor self-supervised loss for DeFT.

Reference: Huang et al., "Neighbor2Neighbor: Self-Supervised Denoising from Single
Noisy Images", CVPR 2021.  Implementation adapted from the official LAN repository
(https://github.com/chjinny/LAN).

This module provides a standalone Neighbor2NeighborLoss class that wraps a denoiser
callable and computes the self-supervised N2N objective with optional patch-level
masking and Lambda annealing.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_nbr2nbr_seed_counter: int = 0


def _get_nbr2nbr_generator(device: torch.device) -> torch.Generator:
    """Return a deterministic RNG seeded by a monotonic counter."""
    global _nbr2nbr_seed_counter
    _nbr2nbr_seed_counter += 1
    g = torch.Generator(device=device)
    g.manual_seed(_nbr2nbr_seed_counter)
    return g


def _space_to_depth(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Rearrange spatial blocks into channel depth (inverse of Depth2Space).

    Equivalent to ``F.pixel_unshuffle`` for square blocks.
    """
    n, c, h, w = x.shape
    unfolded = F.unfold(x, block_size, stride=block_size)
    return unfolded.view(n, c * block_size ** 2, h // block_size, w // block_size)


def _generate_mask_pair(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random complementary pair of 2×2 neighbour masks.

    Each 2×2 cell is randomly split into two sub-samples.  The returned
    masks select complementary pixels so that together they cover every
    pixel exactly once.

    Args:
        img: Input tensor [N, C, H, W].

    Returns:
        mask1, mask2: Boolean tensors of shape (N*H/2*W/2*4,), complementary.
    """
    n, c, h, w = img.shape
    device = img.device

    mask1 = torch.zeros(n * h // 2 * w // 2 * 4, dtype=torch.bool, device=device)
    mask2 = torch.zeros(n * h // 2 * w // 2 * 4, dtype=torch.bool, device=device)

    # 8 neighbour-pair patterns within a 2×2 block
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=device,
    )

    rd_idx = torch.randint(
        low=0,
        high=8,
        size=(n * h // 2 * w // 2,),
        generator=_get_nbr2nbr_generator(device),
        device=device,
    )
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(
        start=0,
        end=n * h // 2 * w // 2 * 4,
        step=4,
        dtype=torch.int64,
        device=device,
    ).reshape(-1, 1)

    mask1[rd_pair_idx[:, 0]] = True
    mask2[rd_pair_idx[:, 1]] = True

    return mask1, mask2


def _generate_subimages(
    img: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Sample subimage pixels selected by a boolean mask.

    Uses ``_space_to_depth`` internally to rearrange 2×2 blocks before
    applying the flat mask.

    Args:
        img:  Input tensor [N, C, H, W].
        mask: Boolean mask of shape (N*H/2*W/2*4,).

    Returns:
        Subimage tensor [N, C, H/2, W/2].
    """
    n, c, h, w = img.shape
    device = img.device
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, device=device)

    for i in range(c):
        img_per_channel = _space_to_depth(img[:, i : i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i : i + 1, :, :] = (
            img_per_channel[mask]
            .reshape(n, h // 2, w // 2, 1)
            .permute(0, 3, 1, 2)
        )

    return subimage


# ---------------------------------------------------------------------------
# Neighbor2Neighbor loss
# ---------------------------------------------------------------------------


class Neighbor2NeighborLoss(nn.Module):
    """Neighbor2Neighbor self-supervised loss (canonical DeFT engine).

    Core idea:
      - Randomly sample neighbour pixel pairs from the noisy image.
      - Predict one neighbour from the other via the denoiser.
      - Regularise predictions toward the clean expectation with Lambda annealing.

    Loss functions supported:
      - ``'l2'`` — MSE.
      - ``'l1'`` — MAE.
      - ``'charbonnier'`` — sqrt(x² + ε²) (default; better detail retention).

    Parameters:
        loss_fn: One of ``{'l2', 'l1', 'charbonnier'}``.
        eps: Epsilon for the Charbonnier penalty.
    """

    def __init__(self, loss_fn: str = "charbonnier", eps: float = 1e-3) -> None:
        super().__init__()
        self.loss_fn = loss_fn.lower()
        self.eps = eps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        diff: torch.Tensor,
        reduction: str = "mean",
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Pixel-wise loss from a difference tensor.

        Args:
            diff:       Difference tensor [B, C, H, W].
            reduction:  ``'mean'`` returns a scalar; ``'none'`` returns the
                        per-pixel loss tensor of the same shape as ``diff``.
            step:       (unused for canonical variants; kept for API compatibility)
            total_steps:(unused for canonical variants; kept for API compatibility)

        Returns:
            Loss tensor or scalar.
        """
        if self.loss_fn == "l2":
            loss = diff ** 2
        elif self.loss_fn == "l1":
            loss = diff.abs()
        elif self.loss_fn == "charbonnier":
            loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        else:
            raise ValueError(f"Unknown loss_fn: {self.loss_fn}")

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy: torch.Tensor,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        step: int = 0,
        total_steps: int = 20,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Neighbor2Neighbor loss.

        Args:
            noisy:       Noisy image [B, C, H, W].
            denoiser:    Denoising callable that maps ``(Tensor) -> Tensor``.
                         If the output is a tuple, the first element is used.
                         If the residual mean is small (< 0.5), the output is
                         treated as a residual added to ``noisy``; otherwise
                         it is used directly as the denoised image.
            step:        Current adaptation step (for Lambda annealing).
            total_steps: Total adaptation steps.
            patch_mask:  Optional patch-level mask [B, 1, H, W] where 1
                         indicates a reliable pixel (included in loss).

        Returns:
            Scalar loss.
        """
        # 1 — generate random neighbour mask pair
        mask1, mask2 = _generate_mask_pair(noisy)

        # 2 — produce subimages
        noisy_sub1 = _generate_subimages(noisy, mask1)
        noisy_sub2 = _generate_subimages(noisy, mask2)

        # 3 — denoise the full image (no-grad reference)
        with torch.no_grad():
            noisy_denoised = denoiser(noisy)
            if isinstance(noisy_denoised, tuple):
                noisy_denoised = noisy_denoised[0]
            if noisy_denoised.abs().mean() < 0.5:
                noisy_denoised = torch.clamp(noisy + noisy_denoised, 0, 1)
            else:
                noisy_denoised = torch.clamp(noisy_denoised, 0, 1)

        # 4 — subsample the denoised reference
        noisy_sub1_denoised = _generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = _generate_subimages(noisy_denoised, mask2)

        # 5 — predict sub2 from sub1
        noisy_output = denoiser(noisy_sub1)
        if isinstance(noisy_output, tuple):
            noisy_output = noisy_output[0]
        if noisy_output.abs().mean() < 0.5:
            noisy_output = torch.clamp(noisy_sub1 + noisy_output, 0, 1)
        else:
            noisy_output = torch.clamp(noisy_output, 0, 1)

        noisy_target = noisy_sub2

        # 6 — Lambda annealing: 0 → 0.1 over total_steps
        Lambda = (step / max(total_steps, 1)) * 0.1

        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        # 7 — loss aggregation (with optional patch masking)
        if patch_mask is not None:
            sub_mask = (
                F.avg_pool2d(patch_mask.float(), kernel_size=2, stride=2)
                .clamp(0.0, 1.0)
            )

            loss1_map = self._compute_loss(diff, reduction="none")
            loss2_map = Lambda * self._compute_loss(
                diff - exp_diff, reduction="none"
            )
            loss_map = loss1_map + loss2_map

            if sub_mask.size(1) == 1 and loss_map.size(1) > 1:
                sub_mask = sub_mask.expand_as(loss_map)

            masked_loss = loss_map * sub_mask
            num_valid = sub_mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / num_valid
        else:
            loss1 = self._compute_loss(diff)
            loss2 = Lambda * self._compute_loss(diff - exp_diff)
            loss = loss1 + loss2

        return loss
