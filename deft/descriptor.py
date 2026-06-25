"""DeFT descriptor — the shared per-image degradation state.

A single DescriptorState is extracted once per test image and consumed
by all three projections: DCI (conditioning), PRM (spatial routing),
and DCS (update scheduling).
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class DescriptorState:
    """Shared degradation state extracted once per test image.

    Six components capture complementary aspects of the degradation.
    Component order matches the canonical implementation used to produce
    all paper results (Eq. 5 groups them as [noise/intensity | structure | source prior]):

        sigma_mad:  MAD-based noise estimate (DCS primary input), index 0
        sigma_std:  foreground intensity spread, index 1
        g_grad:     mean gradient magnitude (anatomical edge density), index 2
        e_hf:       mean high-frequency energy (Laplacian response), index 3
        r_src:      source-prior response residual ‖f(y)−y‖/(‖y‖+ε), index 4
        mu_y:       global intensity level, index 5
    """

    def to_tensor(self, device=None, dtype=None):
        return torch.tensor(
            [self.sigma_mad, self.sigma_std, self.g_grad,
             self.e_hf, self.r_src, self.mu_y],
            device=device, dtype=dtype,
        )

    def as_vector(self):
        return (self.sigma_mad, self.sigma_std, self.g_grad,
                self.e_hf, self.r_src, self.mu_y)


def estimate_noise_level(noisy: torch.Tensor) -> float:
    """Estimate image noise level using Laplacian-MAD.

    Applies a 4-neighbour Laplacian kernel to extract high-frequency
    components, then computes MAD / 0.6745 as the noise standard
    deviation estimate, following Donoho & Johnstone (1994, Biometrika).

    Args:
        noisy: single noisy image, shape (1, 1, H, W), values in [0, 1].

    Returns:
        Estimated noise level as a scalar float.
    """
    with torch.no_grad():
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=noisy.dtype, device=noisy.device,
        ).view(1, 1, 3, 3)

        high_freq = F.conv2d(noisy, laplacian, padding=1)
        median = high_freq.abs().median()
        mad = (high_freq.abs() - median).abs().median()
        noise_std = mad / 0.6745
        noise_level = noise_std.item() / 4.0
        return noise_level


def build_descriptor_tensor(
    noisy: torch.Tensor,
    noise_level: float,
    source_out: torch.Tensor,
) -> torch.Tensor:
    """Build a 6-dim degradation descriptor tensor from image statistics.

    Args:
        noisy: single noisy image, shape (1, 1, H, W).
        noise_level: pre-computed MAD noise estimate from estimate_noise_level().
        source_out: backbone denoiser output on the noisy input.

    Returns:
        Descriptor tensor of shape (1, 6), clamped to [0, 1].
    """
    b = noisy.size(0)
    x_gray = noisy.mean(dim=1, keepdim=True)

    # Intensity statistics
    std = x_gray.flatten(1).std(dim=1, unbiased=False)
    mean_intensity = x_gray.flatten(1).mean(dim=1)

    # Gradient magnitude (anatomical edge density)
    grad_x = torch.abs(x_gray[:, :, :, 1:] - x_gray[:, :, :, :-1]).mean(dim=(1, 2, 3))
    grad_y = torch.abs(x_gray[:, :, 1:, :] - x_gray[:, :, :-1, :]).mean(dim=(1, 2, 3))
    grad_mag = 0.5 * (grad_x + grad_y)

    # High-frequency energy (Laplacian response)
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=noisy.dtype, device=noisy.device,
    ).view(1, 1, 3, 3)
    hf = F.conv2d(x_gray, lap_kernel, padding=1).abs().mean(dim=(1, 2, 3))

    # Source-prior response residual: r_src = ‖f(y)-y‖ / (‖y‖ + ε)
    # Following Eq. (5) in the manuscript.
    diff = (noisy - source_out).reshape(b, -1)
    source_norm = diff.norm(p=2, dim=1)
    noisy_norm = noisy.reshape(b, -1).norm(p=2, dim=1)
    source_resid = source_norm / (noisy_norm + 1e-8)

    # Noise level as a tensor
    nl = torch.full((b,), noise_level, dtype=noisy.dtype, device=noisy.device)

    descriptor = torch.stack(
        [nl, std, grad_mag, hf, source_resid, mean_intensity], dim=1,
    )
    descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=1.0, neginf=0.0)
    descriptor = descriptor.clamp(0.0, 1.0)
    return descriptor
