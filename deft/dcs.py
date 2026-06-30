"""DeFT DCS — Descriptor-Conditioned Scheduler.

Maps the shared descriptor's noise-magnitude signal to an image-specific
adaptation budget (K, η). This is the third projection of the descriptor
state: DCI exposes the condition, PRM routes in space, DCS sets the
temporal update schedule.
"""

from typing import Tuple


def compute_adaptive_budget(
    noise_level: float,
    *,
    base_lr: float = 2e-4,
    low_steps: int = 3,
    high_steps: int = 10,
    low_lr_scale: float = 0.5,
    noise_threshold: float = 0.05,
) -> Tuple[int, float]:
    """Compute per-image adaptation budget from estimated noise level.

    Two effective tiers derived from the source-domain noise range:
      σ_mad < 0.05  →  conservative (K=3, η=1e-4)
      σ_mad ≥ 0.05  →  full adaptation (K=10, η=2e-4)

    The discretisation deliberately tolerates estimator noise under
    Rician and mixed-noise conditions — tier boundaries absorb
    estimation error that a continuous σ → (K,η) regression would
    amplify.

    Args:
        noise_level: MAD-based noise estimate from estimate_noise_level().
        base_lr: learning rate for the high-noise tier (default 2e-4).
        low_steps: step count for the moderate-noise tier (default 3).
        high_steps: step count for the heavy-noise tier (default 10).
        low_lr_scale: scale factor for moderate-noise lr (default 0.5).
        noise_threshold: boundary between moderate and heavy noise (default 0.05).

    Returns:
        (steps, learning_rate) tuple.
    """
    if noise_level < noise_threshold:
        return low_steps, base_lr * low_lr_scale
    else:
        return high_steps, base_lr
