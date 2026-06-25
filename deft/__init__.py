"""
DeFT: Descriptor-Forked Test-Time Adaptation for medical image denoising.

One descriptor state extracted once per test image, forked to three
projections that operate on the shared degradation coordinate:

  - DCI (DescriptorConditioningInterface): exposes the read-only conditioning signal
  - PRM (PolarizedRouteMixture):           spatial dual-route mixing conditioned on d(y)
  - DCS (DescriptorConditionedScheduler):  maps d(y) to image-specific (K_y, eta_y)

Usage:
    >>> from deft import DeFT
    >>> model = DeFT(checkpoint_path="unet_source.pt")
    >>> denoised = model.adapt(noisy_tensor)  # single-image, source-free, test-time only
"""

from .descriptor import DescriptorState, estimate_noise_level, build_descriptor_tensor
from .dci import FiLMLayer, FiLMWrapper, LoRALayer, LoRAWrapper, PromptFiLMLayer, PromptFiLMWrapper
from .prm import PolarizedRouteMixture
from .dcs import compute_adaptive_budget
from .backbone import DeFTBackbone
from .model import DeFT

__all__ = [
    "DeFT",
    "DescriptorState",
    "PolarizedRouteMixture",
    "compute_adaptive_budget",
    "DeFTBackbone",
    "estimate_noise_level",
    "build_descriptor_tensor",
    "FiLMLayer",
    "FiLMWrapper",
    "LoRALayer",
    "LoRAWrapper",
    "PromptFiLMLayer",
    "PromptFiLMWrapper",
]
