"""DeFT conditioning layers — lightweight trainable modulation inserted
into the frozen backbone decoder to serve the DCI (Descriptor Conditioning
Interface) projection.

Contains:
    FiLMLayer, FiLMWrapper          — Feature-wise Linear Modulation (FiLM, AAAI 2018)
    LoRALayer, LoRAWrapper          — Low-Rank Adaptation (LoRA, ICLR 2022)
    PromptFiLMLayer, PromptFiLMWrapper  — Prompt-conditioned FiLM for episodic adaptation

All layers are initialized near identity (γ≈1, β≈0) so they do not disturb
the source-model denoising prior at the start of adaptation.
"""

import math
import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h' = γ ⊙ h + β.

    Initialized to identity (γ=1, β=0). A per-channel affine transform
    with only 2×C trainable parameters. Serves as the basic conditioning
    primitive for DCI.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, 'bypass_film', False):
            return x
        cf = getattr(self, 'film_confidence', 1.0)
        if cf >= 1.0:
            return self.gamma * x + self.beta
        return x + cf * (self.gamma * x + self.beta - x)


class FiLMWrapper(nn.Module):
    """Insert a FiLM layer after an existing module."""
    def __init__(self, module: nn.Module, num_channels: int):
        super().__init__()
        self.module = module
        self.film = FiLMLayer(num_channels)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.film(out[0]),) + out[1:]
        return self.film(out)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation: h' = h + s·(B @ A @ h).

    Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language
    Models", ICLR 2022.
    """
    def __init__(self, in_channels: int, out_channels: int = None,
                 rank: int = 8, scale: float = 1.0, use_rezero: bool = False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.rank = rank
        if use_rezero:
            self.scale = nn.Parameter(torch.zeros(1))
        else:
            self.scale = scale
        self.A = nn.Linear(in_channels, rank, bias=False)
        self.B = nn.Linear(rank, out_channels, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        delta = self.B(self.A(x_flat))
        delta = delta.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return x + self.scale * delta


class LoRAWrapper(nn.Module):
    """Insert a LoRA layer after an existing module."""
    def __init__(self, module: nn.Module, num_channels: int,
                 rank: int = 8, use_rezero: bool = False):
        super().__init__()
        self.module = module
        self.lora = LoRALayer(num_channels, num_channels,
                              rank=rank, use_rezero=use_rezero)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.lora(out[0]),) + out[1:]
        return self.lora(out)


class PromptFiLMLayer(nn.Module):
    """Prompt-conditioned FiLM: uses a learned prompt vector to produce
    per-channel γ, β through a small MLP. Zero-initialized so the layer
    starts as identity and gradually activates through gradient updates.

    This is the primary conditioning primitive for the episodic prompt
    state in DCI.
    """
    def __init__(self, num_channels: int, prior_dim: int = 1,
                 hidden_dim: int = 16):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_gamma = nn.Linear(hidden_dim, num_channels)
        self.fc_beta = nn.Linear(hidden_dim, num_channels)
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: torch.Tensor,
                prompt_prior: torch.Tensor = None) -> torch.Tensor:
        if getattr(self, 'bypass_film', False):
            return x
        if prompt_prior is None:
            return x
        hidden = self.mlp(prompt_prior)
        gamma = self.fc_gamma(hidden).view(-1, self.num_channels, 1, 1)
        beta = self.fc_beta(hidden).view(-1, self.num_channels, 1, 1)
        cf = getattr(self, 'film_confidence', 1.0)
        if cf >= 1.0:
            return (1.0 + gamma) * x + beta
        return x + cf * ((1.0 + gamma) * x + beta - x)


class PromptFiLMWrapper(nn.Module):
    """Insert a PromptFiLM layer after an existing module."""
    def __init__(self, module: nn.Module, num_channels: int,
                 prior_dim: int = 1):
        super().__init__()
        self.module = module
        self.film = PromptFiLMLayer(num_channels, prior_dim=prior_dim)
        self.current_prompt = None

    def set_condition(self, prompt_prior: torch.Tensor = None,
                      descriptor: torch.Tensor = None):
        self.current_prompt = prompt_prior

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.film(out[0], self.current_prompt),) + out[1:]
        return self.film(out, self.current_prompt)
