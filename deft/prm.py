"""
PRM — Polarized Route Mixture for DeFT.

Two structurally complementary adapter routes (aggressive denoising and structure
preservation) blended by a descriptor-conditioned spatial gate. This is the spatial
projection of the descriptor state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from deft.dci import PromptFiLMLayer


def _normalized_edge_map(x: torch.Tensor) -> torch.Tensor:
    """Build a normalized detail heatmap from feature tensors."""
    x_gray = x.mean(dim=1, keepdim=True)
    grad_x = torch.zeros_like(x_gray)
    grad_y = torch.zeros_like(x_gray)
    grad_x[:, :, :, :-1] = torch.abs(x_gray[:, :, :, 1:] - x_gray[:, :, :, :-1])
    grad_y[:, :, :-1, :] = torch.abs(x_gray[:, :, 1:, :] - x_gray[:, :, :-1, :])
    edge = 0.5 * (grad_x + grad_y)
    b = edge.size(0)
    flat = edge.view(b, -1)
    e_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
    e_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (edge - e_min) / (e_max - e_min + 1e-8)


def _noise_proxy_map(x: torch.Tensor) -> torch.Tensor:
    """Build a normalized high-frequency proxy from feature tensors."""
    x_gray = x.mean(dim=1, keepdim=True)
    x_smooth = F.avg_pool2d(x_gray, kernel_size=3, stride=1, padding=1)
    noise = (x_gray - x_smooth).abs()
    b = noise.size(0)
    flat = noise.view(b, -1)
    n_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
    n_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (noise - n_min) / (n_max - n_min + 1e-8)


class PromptBank(nn.Module):
    """
    PromptIR / PromptRestorer inspired degradation prompt bank.

    A compact descriptor vector selects and mixes a small bank of prompt tokens,
    yielding a richer prompt than a single-scalar noise prior.
    """

    def __init__(
        self,
        descriptor_dim: int = 6,
        prompt_dim: int = 16,
        prompt_len: int = 4,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_dim = prompt_dim
        self.prompt_bank = nn.Parameter(torch.randn(prompt_len, prompt_dim) * 0.02)
        self.router = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, prompt_len),
        )
        self.proj = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim, prompt_dim),
        )

        # Use small non-zero init so different descriptor values produce
        # distinguishable logits from step 0.  zeros_ made softmax collapse
        # to uniform mixture regardless of the input (within-bank diff == 0),
        # rendering cross-experiment ablation comparisons meaningless.
        # std=0.1 gives visible weight spread (~0.23-0.29 vs 0.25 uniform) at
        # step 0 while keeping the mixture nearly uniform enough not to
        # destabilize the identity-preserving FiLM init downstream.
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        logits = self.router(descriptor)
        weights = torch.softmax(logits, dim=1)
        prompt = weights @ self.prompt_bank
        return self.proj(prompt)


class PromptConditionMLP(nn.Module):
    """Small prompt-conditioned projector used by the dual-route adapters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        zero_init: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AggressiveRoute(nn.Module):
    """
    Prompt-conditioned aggressive denoising route.

    The route starts from an identity-preserving PromptFiLM branch and adds a
    zero-initialized depthwise residual path so adaptation only becomes stronger
    after optimization proves it useful.
    """

    def __init__(
        self,
        num_channels: int,
        prompt_dim: int,
        hidden_dim: int = 32,
        res_scale: float = 0.25,
        variant: str = "base",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.res_scale = res_scale
        self.variant = variant
        self.prompt_film = PromptFiLMLayer(num_channels, prior_dim=prompt_dim, hidden_dim=hidden_dim)
        self.depthwise = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            groups=num_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
        self.scale_head = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="linear")
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

        self.context_pool = None
        self.context_depthwise = None
        self.context_pointwise = None
        self.mix_head = None
        if self.variant == "hetero":
            self.context_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
            self.context_depthwise = nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=5,
                padding=2,
                groups=num_channels,
                bias=False,
            )
            self.context_pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
            self.mix_head = PromptConditionMLP(
                input_dim=prompt_dim,
                output_dim=num_channels,
                hidden_dim=hidden_dim,
                zero_init=True,
            )
            nn.init.kaiming_normal_(self.context_depthwise.weight, nonlinearity="linear")
            nn.init.zeros_(self.context_pointwise.weight)
            nn.init.zeros_(self.context_pointwise.bias)

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor | None,
        structure_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.prompt_film(x, prompt)
        local_delta = self.pointwise(F.gelu(self.depthwise(x)))
        delta = local_delta
        if self.variant == "hetero":
            context_delta = self.context_pointwise(F.gelu(self.context_depthwise(self.context_pool(x))))
            if prompt is None:
                mix = 0.5
            else:
                mix = torch.sigmoid(self.mix_head(prompt)).view(-1, self.num_channels, 1, 1)
            delta = mix * local_delta + (1.0 - mix) * context_delta
        if prompt is None:
            return base
        scale = torch.tanh(self.scale_head(prompt)).view(-1, self.num_channels, 1, 1)
        return base + self.res_scale * scale * delta


class StructureRoute(nn.Module):
    """
    Structure-preserving route.

    ``film`` keeps the old PromptFiLM behaviour.
    ``freq`` borrows the AnyIR intuition and rebalances low/high-frequency
    components so high-frequency anatomical details are less likely to collapse.
    """

    def __init__(
        self,
        num_channels: int,
        prompt_dim: int,
        mode: str = "film",
        hidden_dim: int = 32,
        freq_strength: float = 0.1,
        variant: str = "auto",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.mode = mode
        self.freq_strength = freq_strength
        self.variant = variant
        self.prompt_film = PromptFiLMLayer(num_channels, prior_dim=prompt_dim, hidden_dim=hidden_dim)
        self.low_gain = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.high_gain = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.structure_conv = None
        self.structure_head = None
        if self.variant == "struct":
            self.structure_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
            self.structure_head = PromptConditionMLP(
                input_dim=prompt_dim,
                output_dim=1,
                hidden_dim=hidden_dim,
                zero_init=True,
            )
            nn.init.zeros_(self.structure_conv.weight)
            nn.init.zeros_(self.structure_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor | None,
        structure_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "film":
            base = self.prompt_film(x, prompt)
        else:
            low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            high = x - low
            low_mod = self.prompt_film(low, prompt)
            if prompt is None:
                base = low_mod + high
            else:
                low_gate = 1.0 + self.freq_strength * torch.tanh(self.low_gain(prompt)).view(
                    -1, self.num_channels, 1, 1
                )
                high_gate = 1.0 + self.freq_strength * torch.tanh(self.high_gain(prompt)).view(
                    -1, self.num_channels, 1, 1
                )
                base = low_gate * low_mod + high_gate * high

        if self.variant != "struct":
            return base

        if structure_target is not None:
            edge = structure_target
        else:
            edge = _normalized_edge_map(x)
        if edge.shape[-2:] != x.shape[-2:]:
            edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if prompt is None:
            prompt_keep = 0.5
        else:
            prompt_keep = torch.sigmoid(self.structure_head(prompt)).view(-1, 1, 1, 1)
        edge_keep = torch.sigmoid(self.structure_conv(edge))
        keep = prompt_keep * edge_keep * edge
        return base + keep * (x - base)


class SpatialGate(nn.Module):
    """
    Descriptor-conditioned spatial gate.

    ``scalar`` gives one mixture ratio per image.
    ``channel`` gives one mixture ratio per channel, which is closer to channel
    routing while remaining much lighter than a full sparse MoE.
    ``spatial`` produces a per-pixel mixture map conditioned on both the
    descriptor-prompt pair and the current feature activation.
    """

    def __init__(
        self,
        descriptor_dim: int,
        prompt_dim: int,
        num_channels: int,
        gate_type: str = "scalar",
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.prompt_dim = prompt_dim
        self.num_channels = num_channels
        self.gate_type = gate_type
        self.output_dim = 1 if gate_type == "scalar" else num_channels
        self.temperature = max(float(temperature), 1e-4)
        self.projector = PromptConditionMLP(
            input_dim=descriptor_dim + prompt_dim,
            output_dim=self.output_dim,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.feature_projector = None
        self.feature_out = None
        self.condition_projector = None
        if self.gate_type == "spatial":
            self.feature_projector = nn.Conv2d(num_channels, hidden_dim, kernel_size=1, bias=True)
            self.feature_out = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
            self.condition_projector = PromptConditionMLP(
                input_dim=descriptor_dim + prompt_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                zero_init=True,
            )
            nn.init.zeros_(self.feature_out.weight)
            nn.init.zeros_(self.feature_out.bias)

    def forward(
        self,
        descriptor: torch.Tensor | None,
        prompt: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if descriptor is None:
            descriptor = torch.zeros(batch_size, self.descriptor_dim, device=device, dtype=dtype)
        elif descriptor.size(0) != batch_size:
            descriptor = descriptor.expand(batch_size, -1)
        if prompt is None:
            prompt = torch.zeros(batch_size, self.prompt_dim, device=device, dtype=dtype)
        elif prompt.size(0) != batch_size:
            prompt = prompt.expand(batch_size, -1)
        cond = torch.cat([descriptor, prompt], dim=1)
        if self.gate_type == "spatial":
            if feature is None:
                raise ValueError("feature must be provided for spatial gate")
            cond_bias = self.condition_projector(cond).view(batch_size, -1, 1, 1)
            hidden = F.gelu(self.feature_projector(feature) + cond_bias)
            logits = self.feature_out(hidden)

            # Inject a detail-aware prior so spatial gating is meaningfully
            # non-uniform even before the tiny test-time optimization converges.
            feat_gray = feature.mean(dim=1, keepdim=True)
            feat_smooth = F.avg_pool2d(feat_gray, kernel_size=3, stride=1, padding=1)
            detail = (feat_gray - feat_smooth).abs()
            flat = detail.view(batch_size, -1)
            d_min = flat.min(dim=1)[0].view(batch_size, 1, 1, 1)
            d_max = flat.max(dim=1)[0].view(batch_size, 1, 1, 1)
            detail = (detail - d_min) / (d_max - d_min + 1e-8)
            smooth_prior = 1.0 - detail
            logits = logits + 2.0 * (smooth_prior - 0.5)

            logits = logits / self.temperature
            return torch.sigmoid(logits)
        gate = torch.sigmoid(self.projector(cond) / self.temperature)
        return gate.view(batch_size, self.output_dim, 1, 1)


class AdaptiveFreqModulator(nn.Module):
    """AdaIR/AnyIR inspired low/high-frequency modulation block."""

    def __init__(
        self,
        num_channels: int,
        prompt_dim: int,
        hidden_dim: int = 32,
        strength: float = 0.1,
    ):
        super().__init__()
        hidden = max(1, num_channels // 8)
        self.strength = strength
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(num_channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, num_channels, kernel_size=1, bias=False),
        )
        self.low_head = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.high_head = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.mix_in = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.mix_out = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.mix_out.bias)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low

        high_mag = high.abs()
        spatial_input = torch.cat(
            [high_mag.amax(dim=1, keepdim=True), high_mag.mean(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial_gate(spatial_input))

        pooled = F.adaptive_avg_pool2d(low, 1) + F.adaptive_max_pool2d(low, 1)
        channel_gate = torch.sigmoid(self.channel_mlp(pooled))

        if prompt is None:
            low_gain = torch.ones_like(channel_gate)
            high_gain = torch.ones_like(channel_gate)
        else:
            low_gain = 1.0 + self.strength * torch.tanh(self.low_head(prompt)).view(-1, x.size(1), 1, 1)
            high_gain = 1.0 + self.strength * torch.tanh(self.high_head(prompt)).view(-1, x.size(1), 1, 1)

        low_mod = low * low_gain * (1.0 + spatial_gate)
        high_mod = high * high_gain * (1.0 + channel_gate)
        fused = self.mix_out(F.gelu(self.mix_in(low_mod + high_mod)))
        return x + self.strength * fused


class MaskGuidedPreserver(nn.Module):
    """RAM inspired preserve-mask generator over pooled spatial cells."""

    def __init__(
        self,
        prompt_dim: int,
        hidden_dim: int = 32,
        keep_ratio: float = 0.25,
        grid_size: int = 8,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.grid_size = grid_size
        self.score_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        self.prompt_bias = PromptConditionMLP(
            input_dim=prompt_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.zeros_(self.score_net[-1].bias)

    def forward(
        self,
        structure_map: torch.Tensor,
        noise_proxy: torch.Tensor,
        prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heuristic = 0.7 * structure_map + 0.3 * (1.0 - noise_proxy)
        logits = self.score_net(torch.cat([structure_map, 1.0 - noise_proxy], dim=1))
        if prompt is not None:
            logits = logits + self.prompt_bias(prompt).view(-1, 1, 1, 1)
        score = (heuristic + torch.sigmoid(logits)).clamp(0.0, 1.0)

        grid = int(min(self.grid_size, score.shape[-2], score.shape[-1]))
        if grid < 1:
            return torch.ones_like(score), score

        score_grid = F.adaptive_avg_pool2d(score, (grid, grid)).flatten(1)
        num_cells = score_grid.size(1)
        keep = min(num_cells, max(1, int(round(self.keep_ratio * num_cells))))
        keep_idx = score_grid.topk(keep, dim=1, largest=True).indices

        mask_grid = score_grid.new_zeros(score_grid.shape)
        mask_grid.scatter_(1, keep_idx, 1.0)
        mask = mask_grid.view(-1, 1, grid, grid)
        mask = F.interpolate(mask, size=score.shape[-2:], mode="nearest")
        return mask.clamp(0.0, 1.0), score


class GatedDegradationPropagation(nn.Module):
    """PromptRestorer inspired prompt-propagation gate."""

    def __init__(
        self,
        num_channels: int,
        prompt_dim: int,
        descriptor_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.descriptor_dim = descriptor_dim
        self.global_head = PromptConditionMLP(
            input_dim=prompt_dim + descriptor_dim,
            output_dim=num_channels,
            hidden_dim=hidden_dim,
            zero_init=True,
        )
        self.local_net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.local_net[-1].weight)
        nn.init.zeros_(self.local_net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor | None = None,
        descriptor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if descriptor is None:
            descriptor = torch.zeros(x.size(0), self.descriptor_dim, device=x.device, dtype=x.dtype)
        if prompt is None:
            prompt = torch.zeros(x.size(0), self.prompt_dim, device=x.device, dtype=x.dtype)
        cond = torch.cat([descriptor, prompt], dim=1)
        global_gate = torch.sigmoid(self.global_head(cond)).view(-1, x.size(1), 1, 1)

        feat_gray = x.mean(dim=1, keepdim=True)
        noise_proxy = _noise_proxy_map(x)
        local_logits = self.local_net(torch.cat([feat_gray, noise_proxy], dim=1))
        local_gate = torch.sigmoid(local_logits + 2.0 * (noise_proxy - 0.5))
        return global_gate * (0.5 + 0.5 * local_gate)


class HardTopKRouter(nn.Module):
    """Deterministic top-k hard router over pooled spatial cells."""

    def __init__(
        self,
        grid_size: int = 4,
        topk: int = 2,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.topk = topk

    def forward(
        self,
        soft_gate: torch.Tensor,
        structure_map: torch.Tensor,
        x: torch.Tensor,
        match_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_map = match_fn(soft_gate.detach(), x[:, :1]).mean(dim=1, keepdim=True)
        structure_map = match_fn(structure_map.detach(), x[:, :1]).mean(dim=1, keepdim=True)
        noise_proxy = _noise_proxy_map(x.detach())
        score = gate_map * (1.0 - structure_map) * (0.5 + 0.5 * noise_proxy)

        grid = int(min(self.grid_size, score.shape[-2], score.shape[-1]))
        if grid < 1:
            return gate_map.clamp(0.0, 1.0), score

        score_grid = F.adaptive_avg_pool2d(score, (grid, grid)).flatten(1)
        num_cells = score_grid.size(1)
        topk = min(num_cells, max(1, self.topk))
        topk_idx = score_grid.topk(topk, dim=1, largest=True).indices

        mask_grid = score_grid.new_zeros(score_grid.shape)
        mask_grid.scatter_(1, topk_idx, 1.0)
        hard_mask = mask_grid.view(-1, 1, grid, grid)
        hard_mask = F.interpolate(hard_mask, size=score.shape[-2:], mode="nearest")
        return hard_mask.clamp(0.0, 1.0), score


class PolarizedRouteMixture(nn.Module):
    """
    Polarized Route Mixture — spatial projection of the DeFT descriptor state.

    Two structurally complementary adapter routes (aggressive denoising and
    structure preservation) are blended by a descriptor-conditioned spatial gate.
    The gate is a function of both the descriptor-prompt pair and the current
    feature activation, producing a per-pixel mixture map that adapts the
    denoising intensity across the image.

    Modes
    -----
    film
        PromptRestorer / PromptIR style two-branch PromptFiLM baseline.
    amir
        Channel-routing style gating inspired by AMIR.
    anyir
        Scalar-gated spatial-frequency preserve route inspired by AnyIR.
    hybrid
        Channel-gated + spatial-frequency preserve route.
    """

    def __init__(
        self,
        module: nn.Module,
        num_channels: int,
        prompt_dim: int,
        descriptor_dim: int = 6,
        route_mode: str = "hybrid",
        route_policy: str = "auto",
        res_scale: float = 0.25,
        freq_strength: float = 0.1,
        hidden_dim: int = 32,
        spatial_gate: bool = False,
        preserve_variant: str = "auto",
        aggressive_variant: str = "base",
        gate_temperature: float = 1.0,
        hard_route: bool = False,
        hard_grid: int = 4,
        hard_topk: int = 2,
        adaptive_freq_mod: bool = False,
        adaptive_freq_strength: float = 0.1,
        mask_guided_preserve: bool = False,
        mask_keep_ratio: float = 0.25,
        gated_degprop: bool = False,
        degprop_strength: float = 1.0,
        use_degfield: bool = False,
        degfield_strength: float = 0.5,
        prm_disabled: bool = False,
        single_route: Optional[str] = None,
    ):
        super().__init__()
        self.module = module
        self.num_channels = num_channels
        self.prompt_dim = prompt_dim
        self.descriptor_dim = descriptor_dim
        self.route_mode = route_mode
        self.route_policy = route_policy
        self.spatial_gate = spatial_gate
        self.preserve_variant = preserve_variant
        self.aggressive_variant = aggressive_variant
        self.gate_temperature = gate_temperature
        self.prm_disabled = prm_disabled
        self.single_route = single_route
        self.hard_route = hard_route
        self.mask_guided_preserve = mask_guided_preserve
        self.gated_degprop = gated_degprop
        self.degprop_strength = degprop_strength
        self.use_degfield = use_degfield
        self.degfield_strength = degfield_strength
        self.current_prompt = None
        self.current_descriptor = None
        self.current_structure_target = None
        self.last_gate_mean = None
        self.last_gate_std = None
        self.last_gate_lowfrac = None
        self.last_gate_highfrac = None
        self.last_aggressive_delta_norm = None
        self.last_preserve_delta_norm = None
        self.last_gate = None
        self.last_soft_gate = None
        self.last_soft_gate_mean = None
        self.last_gate_target = None
        self.last_aggressive = None
        self.last_preserve = None
        self.last_input = None
        self.last_structure_map = None
        self.last_hard_score = None
        self.last_preserve_mask = None
        self.last_preserve_mask_frac = None
        self.last_degprop_gate = None
        self.last_degprop_gate_mean = None
        self.last_freq_response = None
        self.last_degfield_mean = None

        preserve_mode = "film" if route_mode in {"film", "amir"} else "freq"
        gate_type = "spatial" if spatial_gate else ("channel" if route_mode in {"amir", "hybrid"} else "scalar")
        preserve_variant = preserve_variant if preserve_variant != "auto" else "auto"

        if self.single_route == "pres":
            self.aggressive_route = None
        else:
            self.aggressive_route = AggressiveRoute(
                num_channels=num_channels,
                prompt_dim=prompt_dim,
                hidden_dim=hidden_dim,
                res_scale=res_scale,
                variant=aggressive_variant,
            )
        if self.single_route == "agg":
            self.preserve_route = None
        else:
            self.preserve_route = StructureRoute(
                num_channels=num_channels,
                prompt_dim=prompt_dim,
                mode=preserve_mode,
                hidden_dim=hidden_dim,
                freq_strength=freq_strength,
                variant=preserve_variant,
            )
        self.mix_gate = SpatialGate(
            descriptor_dim=descriptor_dim,
            prompt_dim=prompt_dim,
            num_channels=num_channels,
            gate_type=gate_type,
            hidden_dim=hidden_dim,
            temperature=gate_temperature,
        )
        self.hard_router = HardTopKRouter(grid_size=hard_grid, topk=hard_topk) if hard_route else None
        self.adaptive_freq_modulator = (
            AdaptiveFreqModulator(
                num_channels=num_channels,
                prompt_dim=prompt_dim,
                hidden_dim=hidden_dim,
                strength=adaptive_freq_strength,
            )
            if adaptive_freq_mod else None
        )
        self.mask_preserver = (
            MaskGuidedPreserver(
                prompt_dim=prompt_dim,
                hidden_dim=hidden_dim,
                keep_ratio=mask_keep_ratio,
            )
            if mask_guided_preserve else None
        )
        self.degprop_gate = (
            GatedDegradationPropagation(
                num_channels=num_channels,
                prompt_dim=prompt_dim,
                descriptor_dim=descriptor_dim,
                hidden_dim=hidden_dim,
            )
            if gated_degprop else None
        )

    def set_condition(self, prompt_prior: torch.Tensor = None, descriptor: torch.Tensor = None):
        self.current_prompt = prompt_prior
        self.current_descriptor = descriptor

    def set_structure_target(self, structure_target: torch.Tensor | None):
        self.current_structure_target = structure_target

    def clear_runtime_cache(self) -> None:
        self.current_prompt = None
        self.current_descriptor = None
        self.current_structure_target = None
        self.last_gate_mean = None
        self.last_gate_std = None
        self.last_gate_lowfrac = None
        self.last_gate_highfrac = None
        self.last_aggressive_delta_norm = None
        self.last_preserve_delta_norm = None
        self.last_gate = None
        self.last_soft_gate = None
        self.last_soft_gate_mean = None
        self.last_gate_target = None
        self.last_aggressive = None
        self.last_preserve = None
        self.last_input = None
        self.last_structure_map = None
        self.last_hard_score = None
        self.last_preserve_mask = None
        self.last_preserve_mask_frac = None
        self.last_degprop_gate = None
        self.last_degprop_gate_mean = None
        self.last_freq_response = None
        self.last_degfield_mean = None

    def get_route_state(self) -> dict[str, torch.Tensor | float | None]:
        return {
            "gate": self.last_gate,
            "soft_gate": self.last_soft_gate,
            "structure_map": self.last_structure_map,
            "preserve_mask": self.last_preserve_mask,
            "aggressive": self.last_aggressive,
            "preserve": self.last_preserve,
            "hard_score": self.last_hard_score,
            "preserve_mask_frac": self.last_preserve_mask_frac,
        }

    def build_route_mask(
        self,
        kind: str,
        target_shape: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        kind = kind.lower()
        structure_map = self.last_structure_map
        preserve_mask = self.last_preserve_mask
        gate = self.last_gate if self.last_gate is not None else self.last_soft_gate
        if structure_map is None and preserve_mask is None and gate is None:
            return None

        if preserve_mask is None:
            if structure_map is not None:
                preserve_mask = structure_map
            elif gate is not None:
                preserve_mask = 1.0 - gate

        if kind == "preserve":
            mask = preserve_mask
        elif kind == "aggressive":
            if preserve_mask is not None:
                mask = 1.0 - preserve_mask
            else:
                mask = gate
        else:
            raise ValueError(f"Unknown route mask kind: {kind}")

        if mask is None:
            return None
        if target_shape is not None:
            mask = self._match_target_shape(mask, target_shape)
        return mask.clamp(0.0, 1.0)

    def _resolve_gate(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        gate = self.mix_gate(
            descriptor=self.current_descriptor,
            prompt=self.current_prompt,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
            feature=x,
        )
        # Keep ``mixagg`` / ``mixpres`` as biased mixtures instead of hard
        # routing; otherwise the learned gate, especially ``spgate``, never
        # participates.
        if self.route_policy == "aggressive":
            return 0.5 + 0.5 * gate
        if self.route_policy == "preserve":
            return 0.5 * gate
        return gate

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.prm_disabled:
            if self.aggressive_route is not None:
                return self.aggressive_route(x, self.current_prompt)
            elif self.preserve_route is not None:
                s = _normalized_edge_map(x.detach()).detach()
                return self.preserve_route(x, self.current_prompt, structure_target=s)
            return x

        prompt = self.current_prompt
        structure_map = self.current_structure_target
        if structure_map is None:
            structure_map = _normalized_edge_map(x.detach()).detach()
        else:
            structure_map = self._match_target_shape(structure_map.detach(), x[:, :1])
        aggressive = self.aggressive_route(x, prompt) if self.aggressive_route is not None else x
        preserve = self.preserve_route(x, prompt, structure_target=structure_map) if self.preserve_route is not None else x
        self.last_freq_response = None
        if self.adaptive_freq_modulator is not None:
            preserve_before = preserve
            preserve = self.adaptive_freq_modulator(preserve, prompt)
            freq_delta = (preserve - preserve_before).detach()
            self.last_freq_response = torch.sqrt(torch.mean(freq_delta.pow(2))).detach()

        self.last_degprop_gate = None
        self.last_degprop_gate_mean = None
        if self.degprop_gate is not None:
            prop_gate = self.degprop_gate(x, prompt, self.current_descriptor)
            aggressive = x + torch.clamp(self.degprop_strength * prop_gate, 0.0, 1.5) * (aggressive - x)
            self.last_degprop_gate = prop_gate
            self.last_degprop_gate_mean = prop_gate.detach().mean()

        self.last_degfield_mean = None
        if self.use_degfield:
            noise_proxy = _noise_proxy_map(x.detach())
            degfield = noise_proxy * (1.0 - structure_map)
            degfield = self._match_target_shape(degfield, x[:, :1]).detach()
            degfield = degfield - degfield.mean(dim=(-2, -1), keepdim=True)
            degfield_gain = torch.clamp(1.0 + self.degfield_strength * degfield, 0.5, 1.5)
            aggressive = x + degfield_gain * (aggressive - x)
            self.last_degfield_mean = degfield_gain.detach().mean()

        soft_gate = self._resolve_gate(x)
        gate = soft_gate
        self.last_hard_score = None
        if self.hard_router is not None:
            hard_gate, hard_score = self.hard_router(soft_gate, structure_map, x, self._match_target_shape)
            gate = self._match_target_shape(hard_gate, soft_gate)
            self.last_hard_score = hard_score

        if self.use_degfield:
            gate_degfield = self._match_target_shape(degfield_gain, gate)
            gate = torch.clamp(gate * gate_degfield, 0.0, 1.0)

        self.last_preserve_mask = None
        self.last_preserve_mask_frac = None
        if self.mask_preserver is not None:
            preserve_mask, _ = self.mask_preserver(structure_map, _noise_proxy_map(x.detach()), prompt)
            self.last_preserve_mask = preserve_mask
            self.last_preserve_mask_frac = preserve_mask.detach().mean()
            gate = gate * (1.0 - self._match_target_shape(preserve_mask, gate))

        self.last_input = x
        self.last_aggressive = aggressive
        self.last_preserve = preserve
        self.last_gate = gate
        self.last_soft_gate = soft_gate
        self.last_structure_map = structure_map
        self.last_gate_target = self._match_target_shape(1.0 - structure_map, soft_gate).detach()
        soft_gate_detached = soft_gate.detach()
        gate_detached = gate.detach()
        aggressive_delta = (aggressive - x).detach()
        preserve_delta = (preserve - x).detach()
        self.last_gate_mean = gate_detached.mean()
        self.last_soft_gate_mean = soft_gate_detached.mean()
        self.last_gate_std = gate_detached.std(unbiased=False)
        self.last_gate_lowfrac = (gate_detached < 0.1).float().mean()
        self.last_gate_highfrac = (gate_detached > 0.9).float().mean()
        self.last_aggressive_delta_norm = torch.sqrt(torch.mean(aggressive_delta.pow(2)))
        self.last_preserve_delta_norm = torch.sqrt(torch.mean(preserve_delta.pow(2)))
        return gate * aggressive + (1.0 - gate) * preserve

    @staticmethod
    def _match_target_shape(target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if target.shape == ref.shape:
            return target
        matched = target
        if matched.size(1) == 1 and ref.size(1) > 1:
            matched = matched.expand(-1, ref.size(1), -1, -1)
        if matched.shape[-2:] != ref.shape[-2:]:
            matched = F.interpolate(matched, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        if ref.shape[-2:] == (1, 1):
            matched = matched.mean(dim=(-2, -1), keepdim=True)
        if matched.size(1) != ref.size(1):
            matched = matched.mean(dim=1, keepdim=True)
            if ref.size(1) > 1:
                matched = matched.expand(-1, ref.size(1), -1, -1)
        return matched.clamp(0.0, 1.0)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        # E3 sigma bypass: skip polarised route blending, return backbone output
        # directly.
        if getattr(self, 'bypass_routing', False):
            return out
        if isinstance(out, tuple):
            return (self._mix(out[0]),) + out[1:]
        return self._mix(out)


def anti_forgetting_regularizer(
    named_params: list[tuple[str, torch.Tensor]],
    anchors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """L2 penalty that anchors trainable PRM parameters near their source-pretrained values."""
    if not named_params:
        return torch.tensor(0.0)
    reg = None
    for name, param in named_params:
        anchor = anchors.get(name)
        if anchor is None:
            continue
        term = (param - anchor).pow(2).sum()
        reg = term if reg is None else reg + term
    if reg is None:
        device = named_params[0][1].device
        return torch.tensor(0.0, device=device)
    return reg
