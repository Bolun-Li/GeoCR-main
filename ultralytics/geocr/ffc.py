"""Frequency-Decoupled Feature Calibration (FFC) from the GeoCR paper."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_FFC_CONFIG = {
    "use_frequency": True,
    "use_fd": True,
    "dual_affine": True,
    "use_beta": True,
    "use_modulation": True,
    "use_weighted_fusion": True,
    "residual": True,
    "residual_scale_init": 1e-6,
    "fd_kernel": 3,
    "gate_reduction": 4,
    "gamma_scale": 1.0,
    "beta_scale": 1.0,
    "rc_kernel": 3,
    "fcd_eps": 1e-6,
    "fcd_loss_gain": 1.0,
}


class LiftingFrequencyDecomposition(nn.Module):
    """Implement Eqs. (6)-(8): channel partition followed by predict/update lifting."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if channels % 2:
            raise ValueError(f"FFC requires an even channel count, got {channels}")
        half = channels // 2
        padding = kernel_size // 2
        self.predict = nn.Sequential(
            nn.Conv2d(half, half, kernel_size, padding=padding, groups=half, bias=False),
            nn.GELU(),
            nn.Conv2d(half, half, 1, bias=True),
        )
        self.update = nn.Sequential(
            nn.Conv2d(half, half, kernel_size, padding=padding, groups=half, bias=False),
            nn.GELU(),
            nn.Conv2d(half, half, 1, bias=True),
        )
        for branch in (self.predict, self.update):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse, fine = torch.chunk(feature, 2, dim=1)
        high = fine - self.predict(coarse)
        low = coarse + self.update(high)
        return low, high


class ModulationRouting(nn.Module):
    """Prompt-free MR branch in Fig. 3 and Eqs. (9)-(10)."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, use_beta: bool = True):
        super().__init__()
        hidden = max(in_channels // max(int(reduction), 1), 8)
        parameter_dim = out_channels * (2 if use_beta else 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU())
        self.modulation = nn.Linear(hidden, parameter_dim)
        self.routing = nn.Linear(hidden, 1)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.zeros_(self.routing.weight)
        nn.init.zeros_(self.routing.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.pool(feature).flatten(1)
        context = self.shared(context)
        return self.modulation(context), self.routing(context)


class ResidualCorrection(nn.Module):
    """RC block from Fig. 3: DWConv-BN-GELU-PWConv-Sigmoid."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.block(feature)


class FrequencyDecoupledFeatureCalibration(nn.Module):
    """Teacher-guided training and prompt-free student inference for one feature level."""

    def __init__(
        self,
        visual_channels: int,
        text_dim: int = 1024,
        use_frequency: bool = True,
        use_fd: bool = True,
        dual_affine: bool = True,
        use_beta: bool = True,
        use_modulation: bool = True,
        use_weighted_fusion: bool = True,
        residual: bool = True,
        residual_scale_init: float = 1e-6,
        fd_kernel: int = 3,
        gate_reduction: int = 4,
        gamma_scale: float = 1.0,
        beta_scale: float = 1.0,
        rc_kernel: int = 3,
        fcd_eps: float = 1e-6,
        fcd_loss_gain: float = 1.0,
    ):
        super().__init__()
        if visual_channels % 2:
            raise ValueError(f"FFC requires an even channel count, got {visual_channels}")
        self.visual_channels = int(visual_channels)
        self.half_channels = self.visual_channels // 2
        self.use_frequency = bool(use_frequency)
        self.use_beta = bool(use_beta)
        self.dual_affine = bool(dual_affine)
        self.use_weighted_fusion = bool(use_weighted_fusion)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.fcd_eps = float(fcd_eps)
        self.fcd_loss_gain = float(fcd_loss_gain)
        self.last_fcd_loss: torch.Tensor | None = None

        if not self.use_frequency:
            self.modulation = nn.Linear(text_dim, self.visual_channels * 2)
            self.student_plain = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.visual_channels, self.visual_channels * 2),
            )
            nn.init.zeros_(self.student_plain[-1].weight)
            nn.init.zeros_(self.student_plain[-1].bias)
            return

        # Eq. (5): the fused CPE prior produces teacher low/high affine factors.
        parameter_parts = 2 if self.use_beta else 1
        parameter_sets = 2 if self.dual_affine else 1
        parameter_dim = self.half_channels * parameter_parts * parameter_sets
        self.modulation = nn.Linear(text_dim, parameter_dim) if use_modulation else None

        self.decomposition = LiftingFrequencyDecomposition(self.visual_channels, fd_kernel) if use_fd else None
        self.student_low = ModulationRouting(
            self.visual_channels, self.half_channels, gate_reduction, self.use_beta
        )
        self.student_high = ModulationRouting(
            self.visual_channels, self.half_channels, gate_reduction, self.use_beta
        )
        self.rc_low = ResidualCorrection(self.half_channels, self.visual_channels, rc_kernel)
        self.rc_high = ResidualCorrection(self.half_channels, self.visual_channels, rc_kernel)
        # Eq. (16) uses one learnable alpha per feature level, not one value per channel.
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init))) if residual else None

    @staticmethod
    def _prepare_prior(prior: torch.Tensor) -> torch.Tensor:
        return prior.mean(dim=1) if prior.dim() == 3 else prior

    @staticmethod
    def _bound(parameters: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.tanh(parameters) * scale if scale > 0 else parameters

    def _split_branch_parameters(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        gamma, beta = torch.chunk(parameters, 2, dim=1) if self.use_beta else (parameters, None)
        shape = (-1, self.half_channels, 1, 1)
        gamma = self._bound(gamma, self.gamma_scale).view(shape)
        if beta is not None:
            beta = self._bound(beta, self.beta_scale).view(shape)
        return gamma, beta

    def _split_teacher_parameters(self, parameters: torch.Tensor):
        if self.use_beta:
            if self.dual_affine:
                gamma_low, beta_low, gamma_high, beta_high = torch.chunk(parameters, 4, dim=1)
            else:
                gamma_low, beta_low = torch.chunk(parameters, 2, dim=1)
                gamma_high, beta_high = gamma_low, beta_low
        elif self.dual_affine:
            gamma_low, gamma_high = torch.chunk(parameters, 2, dim=1)
            beta_low = beta_high = None
        else:
            gamma_low = gamma_high = parameters
            beta_low = beta_high = None

        shape = (-1, self.half_channels, 1, 1)
        gamma_low = self._bound(gamma_low, self.gamma_scale).view(shape)
        gamma_high = self._bound(gamma_high, self.gamma_scale).view(shape)
        if beta_low is not None:
            beta_low = self._bound(beta_low, self.beta_scale).view(shape)
            beta_high = self._bound(beta_high, self.beta_scale).view(shape)
        return gamma_low, beta_low, gamma_high, beta_high

    @staticmethod
    def _modulate(feature: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor | None) -> torch.Tensor:
        feature = (1.0 + gamma) * feature
        return feature + beta if beta is not None else feature

    @staticmethod
    def _descriptor(residual: torch.Tensor) -> torch.Tensor:
        descriptor = F.adaptive_avg_pool2d(residual, 1).flatten(1)
        return F.normalize(descriptor.float(), p=2, dim=1)

    def _residuals(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        gamma_low: torch.Tensor,
        beta_low: torch.Tensor | None,
        gamma_high: torch.Tensor,
        beta_high: torch.Tensor | None,
        routing_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_residual = self.rc_low(self._modulate(low, gamma_low, beta_low))
        high_residual = self.rc_high(self._modulate(high, gamma_high, beta_high))
        if self.use_weighted_fusion:
            low_weight = routing_weights[:, 0:1, None, None]
            high_weight = routing_weights[:, 1:2, None, None]
            fused = low_weight * low_residual + high_weight * high_residual
        else:
            fused = 0.5 * (low_residual + high_residual)
        return fused, low_residual, high_residual

    def _frequency_contrastive_distillation(
        self,
        student_low: torch.Tensor,
        student_high: torch.Tensor,
        teacher_low: torch.Tensor,
        teacher_high: torch.Tensor,
    ) -> torch.Tensor:
        """Implement Eq. (18), treating teacher descriptors as stop-gradient targets."""
        zsl, zsh = self._descriptor(student_low), self._descriptor(student_high)
        ztl = self._descriptor(teacher_low).detach()
        zth = self._descriptor(teacher_high).detach()
        numerator = (zsl - ztl).square().sum(1) + (zsh - zth).square().sum(1)
        denominator = (zsl - zth).square().sum(1) + (zsh - ztl).square().sum(1)
        return (numerator / denominator.clamp_min(self.fcd_eps)).mean()

    def _plain_calibration(self, feature: torch.Tensor, prior: torch.Tensor | None) -> torch.Tensor:
        if self.training and prior is not None:
            parameters = self.modulation(self._prepare_prior(prior))
        else:
            parameters = self.student_plain(feature)
        parameters = parameters.to(dtype=feature.dtype)
        gamma, beta = torch.chunk(parameters, 2, dim=1)
        gamma = self._bound(gamma, self.gamma_scale).view(-1, self.visual_channels, 1, 1)
        beta = self._bound(beta, self.beta_scale).view(-1, self.visual_channels, 1, 1)
        return self._modulate(feature, gamma, beta)

    def switch_to_deploy(self):
        """Permanently remove the training-only CPE teacher from this FFC level."""
        self.modulation = None
        self.last_fcd_loss = None
        return self

    def forward(self, feature: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        self.last_fcd_loss = None
        if not self.use_frequency:
            if prior is not None:
                prior = prior.to(device=feature.device, dtype=feature.dtype)
            return self._plain_calibration(feature, prior)

        low, high = self.decomposition(feature) if self.decomposition is not None else torch.chunk(feature, 2, 1)

        student_low_parameters, rho_low = self.student_low(feature)
        student_high_parameters, rho_high = self.student_high(feature)
        student_gamma_low, student_beta_low = self._split_branch_parameters(student_low_parameters)
        student_gamma_high, student_beta_high = self._split_branch_parameters(student_high_parameters)
        routing_weights = torch.softmax(torch.cat((rho_low, rho_high), dim=1), dim=1)
        student_fused, student_low_residual, student_high_residual = self._residuals(
            low,
            high,
            student_gamma_low,
            student_beta_low,
            student_gamma_high,
            student_beta_high,
            routing_weights,
        )

        # CPE and the teacher branch exist only during training. Evaluation and
        # deployment always use the prompt-free student, even if a caller passes priors.
        output_residual = student_fused
        if self.training and prior is not None and self.modulation is not None:
            prior = self._prepare_prior(prior)
            prior = prior.to(device=self.modulation.weight.device, dtype=self.modulation.weight.dtype)
            teacher_parameters = self.modulation(prior).to(dtype=feature.dtype)
            teacher_gamma_low, teacher_beta_low, teacher_gamma_high, teacher_beta_high = (
                self._split_teacher_parameters(teacher_parameters)
            )
            teacher_fused, teacher_low_residual, teacher_high_residual = self._residuals(
                low,
                high,
                teacher_gamma_low,
                teacher_beta_low,
                teacher_gamma_high,
                teacher_beta_high,
                routing_weights,
            )
            output_residual = teacher_fused
            self.last_fcd_loss = self._frequency_contrastive_distillation(
                student_low_residual,
                student_high_residual,
                teacher_low_residual,
                teacher_high_residual,
            )

        if self.residual_scale is None:
            return output_residual
        return feature + self.residual_scale.to(dtype=feature.dtype) * output_residual
