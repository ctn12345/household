"""
ADLG-HFormer: Adaptive Decomposition and Local-Global Horizon-aware Transformer.

This module implements a multivariate time-series forecasting model with:

1. Adaptive multi-scale trend decomposition.
2. Difference-aware dynamic multi-scale temporal convolution.
3. Transformer-based global dependency modeling.
4. Horizon-aware query decoding.
5. Gated fusion of trend and residual predictions.

Input:
    x: [batch_size, input_length, input_dim]

Output:
    prediction: [batch_size, prediction_length]
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first sequence tensors."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(position * div_term)

        cosine_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(
            position * div_term[:cosine_width]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional information to an input sequence.

        Args:
            x: Tensor with shape [B, T, D].

        Returns:
            Tensor with shape [B, T, D].
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3-D tensor [B, T, D], got {tuple(x.shape)}."
            )

        sequence_length = x.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"max_len={self.encoding.size(1)}."
            )

        positional_encoding = self.encoding[
            :, :sequence_length, :
        ].to(
            device=x.device,
            dtype=x.dtype,
        )

        return self.dropout(x + positional_encoding)


class AdaptiveTrendDecomposition(nn.Module):
    """
    Adaptive multi-scale trend decomposition.

    Several moving-average branches extract candidate trends at different
    temporal scales. A sample-dependent gating network calculates the fusion
    weights for the candidate trends.

    Formally:

        T_k = MovingAverage_k(X)
        alpha = Softmax(Gate(MeanPool(X)))
        T = sum_k alpha_k * T_k
        R = X - T
    """

    def __init__(
        self,
        input_dim: int,
        kernel_sizes: Sequence[int] = (3, 7, 15),
        gate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        self._validate_kernel_sizes(kernel_sizes)

        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be positive, got {input_dim}."
            )
        if gate_hidden_dim <= 0:
            raise ValueError(
                "gate_hidden_dim must be positive, "
                f"got {gate_hidden_dim}."
            )

        self.kernel_sizes = tuple(
            int(kernel_size)
            for kernel_size in kernel_sizes
        )

        self.scale_gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dim,
                len(self.kernel_sizes),
            ),
        )

    @staticmethod
    def _validate_kernel_sizes(
        kernel_sizes: Sequence[int],
    ) -> None:
        if not kernel_sizes:
            raise ValueError(
                "kernel_sizes must contain at least one value."
            )

        for kernel_size in kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    "Each moving-average kernel size must be a "
                    f"positive odd integer, got {kernel_size}."
                )

    @staticmethod
    def _moving_average(
        x: Tensor,
        kernel_size: int,
    ) -> Tensor:
        """
        Calculate a length-preserving moving average.

        Replication padding is used instead of zero padding to avoid artificial
        boundary drops.

        Args:
            x: Tensor with shape [B, T, C].
            kernel_size: Positive odd moving-average window.

        Returns:
            Tensor with shape [B, T, C].
        """
        padding = kernel_size // 2

        x_channels_first = x.transpose(1, 2)
        x_padded = F.pad(
            x_channels_first,
            pad=(padding, padding),
            mode="replicate",
        )

        smoothed = F.avg_pool1d(
            x_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        return smoothed.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Decompose the input sequence into trend and residual components.

        Args:
            x: Tensor with shape [B, T, C].

        Returns:
            trend:
                Tensor with shape [B, T, C].
            residual:
                Tensor with shape [B, T, C].
            scale_weights:
                Tensor with shape [B, K].
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, T, C], got {tuple(x.shape)}."
            )

        candidate_trends = torch.stack(
            [
                self._moving_average(
                    x,
                    kernel_size,
                )
                for kernel_size in self.kernel_sizes
            ],
            dim=-1,
        )
        # candidate_trends: [B, T, C, K]

        global_context = x.mean(dim=1)

        scale_weights = torch.softmax(
            self.scale_gate(global_context),
            dim=-1,
        )
        # scale_weights: [B, K]

        expanded_weights = scale_weights[
            :, None, None, :
        ]
        trend = torch.sum(
            candidate_trends * expanded_weights,
            dim=-1,
        )
        residual = x - trend

        return trend, residual, scale_weights


class DifferenceAwareDynamicMultiScaleConv(nn.Module):
    """
    Difference-aware dynamic multi-scale temporal convolution.

    The input representation contains residual values and their first-order
    differences. Parallel temporal convolution branches capture short-,
    medium-, and relatively long-range local patterns. A sample-dependent gate
    dynamically fuses these branches.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        dilations: Sequence[int] = (1, 2, 4),
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._validate_branch_configuration(
            kernel_sizes=kernel_sizes,
            dilations=dilations,
        )

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}."
            )
        if gate_hidden_dim <= 0:
            raise ValueError(
                "gate_hidden_dim must be positive, "
                f"got {gate_hidden_dim}."
            )

        self.kernel_sizes = tuple(
            int(kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dilations = tuple(
            int(dilation)
            for dilation in dilations
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=(
                            dilation
                            * (kernel_size - 1)
                            // 2
                        ),
                    ),
                    nn.GELU(),
                )
                for kernel_size, dilation in zip(
                    self.kernel_sizes,
                    self.dilations,
                )
            ]
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dim,
                len(self.branches),
            ),
        )

        self.output_projection = nn.Linear(
            d_model,
            d_model,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _validate_branch_configuration(
        kernel_sizes: Sequence[int],
        dilations: Sequence[int],
    ) -> None:
        if not kernel_sizes:
            raise ValueError(
                "kernel_sizes must contain at least one value."
            )

        if len(kernel_sizes) != len(dilations):
            raise ValueError(
                "kernel_sizes and dilations must have the same length."
            )

        for kernel_size in kernel_sizes:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    "Each convolution kernel size must be a "
                    f"positive odd integer, got {kernel_size}."
                )

        for dilation in dilations:
            if dilation <= 0:
                raise ValueError(
                    "Each dilation must be positive, "
                    f"got {dilation}."
                )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Tensor with shape [B, T, D].

        Returns:
            output:
                Tensor with shape [B, T, D].
            branch_weights:
                Tensor with shape [B, K].
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, T, D], got {tuple(x.shape)}."
            )

        residual = x
        x_channels_first = x.transpose(1, 2)

        branch_outputs = torch.stack(
            [
                branch(x_channels_first)
                for branch in self.branches
            ],
            dim=-1,
        )
        # branch_outputs: [B, D, T, K]

        branch_outputs = branch_outputs.permute(
            0,
            2,
            1,
            3,
        )
        # branch_outputs: [B, T, D, K]

        global_context = x.mean(dim=1)

        branch_weights = torch.softmax(
            self.branch_gate(global_context),
            dim=-1,
        )
        # branch_weights: [B, K]

        expanded_weights = branch_weights[
            :, None, None, :
        ]

        fused = torch.sum(
            branch_outputs * expanded_weights,
            dim=-1,
        )
        fused = self.output_projection(fused)
        fused = self.dropout(fused)

        output = self.norm(
            residual + fused
        )

        return output, branch_weights


class HorizonQueryDecoder(nn.Module):
    """
    Horizon-aware cross-attention decoder.

    A learnable query vector is assigned to every future prediction step.
    Therefore, different forecast horizons can attend to different positions
    of the encoded historical sequence.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        pred_len: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if pred_len <= 0:
            raise ValueError(
                f"pred_len must be positive, got {pred_len}."
            )
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}."
            )

        self.pred_len = pred_len

        self.horizon_queries = nn.Parameter(
            torch.empty(
                1,
                pred_len,
                d_model,
            )
        )
        nn.init.normal_(
            self.horizon_queries,
            mean=0.0,
            std=0.02,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(
                d_model,
                dim_feedforward,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                dim_feedforward,
                d_model,
            ),
            nn.Dropout(dropout),
        )
        self.feedforward_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        memory: Tensor,
    ) -> Tensor:
        """
        Args:
            memory: Tensor with shape [B, T, D].

        Returns:
            Tensor with shape [B, pred_len, D].
        """
        if memory.ndim != 3:
            raise ValueError(
                f"Expected memory shape [B, T, D], got {tuple(memory.shape)}."
            )

        batch_size = memory.size(0)

        queries = self.horizon_queries.expand(
            batch_size,
            -1,
            -1,
        )

        attended, _ = self.cross_attention(
            query=queries,
            key=memory,
            value=memory,
            need_weights=False,
        )

        hidden = self.attention_norm(
            queries
            + self.attention_dropout(attended)
        )

        hidden = self.feedforward_norm(
            hidden
            + self.feedforward(hidden)
        )

        return hidden


class ADLG_HFormer(nn.Module):
    """
    Adaptive Decomposition and Local-Global Horizon-aware Transformer.

    Architecture:
        1. Adaptive multi-scale trend decomposition.
        2. Difference-aware residual enhancement.
        3. Dynamic multi-scale local temporal convolution.
        4. Transformer-based global temporal encoding.
        5. Horizon-aware query decoding.
        6. Gated fusion of trend and residual predictions.

    Args:
        input_dim:
            Number of input variables.
        d_model:
            Hidden representation dimension.
        nhead:
            Number of attention heads.
        num_layers:
            Number of Transformer encoder layers.
        dim_feedforward:
            Hidden dimension of Transformer feed-forward layers.
        pred_len:
            Number of future time steps to predict.
        dropout:
            Dropout probability.
        trend_kernel_sizes:
            Moving-average window sizes used by trend decomposition.
        conv_kernel_sizes:
            Kernel sizes used by local temporal convolution.
        conv_dilations:
            Dilation factors corresponding to the convolution branches.
        max_len:
            Maximum supported input sequence length.

    Input:
        x: Tensor with shape [B, input_len, input_dim].

    Output:
        prediction: Tensor with shape [B, pred_len].
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        pred_len: int = 90,
        dropout: float = 0.1,
        trend_kernel_sizes: Sequence[int] = (3, 7, 15),
        conv_kernel_sizes: Sequence[int] = (3, 5, 7),
        conv_dilations: Sequence[int] = (1, 2, 4),
        max_len: int = 5000,
    ) -> None:
        super().__init__()

        self._validate_hyperparameters(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            pred_len=pred_len,
            dropout=dropout,
        )

        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_len = pred_len

        self.decomposition = AdaptiveTrendDecomposition(
            input_dim=input_dim,
            kernel_sizes=trend_kernel_sizes,
        )

        # Concatenate residual values and their first-order differences.
        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                d_model,
            ),
            nn.LayerNorm(d_model),
        )

        self.local_encoder = (
            DifferenceAwareDynamicMultiScaleConv(
                d_model=d_model,
                kernel_sizes=conv_kernel_sizes,
                dilations=conv_dilations,
                dropout=dropout,
            )
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.global_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.horizon_decoder = HorizonQueryDecoder(
            d_model=d_model,
            nhead=nhead,
            pred_len=pred_len,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # The trend head uses both the average trend and the latest trend.
        self.trend_head = nn.Sequential(
            nn.Linear(
                input_dim * 2,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                d_model,
                pred_len,
            ),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                d_model // 2,
                1,
            ),
        )

        self.correction_gate = nn.Sequential(
            nn.Linear(
                d_model,
                1,
            ),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    @staticmethod
    def _validate_hyperparameters(
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        pred_len: int,
        dropout: float,
    ) -> None:
        positive_parameters = {
            "input_dim": input_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "pred_len": pred_len,
        }

        for name, value in positive_parameters.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive, got {value}."
                )

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}."
            )

    def _reset_parameters(self) -> None:
        """Initialize trainable linear and convolution parameters."""
        for module in self.modules():
            if isinstance(
                module,
                (nn.Linear, nn.Conv1d),
            ):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    @staticmethod
    def _first_difference(
        x: Tensor,
    ) -> Tensor:
        """
        Calculate a length-preserving first-order temporal difference.

        The first time step is filled with zeros.
        """
        difference = torch.zeros_like(x)
        difference[:, 1:, :] = (
            x[:, 1:, :]
            - x[:, :-1, :]
        )
        return difference

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Forecast a future sequence.

        Args:
            x: Input tensor with shape [B, T, C].

        Returns:
            Prediction tensor with shape [B, pred_len].
        """
        self._validate_input(x)

        # 1. Adaptive trend-residual decomposition.
        trend, residual, _ = self.decomposition(x)

        # 2. Difference-aware residual representation.
        residual_difference = self._first_difference(
            residual
        )

        local_input = torch.cat(
            [
                residual,
                residual_difference,
            ],
            dim=-1,
        )

        # 3. Local multi-scale temporal modeling.
        hidden = self.input_projection(
            local_input
        )
        hidden, _ = self.local_encoder(
            hidden
        )

        # 4. Global dependency modeling.
        hidden = self.positional_encoding(
            hidden
        )
        memory = self.global_encoder(
            hidden
        )

        # 5. Horizon-specific historical retrieval.
        horizon_features = self.horizon_decoder(
            memory
        )

        # 6. Trend baseline prediction.
        trend_mean = trend.mean(
            dim=1
        )
        trend_last = trend[
            :, -1, :
        ]

        trend_summary = torch.cat(
            [
                trend_mean,
                trend_last,
            ],
            dim=-1,
        )

        trend_prediction = self.trend_head(
            trend_summary
        )

        # 7. Nonlinear residual correction.
        residual_prediction = self.residual_head(
            horizon_features
        ).squeeze(-1)

        correction_weight = self.correction_gate(
            horizon_features
        ).squeeze(-1)

        prediction = (
            trend_prediction
            + correction_weight
            * residual_prediction
        )

        return prediction

    def forward_with_auxiliary(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """
        Run forecasting and return interpretable intermediate variables.

        This method is intended for visualization and ablation analysis.
        The standard training loop should continue using ``forward``.

        Returns:
            Dictionary containing:
                prediction: [B, pred_len]
                trend: [B, T, C]
                residual: [B, T, C]
                trend_scale_weights: [B, K_trend]
                conv_branch_weights: [B, K_conv]
                correction_weights: [B, pred_len]
        """
        self._validate_input(x)

        trend, residual, trend_scale_weights = (
            self.decomposition(x)
        )

        residual_difference = self._first_difference(
            residual
        )

        local_input = torch.cat(
            [
                residual,
                residual_difference,
            ],
            dim=-1,
        )

        hidden = self.input_projection(
            local_input
        )

        hidden, conv_branch_weights = (
            self.local_encoder(hidden)
        )

        hidden = self.positional_encoding(
            hidden
        )
        memory = self.global_encoder(
            hidden
        )

        horizon_features = self.horizon_decoder(
            memory
        )

        trend_summary = torch.cat(
            [
                trend.mean(dim=1),
                trend[:, -1, :],
            ],
            dim=-1,
        )

        trend_prediction = self.trend_head(
            trend_summary
        )

        residual_prediction = self.residual_head(
            horizon_features
        ).squeeze(-1)

        correction_weights = self.correction_gate(
            horizon_features
        ).squeeze(-1)

        prediction = (
            trend_prediction
            + correction_weights
            * residual_prediction
        )

        return {
            "prediction": prediction,
            "trend": trend,
            "residual": residual,
            "trend_scale_weights": trend_scale_weights,
            "conv_branch_weights": conv_branch_weights,
            "correction_weights": correction_weights,
        }

    def _validate_input(
        self,
        x: Tensor,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, T, C], got {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got {x.size(-1)}."
            )


# Optional alias following standard CamelCase naming.
ADLGHFormer = ADLG_HFormer


def run_model_test() -> None:
    """Run minimal forward and backward tests."""
    torch.manual_seed(2026)

    batch_size = 4
    input_len = 90
    input_dim = 20

    for pred_len in (90, 365):
        model = ADLG_HFormer(
            input_dim=input_dim,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            pred_len=pred_len,
            dropout=0.1,
        )

        inputs = torch.randn(
            batch_size,
            input_len,
            input_dim,
        )
        targets = torch.randn(
            batch_size,
            pred_len,
        )

        predictions = model(inputs)
        loss = F.mse_loss(
            predictions,
            targets,
        )
        loss.backward()

        assert predictions.shape == (
            batch_size,
            pred_len,
        )

        print(
            f"pred_len={pred_len:3d} | "
            f"input={tuple(inputs.shape)} | "
            f"output={tuple(predictions.shape)} | "
            f"loss={loss.item():.6f}"
        )


if __name__ == "__main__":
    run_model_test()
