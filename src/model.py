"""
Hybrid GRU + TCN Model Architecture
=====================================
A multitask deep learning model for Smart Grid analytics:
  - Shared Encoder: GRU (sequential patterns) + TCN (temporal convolutions)
  - Forecasting Head: Dense regression for MW prediction
  - Theft Detection Head: Dense + Sigmoid for theft probability

Architecture (matches the project design diagram):
  Input → GRU Encoder → }
                         } → Concatenate → Latent Feature → Forecasting Head → MW
  Input → TCN Encoder → }                                 → Theft Head → P(theft)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Temporal Convolutional Network (TCN) Components
# ============================================================

class CausalConv1d(nn.Module):
    """Causal convolution: ensures no future information leakage."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class TCNBlock(nn.Module):
    """
    A single TCN residual block with two causal convolutions.
    Includes residual connection and optional downsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)

        # Residual connection (1x1 conv if channel sizes differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.
    Stacks multiple TCN blocks with exponentially increasing dilation.
    """

    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features) - input time series
        Returns:
            out: (batch, out_channels) - last time step output
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # Take the last time step
        return out[:, :, -1]


# ============================================================
# GRU Encoder
# ============================================================

class GRUEncoder(nn.Module):
    """
    GRU-based recurrent encoder for sequential pattern learning.
    Uses multi-layer GRU with dropout.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            out: (batch, hidden_size) - last hidden state
        """
        output, hidden = self.gru(x)
        # Use the last time step output
        last_output = output[:, -1, :]
        return self.layer_norm(last_output)


# ============================================================
# Task-Specific Heads
# ============================================================

class ForecastingHead(nn.Module):
    """
    Dense regression head for demand forecasting.
    Predicts a single MW value.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class TheftDetectionHead(nn.Module):
    """
    Dense + Sigmoid head for theft detection.
    Outputs a probability of energy theft.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================
# Unified Multitask Model
# ============================================================

class SmartGridModel(nn.Module):
    """
    Hybrid GRU + TCN Multitask Model for Smart Grid Analytics.

    Architecture:
        Input Sequence
            ├→ GRU Encoder → gru_features
            ├→ TCN Encoder → tcn_features
            └→ Concatenate [gru_features, tcn_features]
                └→ Projection → Latent Feature Representation
                    ├→ Forecasting Head → Predicted MW
                    └→ Theft Detection Head → Theft Probability
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config['model']

        # Get input sizes (will be set dynamically)
        self.demand_input_size = None
        self.theft_input_size = None

        # Shared encoders
        gru_hidden = model_cfg['gru_hidden_size']
        gru_layers = model_cfg['gru_num_layers']
        gru_dropout = model_cfg['gru_dropout']
        tcn_channels = model_cfg['tcn_num_channels']
        tcn_kernel = model_cfg['tcn_kernel_size']
        tcn_dropout = model_cfg['tcn_dropout']
        latent_dim = model_cfg['latent_dim']

        # These will be initialized when input sizes are known
        self.gru_encoder = None
        self.tcn_encoder = None

        # Store config for lazy initialization
        self._gru_hidden = gru_hidden
        self._gru_layers = gru_layers
        self._gru_dropout = gru_dropout
        self._tcn_channels = tcn_channels
        self._tcn_kernel = tcn_kernel
        self._tcn_dropout = tcn_dropout
        self._latent_dim = latent_dim

        # Combined feature dimension = gru_hidden + tcn_out_channels
        combined_dim = gru_hidden + tcn_channels[-1]

        # Latent feature projection
        self.latent_projection = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.2)
        )

        # Task-specific heads
        self.forecasting_head = ForecastingHead(
            input_dim=latent_dim,
            hidden_dim=model_cfg['forecast_hidden'],
            output_dim=model_cfg['forecast_output']
        )

        self.theft_head = TheftDetectionHead(
            input_dim=latent_dim,
            hidden_dim=model_cfg['theft_hidden'],
            output_dim=model_cfg['theft_output']
        )

    def _init_encoders(self, input_size):
        """Lazily initialize encoders when input size is known."""
        device = next(self.latent_projection.parameters()).device

        self.gru_encoder = GRUEncoder(
            input_size=input_size,
            hidden_size=self._gru_hidden,
            num_layers=self._gru_layers,
            dropout=self._gru_dropout
        ).to(device)

        self.tcn_encoder = TCNEncoder(
            input_size=input_size,
            num_channels=self._tcn_channels,
            kernel_size=self._tcn_kernel,
            dropout=self._tcn_dropout
        ).to(device)

    def encode(self, x):
        """
        Shared encoding: GRU + TCN → Latent features.

        Args:
            x: (batch, seq_len, features)
        Returns:
            latent: (batch, latent_dim)
        """
        input_size = x.shape[-1]

        # Lazy init encoders
        if self.gru_encoder is None or self.gru_encoder.gru.input_size != input_size:
            self._init_encoders(input_size)

        gru_out = self.gru_encoder(x)      # (batch, gru_hidden)
        tcn_out = self.tcn_encoder(x)      # (batch, tcn_out_channels)

        # Concatenate GRU and TCN features
        combined = torch.cat([gru_out, tcn_out], dim=-1)

        # Project to latent space
        latent = self.latent_projection(combined)

        return latent

    def forward(self, x, task='both'):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features)
            task: 'forecast', 'theft', or 'both'

        Returns:
            dict with 'forecast' and/or 'theft' predictions
        """
        latent = self.encode(x)
        output = {}

        if task in ('forecast', 'both'):
            output['forecast'] = self.forecasting_head(latent)

        if task in ('theft', 'both'):
            output['theft'] = self.theft_head(latent)

        return output

    def get_model_summary(self):
        """Return a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Hybrid GRU + TCN (Multitask)',
            'tasks': ['Demand Forecasting (Regression)', 'Theft Detection (Classification)']
        }


# ============================================================
# Model Factory
# ============================================================

def create_model(config, device=None):
    """Create and initialize the SmartGrid model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SmartGridModel(config).to(device)
    print(f"\n[✓] Model created on device: {device}")

    summary = model.get_model_summary()
    print(f"    Architecture: {summary['architecture']}")
    print(f"    Tasks: {summary['tasks']}")

    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.utils import load_config

    config = load_config()
    model = create_model(config)

    # Test with dummy data
    batch_size = 4
    seq_len = 24
    features = 15  # example feature count

    dummy_input = torch.randn(batch_size, seq_len, features)
    output = model(dummy_input, task='both')

    print(f"\n    Input shape:     {dummy_input.shape}")
    print(f"    Forecast output: {output['forecast'].shape}")
    print(f"    Theft output:    {output['theft'].shape}")

    summary = model.get_model_summary()
    print(f"    Total params:    {summary['total_parameters']:,}")
    print(f"    Trainable:       {summary['trainable_parameters']:,}")
