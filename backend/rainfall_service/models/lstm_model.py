"""
Bidirectional LSTM with Attention for Rainfall Prediction.

Architecture:
  Input: (batch, seq_len=30, n_features=15) — 30-day meteorological window
       │
  ┌────▼──────────────────────────────────────────────┐
  │  Input projection + positional encoding            │
  └────────────────────────────────────────┬──────────┘
       │
  ┌────▼──────────────────────────────────────────────┐
  │  BiLSTM Stack (3 layers, 128 hidden, dropout=0.3)  │
  └────────────────────────────────────────┬──────────┘
       │
  ┌────▼──────────────────────────────────────────────┐
  │  Multi-Head Self-Attention (4 heads)               │
  └────────────────────────────────────────┬──────────┘
       │
  ┌────▼──────────────────────────────────────────────┐
  │  Projection Head → regression (rainfall mm)        │
  │                  → classification (5 classes)      │
  └────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttentionPool(nn.Module):
    """Multi-head attention for temporal pooling over LSTM output."""
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # x: (batch, seq, d_model)
        q = self.query.expand(x.size(0), -1, -1)   # (batch, 1, d_model)
        out, weights = self.attn(q, x, x)           # (batch, 1, d_model)
        return out.squeeze(1), weights               # (batch, d_model)


class RainfallLSTM(nn.Module):
    """
    Bidirectional LSTM with multi-head attention for rainfall prediction.
    Outputs both regression (mm) and classification (6 categories) heads.
    """

    N_CLASSES = 5   # No Rain / Light / Moderate / Heavy / Very Heavy 

    def __init__(
        self,
        n_features: int = 15,
        hidden_size: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        proj_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(proj_dim, dropout=dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        lstm_out = hidden_size * 2  # bidirectional

        # Layer norm
        self.norm = nn.LayerNorm(lstm_out)

        # Attention pooling
        self.attn_pool = MultiHeadAttentionPool(lstm_out, n_heads)

        # Shared MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(lstm_out, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression head (predict rainfall mm)
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),          # ensure non-negative output
        )

        # Classification head (5 categories)
        self.classification_head = nn.Linear(128, self.N_CLASSES)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            mm_pred:   (batch, 1)
            class_logits: (batch, N_CLASSES)
            features:  (batch, 128) — for ensemble
        """
        # Project + positional encoding
        out = self.input_proj(x)
        out = self.pos_enc(out)

        # LSTM
        lstm_out, _ = self.lstm(out)
        lstm_out = self.norm(lstm_out)

        # Attention pool
        pooled, _ = self.attn_pool(lstm_out)

        # Shared MLP
        features = self.shared_mlp(pooled)

        mm_pred = self.regression_head(features)
        class_logits = self.classification_head(features)

        return mm_pred, class_logits, features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-dim feature embedding for ensemble."""
        with torch.no_grad():
            _, _, feat = self.forward(x)
        return feat
