"""
Attention-Based Neural Network for DeFi Price Prediction

Architecture:
- Multi-head self-attention mechanism
- Transformer-style architecture for time-series
- Input: 30 timesteps × 40 features
- Output: Continuous price movement prediction

Benefits:
- Learns which timesteps are most important
- Interpretable attention weights
- Captures long-range dependencies efficiently
- No recurrence needed (parallel processing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Allows the model to attend to information from different representation subspaces.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize multi-head attention.

        Args:
            hidden_size: Dimension of features (128)
            num_heads: Number of attention heads (8)
            dropout: Dropout rate (0.1)
            device: Device to run on
        """
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.device = device

        # Linear transformations for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional mask tensor

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)    # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)

        # Reshape for multi-head: (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now shape: (batch_size, num_heads, seq_len, head_dim)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        # context shape: (batch_size, seq_len, num_heads, head_dim)
        context = context.view(batch_size, seq_len, hidden_size)
        # context shape: (batch_size, seq_len, hidden_size)

        # Output projection
        output = self.out_proj(context)

        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)

        return output, avg_attention

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights without computing output."""
        _, attention_weights = self.forward(x)
        return attention_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 8,
        ff_size: int = 512,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize transformer block.

        Args:
            hidden_size: Feature dimension
            num_heads: Number of attention heads
            ff_size: Feed-forward network hidden size
            dropout: Dropout rate
            device: Device to run on
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (output, attention_weights)
        """
        # Multi-head attention with residual connection
        attn_out, attn_weights = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward network with residual connection
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        return x, attn_weights


class AttentionModel(nn.Module):
    """
    Attention-based model for time-series prediction.

    Uses multi-head self-attention to learn which timesteps are important.
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_size: int = 512,
        dropout: float = 0.1,
        output_size: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize attention model.

        Args:
            input_size: Number of features per timestep (40)
            hidden_size: Embedding dimension (128)
            num_heads: Number of attention heads (8)
            num_layers: Number of transformer blocks (2)
            ff_size: Feed-forward network hidden size
            dropout: Dropout rate (0.1)
            output_size: Output dimension (1)
            device: Device to run on
        """
        super(AttentionModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        # Input embedding
        self.embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.register_buffer('positional_encoding', self._generate_positional_encoding(30, hidden_size))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_size=ff_size,
                dropout=dropout,
                device=device
            )
            for _ in range(num_layers)
        ])

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)

        self._initialize_weights()

    def _generate_positional_encoding(
        self,
        seq_len: int,
        hidden_size: int
    ) -> torch.Tensor:
        """
        Generate positional encoding for transformer.

        Args:
            seq_len: Sequence length
            hidden_size: Embedding dimension

        Returns:
            Positional encoding tensor
        """
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, seq_len, hidden_size)

    def _initialize_weights(self):
        """Initialize model weights."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            output: Prediction tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape

        # Embed input
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)

        # Apply transformer blocks
        attention_weights = None
        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)

        # Use mean pooling across sequence dimension
        x = x.mean(dim=1)  # (batch_size, hidden_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout_layer(x)

        output = self.fc2(x)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Args:
            x: Input tensor

        Returns:
            Attention weights from last transformer block
        """
        batch_size, seq_len, _ = x.shape

        x = self.embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)

        for transformer_block in self.transformer_blocks:
            x, attention_weights = transformer_block(x)

        return attention_weights

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()
        return self

    def to_train_mode(self):
        """Set model to training mode."""
        self.train()
        return self


def create_attention_model(
    input_size: int = 40,
    hidden_size: int = 128,
    device: str = 'cpu'
) -> AttentionModel:
    """
    Factory function to create attention model.

    Args:
        input_size: Number of input features
        hidden_size: Embedding dimension
        device: Device to run on

    Returns:
        AttentionModel instance
    """
    model = AttentionModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=8,
        num_layers=2,
        ff_size=512,
        dropout=0.1,
        output_size=1,
        device=device
    )

    logger.info(f"Created Attention model with {model.count_parameters():,} parameters")
    return model.to(device)
