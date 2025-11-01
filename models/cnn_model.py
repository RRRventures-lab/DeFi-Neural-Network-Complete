"""
Convolutional Neural Network Model for DeFi Price Prediction

Architecture:
- Input: 30 timesteps × 40 features
- Multiple 1D Convolutional layers with varying kernel sizes
- MaxPooling for dimensionality reduction
- Output: Continuous price movement prediction

Benefits:
- Learns hierarchical feature patterns
- Efficient computation with multi-scale kernels
- Captures both local and broader temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class CNNModel(nn.Module):
    """
    1D Convolutional Neural Network for time-series prediction.

    Uses multiple parallel convolutional paths with different kernel sizes
    to capture patterns at multiple time scales.
    """

    def __init__(
        self,
        input_size: int = 40,
        num_filters: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.2,
        output_size: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize CNN model.

        Args:
            input_size: Number of features per timestep (40)
            num_filters: Number of filters for each layer [32, 64, 128]
            kernel_sizes: Kernel sizes for convolution [3, 5, 7]
            dropout: Dropout rate (0.2)
            output_size: Output dimension (1 for regression)
            device: Device to run on ('cpu' or 'cuda')
        """
        super(CNNModel, self).__init__()

        if num_filters is None:
            num_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.output_size = output_size
        self.device = device

        # Parallel convolutional paths with different kernel sizes
        self.conv_layers = nn.ModuleList()

        for i, (num_filter, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            if i == 0:
                in_channels = input_size
            else:
                in_channels = num_filters[i - 1]

            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                ),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv)

        # Calculate size after convolutions
        # Each MaxPool1d with stride=2 reduces dimension by 2
        self._calculate_fc_input_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self._initialize_weights()

    def _calculate_fc_input_size(self):
        """Calculate flattened size after convolutional layers."""
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, self.input_size, 30)
        with torch.no_grad():
            x = dummy_input
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            self.fc_input_size = x.view(x.size(0), -1).size(1)

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 30, 40) - 32 samples, 30 timesteps, 40 features

        Returns:
            output: Prediction tensor of shape (batch_size, output_size)
        """
        # x shape: (batch_size, seq_len, input_size)
        # Transpose to (batch_size, input_size, seq_len) for Conv1d
        x = x.transpose(1, 2)
        # x shape: (batch_size, input_size, seq_len)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # x shape: (batch_size, num_filters[-1], reduced_seq_len)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        # x shape: (batch_size, fc_input_size)

        # Fully connected layers with batch norm and dropout
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout_layer(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout_layer(x)

        output = self.fc3(x)
        # output shape: (batch_size, output_size)

        return output

    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get features from convolutional layers (before FC layers).

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Feature tensor from last conv layer
        """
        x = x.transpose(1, 2)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x

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


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN with parallel branches at different depths.

    Combines shallow and deep convolutional branches for multi-resolution
    feature extraction.
    """

    def __init__(
        self,
        input_size: int = 40,
        dropout: float = 0.2,
        device: str = 'cpu'
    ):
        """
        Initialize multi-scale CNN.

        Args:
            input_size: Number of features per timestep
            dropout: Dropout rate
            device: Device to run on
        """
        super(MultiScaleCNN, self).__init__()

        self.input_size = input_size
        self.dropout = dropout
        self.device = device

        # Shallow branch (fast, captures local patterns)
        self.shallow = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        # Medium branch (captures mid-range patterns)
        self.medium = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        # Deep branch (captures long-range patterns)
        self.deep = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        # Calculate combined size
        dummy_input = torch.zeros(1, input_size, 30)
        with torch.no_grad():
            shallow_out = self.shallow(dummy_input)
            medium_out = self.medium(dummy_input)
            deep_out = self.deep(dummy_input)

            combined_size = (
                shallow_out.view(1, -1).size(1) +
                medium_out.view(1, -1).size(1) +
                deep_out.view(1, -1).size(1)
            )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize all weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            output: Prediction tensor
        """
        x_t = x.transpose(1, 2)

        # Process through different scales
        shallow = self.shallow(x_t).view(x_t.size(0), -1)
        medium = self.medium(x_t).view(x_t.size(0), -1)
        deep = self.deep(x_t).view(x_t.size(0), -1)

        # Concatenate features
        combined = torch.cat([shallow, medium, deep], dim=1)

        # Fusion
        output = self.fusion(combined)

        return output

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_model(
    input_size: int = 40,
    num_filters: List[int] = None,
    device: str = 'cpu'
) -> CNNModel:
    """
    Factory function to create CNN model.

    Args:
        input_size: Number of input features
        num_filters: Number of filters for each layer
        device: Device to run on

    Returns:
        CNNModel instance
    """
    if num_filters is None:
        num_filters = [32, 64, 128]

    model = CNNModel(
        input_size=input_size,
        num_filters=num_filters,
        kernel_sizes=[3, 5, 7],
        dropout=0.2,
        output_size=1,
        device=device
    )

    logger.info(f"Created CNN model with {model.count_parameters():,} parameters")
    return model.to(device)
