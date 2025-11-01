"""
LSTM Neural Network Model for DeFi Price Prediction

Architecture:
- Input: 30 timesteps × 40 features
- LSTM Layers: 2 bidirectional layers with 128 hidden units
- Output: Continuous price movement prediction
- Training: Adam optimizer with learning rate decay

Performance:
- Captures long-term temporal dependencies
- Bidirectional processing for context from both directions
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory model for time-series prediction.

    Bidirectional LSTM with dropout for regularization.
    Designed for 30-timestep windows of 40 features.
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        output_size: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of features per timestep (40)
            hidden_size: LSTM hidden state dimension (128)
            num_layers: Number of LSTM layers (2)
            dropout: Dropout rate between layers (0.2)
            bidirectional: Use bidirectional LSTM (True)
            output_size: Output dimension (1 for regression)
            device: Device to run on ('cpu' or 'cuda')
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size
        self.device = device

        # Number of directions
        num_directions = 2 if bidirectional else 1

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Linear layers for output
        lstm_output_size = hidden_size * num_directions

        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 30, 40) - 32 samples, 30 timesteps, 40 features

        Returns:
            output: Prediction tensor of shape (batch_size, output_size)
        """
        # x shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)

        # Use last timestep's output for prediction
        last_output = lstm_out[:, -1, :]
        # last_output shape: (batch_size, hidden_size * num_directions)

        # Fully connected layers with dropout
        fc1_out = F.relu(self.fc1(last_output))
        # fc1_out shape: (batch_size, hidden_size)

        fc1_out = self.batch_norm(fc1_out)
        fc1_out = self.dropout_layer(fc1_out)

        output = self.fc2(fc1_out)
        # output shape: (batch_size, output_size)

        return output

    def get_hidden_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get hidden states at each timestep (useful for attention mechanism).

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tuple of (hidden states, final hidden state, final cell state)
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out, hidden, cell

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layers(self, num_layers_to_freeze: int):
        """
        Freeze the first N layers for transfer learning.

        Args:
            num_layers_to_freeze: Number of layers to freeze from the bottom
        """
        for i, (name, param) in enumerate(self.named_parameters()):
            if i < num_layers_to_freeze:
                param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True

    def to_eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()
        return self

    def to_train_mode(self):
        """Set model to training mode."""
        self.train()
        return self


class LSTMEnsemble(nn.Module):
    """
    Multiple LSTM models for ensemble prediction.

    Useful for creating different model configurations for voting/averaging.
    """

    def __init__(
        self,
        num_models: int = 3,
        input_size: int = 40,
        hidden_sizes: list = None,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble of LSTM models.

        Args:
            num_models: Number of LSTM models in ensemble
            input_size: Number of features per timestep
            hidden_sizes: List of hidden sizes for each model
            device: Device to run on
        """
        super(LSTMEnsemble, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128] * num_models

        self.num_models = num_models
        self.models = nn.ModuleList([
            LSTMModel(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=2,
                dropout=0.2,
                bidirectional=True,
                output_size=1,
                device=device
            )
            for i in range(num_models)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all models.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tuple of (ensemble_prediction, individual_predictions)
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Stack predictions: (num_models, batch_size, output_size)
        stacked = torch.stack(predictions, dim=0)

        # Average across models: (batch_size, output_size)
        ensemble_pred = stacked.mean(dim=0)

        return ensemble_pred, stacked

    def count_total_parameters(self) -> int:
        """Count total parameters across all models."""
        return sum(model.count_parameters() for model in self.models)


def create_lstm_model(
    input_size: int = 40,
    hidden_size: int = 128,
    device: str = 'cpu'
) -> LSTMModel:
    """
    Factory function to create LSTM model.

    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden dimension
        device: Device to run on

    Returns:
        LSTMModel instance
    """
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        output_size=1,
        device=device
    )

    logger.info(f"Created LSTM model with {model.count_parameters():,} parameters")
    return model.to(device)
