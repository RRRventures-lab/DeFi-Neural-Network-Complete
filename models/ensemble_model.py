"""
Ensemble Neural Network Model for DeFi Price Prediction

Combines LSTM, CNN, and Attention models for robust predictions.

Architecture:
- 3 base learners: LSTM, CNN, Attention
- Meta-learner: Learns optimal combination weights
- Training: End-to-end learning of all components

Benefits:
- Captures temporal (LSTM), pattern (CNN), and importance (Attention) aspects
- Reduces overfitting through model diversity
- Interpretable through component contributions
- State-of-the-art performance on time-series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble model combining LSTM, CNN, and Attention.

    Each base learner is frozen after pre-training, and a meta-learner learns
    how to optimally combine their predictions.
    """

    def __init__(
        self,
        lstm_model: nn.Module,
        cnn_model: nn.Module,
        attention_model: nn.Module,
        meta_hidden_size: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble model.

        Args:
            lstm_model: Pre-trained LSTM model
            cnn_model: Pre-trained CNN model
            attention_model: Pre-trained Attention model
            meta_hidden_size: Hidden size for meta-learner (64)
            dropout: Dropout rate (0.1)
            device: Device to run on
        """
        super(EnsembleModel, self).__init__()

        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.attention_model = attention_model
        self.device = device

        # Meta-learner: learns to combine base learner outputs
        # Input: 3 predictions + features (optional context)
        # Output: final prediction + confidence weights
        self.meta_learner = nn.Sequential(
            nn.Linear(3, meta_hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(meta_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size, meta_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(meta_hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size // 2, 1)
        )

        # Weight predictor: learns confidence weights for each base learner
        self.weight_predictor = nn.Sequential(
            nn.Linear(3, meta_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size, 3),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize meta-learner weights."""
        for module in self.meta_learner.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for module in self.weight_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            training: Whether to return individual predictions

        Returns:
            Tuple of (final_prediction, details_dict)
            details_dict contains:
                - lstm_pred: LSTM prediction
                - cnn_pred: CNN prediction
                - attention_pred: Attention prediction
                - weights: Learned combination weights
                - ensemble_pred: Ensemble prediction
        """
        # Get predictions from base learners
        with torch.no_grad():
            lstm_pred = self.lstm_model(x)
            cnn_pred = self.cnn_model(x)
            attention_pred = self.attention_model(x)

        # Stack predictions: (batch_size, 3)
        stacked_preds = torch.cat([lstm_pred, cnn_pred, attention_pred], dim=1)

        # Get learned weights
        weights = self.weight_predictor(stacked_preds)
        # weights shape: (batch_size, 3)

        # Weighted ensemble prediction
        weighted_preds = stacked_preds * weights
        ensemble_pred = weighted_preds.sum(dim=1, keepdim=True)

        # Meta-learner refinement (optional)
        meta_pred = self.meta_learner(stacked_preds)

        # Average ensemble and meta predictions
        final_pred = (ensemble_pred + meta_pred) / 2

        details = {
            'lstm_pred': lstm_pred,
            'cnn_pred': cnn_pred,
            'attention_pred': attention_pred,
            'weights': weights,
            'ensemble_pred': ensemble_pred,
            'meta_pred': meta_pred,
            'final_pred': final_pred
        }

        return final_pred, details

    def get_base_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions from each base learner.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions from each base learner
        """
        with torch.no_grad():
            lstm_pred = self.lstm_model(x)
            cnn_pred = self.cnn_model(x)
            attention_pred = self.attention_model(x)

        return {
            'lstm': lstm_pred,
            'cnn': cnn_pred,
            'attention': attention_pred
        }

    def get_confidence_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get learned confidence weights for base learners.

        Args:
            x: Input tensor

        Returns:
            Dictionary with weights for each base learner
        """
        with torch.no_grad():
            lstm_pred = self.lstm_model(x)
            cnn_pred = self.cnn_model(x)
            attention_pred = self.attention_model(x)

            stacked_preds = torch.cat([lstm_pred, cnn_pred, attention_pred], dim=1)
            weights = self.weight_predictor(stacked_preds)

        return {
            'lstm': weights[:, 0],
            'cnn': weights[:, 1],
            'attention': weights[:, 2]
        }

    def freeze_base_learners(self):
        """Freeze all base learner parameters."""
        for param in self.lstm_model.parameters():
            param.requires_grad = False
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        for param in self.attention_model.parameters():
            param.requires_grad = False

    def unfreeze_base_learners(self):
        """Unfreeze all base learner parameters."""
        for param in self.lstm_model.parameters():
            param.requires_grad = True
        for param in self.cnn_model.parameters():
            param.requires_grad = True
        for param in self.attention_model.parameters():
            param.requires_grad = True

    def count_parameters(self, include_base_learners: bool = True) -> int:
        """
        Count total parameters.

        Args:
            include_base_learners: Whether to include base learner parameters

        Returns:
            Total parameter count
        """
        count = 0

        if include_base_learners:
            count += sum(p.numel() for p in self.lstm_model.parameters() if p.requires_grad)
            count += sum(p.numel() for p in self.cnn_model.parameters() if p.requires_grad)
            count += sum(p.numel() for p in self.attention_model.parameters() if p.requires_grad)

        # Meta-learner parameters
        count += sum(p.numel() for p in self.meta_learner.parameters() if p.requires_grad)
        count += sum(p.numel() for p in self.weight_predictor.parameters() if p.requires_grad)

        return count

    def to_eval_mode(self):
        """Set all models to evaluation mode."""
        self.lstm_model.eval()
        self.cnn_model.eval()
        self.attention_model.eval()
        self.meta_learner.eval()
        self.weight_predictor.eval()
        return self

    def to_train_mode(self):
        """Set all models to training mode."""
        self.lstm_model.train()
        self.cnn_model.train()
        self.attention_model.train()
        self.meta_learner.train()
        self.weight_predictor.train()
        return self


class StackedEnsemble(nn.Module):
    """
    Stacked ensemble using multiple layers of base learners.

    First layer: multiple LSTM, CNN, and Attention models with different seeds/configs
    Second layer: meta-learner that combines first layer outputs
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        num_base_per_type: int = 3,
        meta_hidden_size: int = 128,
        device: str = 'cpu'
    ):
        """
        Initialize stacked ensemble.

        Args:
            base_models: List of base model instances
            num_base_per_type: Number of each model type
            meta_hidden_size: Hidden size for meta-learner
            device: Device to run on
        """
        super(StackedEnsemble, self).__init__()

        self.base_models = nn.ModuleList(base_models)
        self.num_models = len(base_models)
        self.device = device

        # Meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(self.num_models, meta_hidden_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(meta_hidden_size),
            nn.Dropout(0.1),
            nn.Linear(meta_hidden_size, meta_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(meta_hidden_size // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through stacked ensemble.

        Args:
            x: Input tensor

        Returns:
            Tuple of (final_prediction, individual_predictions)
        """
        predictions = []

        with torch.no_grad():
            for model in self.base_models:
                pred = model(x)
                predictions.append(pred)

        # Stack predictions
        stacked = torch.cat(predictions, dim=1)

        # Meta-learner
        final_pred = self.meta_learner(stacked)

        return final_pred, predictions

    def count_parameters(self, include_base: bool = True) -> int:
        """Count total parameters."""
        count = 0

        if include_base:
            for model in self.base_models:
                if hasattr(model, 'count_parameters'):
                    count += model.count_parameters()
                else:
                    count += sum(p.numel() for p in model.parameters() if p.requires_grad)

        count += sum(p.numel() for p in self.meta_learner.parameters() if p.requires_grad)
        return count


def create_ensemble_model(
    lstm_model: nn.Module,
    cnn_model: nn.Module,
    attention_model: nn.Module,
    device: str = 'cpu'
) -> EnsembleModel:
    """
    Factory function to create ensemble model.

    Args:
        lstm_model: Pre-trained LSTM model
        cnn_model: Pre-trained CNN model
        attention_model: Pre-trained Attention model
        device: Device to run on

    Returns:
        EnsembleModel instance
    """
    model = EnsembleModel(
        lstm_model=lstm_model,
        cnn_model=cnn_model,
        attention_model=attention_model,
        meta_hidden_size=64,
        dropout=0.1,
        device=device
    )

    logger.info(f"Created Ensemble model with {model.count_parameters():,} parameters")
    return model.to(device)
