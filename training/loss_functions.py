"""
Loss Functions for DeFi Price Prediction

Includes standard and custom loss functions optimized for financial time-series:
- MSE: Mean Squared Error (baseline regression)
- MAE: Mean Absolute Error (robust to outliers)
- Huber: Huber loss (combination of MSE and MAE)
- Sharpe Ratio Loss: Maximizes risk-adjusted returns
- Quantile Loss: For specific percentile predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss - Standard regression loss.

    Penalizes large errors quadratically.
    Good for Gaussian-distributed errors.
    """

    def __init__(self):
        """Initialize MSE loss."""
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Target values (batch_size, 1)

        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss - Robust to outliers.

    Linear penalty for errors.
    Less affected by extreme values than MSE.
    """

    def __init__(self):
        """Initialize MAE loss."""
        super(MAELoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate MAE loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Target values (batch_size, 1)

        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class HuberLoss(nn.Module):
    """
    Huber Loss - Combines MSE and MAE.

    Quadratic for small errors, linear for large errors.
    Robust to outliers while preserving gradient stability.
    """

    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.

        Args:
            delta: Threshold between quadratic and linear regions
        """
        super(HuberLoss, self).__init__()
        self.criterion = nn.HuberLoss(delta=delta)
        self.delta = delta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Huber loss.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class SharpeRatioLoss(nn.Module):
    """
    Sharpe Ratio Loss - Maximizes risk-adjusted returns.

    Loss = -Sharpe Ratio = -(mean_return / std_return)

    Encourages models to maximize return per unit of risk.
    Optimal for financial applications.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize Sharpe ratio loss.

        Args:
            epsilon: Small value for numerical stability
        """
        super(SharpeRatioLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Target values (batch_size, 1)

        Returns:
            Negative Sharpe ratio (scalar)
        """
        # Calculate prediction errors
        errors = predictions - targets

        # Mean return (average prediction error)
        mean_error = torch.mean(errors)

        # Return volatility (std of prediction errors)
        std_error = torch.std(errors) + self.epsilon

        # Sharpe ratio
        sharpe = mean_error / std_error

        # Return negative Sharpe (for minimization)
        return -sharpe


class QuantileLoss(nn.Module):
    """
    Quantile Loss - For specific percentile predictions.

    Asymmetric loss that allows modeling conditional quantiles.
    Useful for predicting confidence intervals.
    """

    def __init__(self, quantile: float = 0.5):
        """
        Initialize quantile loss.

        Args:
            quantile: Quantile level (0 < quantile < 1)
        """
        super(QuantileLoss, self).__init__()
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        self.quantile = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate quantile loss.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar loss value
        """
        errors = targets - predictions
        weights = torch.where(
            errors > 0,
            torch.tensor(self.quantile, device=errors.device),
            torch.tensor(self.quantile - 1.0, device=errors.device)
        )
        loss = torch.mean(weights * errors.abs())
        return loss


class DirectionalAccuracyLoss(nn.Module):
    """
    Directional Accuracy Loss - Penalizes directional mistakes.

    High penalty when prediction and actual move in opposite directions.
    Good for trading signals (buy/sell classification).
    """

    def __init__(self, mse_weight: float = 0.5):
        """
        Initialize directional accuracy loss.

        Args:
            mse_weight: Weight of MSE component (0 to 1)
        """
        super(DirectionalAccuracyLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mse_criterion = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate directional accuracy loss.

        Combination of:
        1. MSE for magnitude accuracy
        2. Directional penalty for sign mistakes

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar loss value
        """
        # MSE component
        mse_loss = self.mse_criterion(predictions, targets)

        # Directional component
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        direction_match = (pred_direction == target_direction).float()
        directional_loss = 1.0 - direction_match.mean()

        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            (1 - self.mse_weight) * directional_loss
        )

        return total_loss


class ReturnVolatilityLoss(nn.Module):
    """
    Return-Volatility Loss - Balances absolute return with volatility.

    Loss = -mean_return + lambda * volatility

    Encourages high returns with controlled volatility.
    """

    def __init__(self, lambda_volatility: float = 0.5):
        """
        Initialize return-volatility loss.

        Args:
            lambda_volatility: Weight of volatility component
        """
        super(ReturnVolatilityLoss, self).__init__()
        self.lambda_volatility = lambda_volatility

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate return-volatility loss.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar loss value
        """
        # Calculate returns (prediction errors viewed as returns)
        returns = predictions - targets

        # Mean absolute return
        mean_return = torch.mean(torch.abs(returns))

        # Volatility (standard deviation of returns)
        volatility = torch.std(returns)

        # Loss = -mean_return + lambda * volatility
        loss = -mean_return + self.lambda_volatility * volatility

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss - Weighted combination of multiple losses.

    Allows flexible loss design for multi-objective optimization.
    """

    def __init__(
        self,
        losses: dict,
        weights: dict
    ):
        """
        Initialize combined loss.

        Args:
            losses: Dict of {name: loss_module}
            weights: Dict of {name: weight}
        """
        super(CombinedLoss, self).__init__()

        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, loss in losses.items():
            self.losses[name] = loss
            self.weights[name] = weights.get(name, 1.0)

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted combination of losses.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar combined loss
        """
        total_loss = torch.tensor(0.0, device=predictions.device)

        for name, loss_module in self.losses.items():
            loss_value = loss_module(predictions, targets)
            weight = self.weights[name]
            total_loss = total_loss + weight * loss_value

        return total_loss

    def get_individual_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Get individual loss components.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Dict of {loss_name: loss_value}
        """
        losses = {}
        for name, loss_module in self.losses.items():
            losses[name] = loss_module(predictions, targets).item()
        return losses


def create_loss_function(
    loss_type: str = 'mse',
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss ('mse', 'mae', 'huber', 'sharpe', 'quantile',
                   'directional', 'return_volatility', 'combined')
        **kwargs: Additional arguments for specific loss types

    Returns:
        Loss module instance
    """
    loss_type = loss_type.lower()

    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'mae':
        return MAELoss()
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        return HuberLoss(delta=delta)
    elif loss_type == 'sharpe':
        epsilon = kwargs.get('epsilon', 1e-8)
        return SharpeRatioLoss(epsilon=epsilon)
    elif loss_type == 'quantile':
        quantile = kwargs.get('quantile', 0.5)
        return QuantileLoss(quantile=quantile)
    elif loss_type == 'directional':
        mse_weight = kwargs.get('mse_weight', 0.5)
        return DirectionalAccuracyLoss(mse_weight=mse_weight)
    elif loss_type == 'return_volatility':
        lambda_vol = kwargs.get('lambda_volatility', 0.5)
        return ReturnVolatilityLoss(lambda_volatility=lambda_vol)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Preset loss combinations for common scenarios
LOSS_PRESETS = {
    'regression': {
        'loss': create_loss_function('mse'),
        'description': 'Simple MSE for baseline regression'
    },
    'robust': {
        'loss': create_loss_function('huber', delta=1.0),
        'description': 'Huber loss for outlier robustness'
    },
    'financial': {
        'loss': create_loss_function('sharpe'),
        'description': 'Sharpe ratio maximization'
    },
    'directional': {
        'loss': create_loss_function('directional', mse_weight=0.6),
        'description': 'Focus on directional accuracy'
    },
    'volatility_aware': {
        'loss': create_loss_function('return_volatility', lambda_volatility=0.3),
        'description': 'Balance returns with volatility control'
    }
}
