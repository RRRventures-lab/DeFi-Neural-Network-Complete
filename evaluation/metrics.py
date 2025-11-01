"""
Performance Metrics for Backtesting

Calculates comprehensive metrics for evaluating trading models:
- Return metrics: Total, annualized, Sharpe ratio
- Risk metrics: Volatility, max drawdown, Sortino ratio
- Trading metrics: Win rate, profit factor
- Statistical: Skewness, kurtosis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    """

    def __init__(
        self,
        returns: np.ndarray,
        predictions: np.ndarray,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize metrics calculator.

        Args:
            returns: Array of returns (daily percentage changes)
            predictions: Array of model predictions
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns.flatten()
        self.predictions = predictions.flatten()
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252  # Standard trading days per year

    def calculate_all(self) -> Dict:
        """
        Calculate all metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {
            'return_metrics': self._calculate_return_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'trading_metrics': self._calculate_trading_metrics(),
            'statistical_metrics': self._calculate_statistical_metrics()
        }
        return metrics

    def _calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics."""
        total_return = np.sum(self.returns)
        cumulative_return = (1 + self.returns).prod() - 1

        # Annualized return
        num_years = len(self.returns) / self.trading_days
        if num_years > 0:
            annualized_return = (1 + cumulative_return) ** (1 / num_years) - 1
        else:
            annualized_return = 0

        # Average daily return
        avg_daily_return = np.mean(self.returns)

        # Sharpe ratio
        daily_volatility = np.std(self.returns)
        daily_risk_free_rate = self.risk_free_rate / self.trading_days

        if daily_volatility > 0:
            sharpe_ratio = (avg_daily_return - daily_risk_free_rate) / daily_volatility * np.sqrt(self.trading_days)
        else:
            sharpe_ratio = 0

        return {
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'avg_daily_return': avg_daily_return,
            'sharpe_ratio': sharpe_ratio
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics."""
        # Volatility
        volatility = np.std(self.returns)
        annualized_volatility = volatility * np.sqrt(self.trading_days)

        # Maximum drawdown
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Sortino ratio (downside deviation)
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
            daily_risk_free = self.risk_free_rate / self.trading_days
            sortino_ratio = (np.mean(self.returns) - daily_risk_free) / downside_deviation * np.sqrt(self.trading_days)
        else:
            sortino_ratio = 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = np.mean(self.returns[self.returns <= var_95])

        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }

    def _calculate_trading_metrics(self) -> Dict:
        """Calculate trading-specific metrics."""
        # Directional accuracy
        predictions_direction = np.sign(self.predictions)
        returns_direction = np.sign(self.returns)
        accuracy = np.mean(predictions_direction == returns_direction)

        # Win rate
        wins = np.sum(self.returns > 0)
        total_trades = len(self.returns)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Average win and loss
        positive_returns = self.returns[self.returns > 0]
        negative_returns = self.returns[self.returns < 0]

        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

        # Profit factor
        total_gains = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        total_losses = np.abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
        profit_factor = total_gains / total_losses if total_losses > 0 else 0

        return {
            'directional_accuracy': accuracy,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': total_trades
        }

    def _calculate_statistical_metrics(self) -> Dict:
        """Calculate statistical metrics."""
        skewness = pd.Series(self.returns).skew()
        kurtosis = pd.Series(self.returns).kurtosis()

        # Correlation between predictions and returns
        correlation = np.corrcoef(self.predictions, self.returns)[0, 1]

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'prediction_correlation': correlation
        }

    def get_summary(self) -> Dict:
        """Get summary of key metrics."""
        all_metrics = self.calculate_all()

        return {
            'total_return': all_metrics['return_metrics']['total_return'],
            'annualized_return': all_metrics['return_metrics']['annualized_return'],
            'sharpe_ratio': all_metrics['return_metrics']['sharpe_ratio'],
            'max_drawdown': all_metrics['risk_metrics']['max_drawdown'],
            'volatility': all_metrics['risk_metrics']['annualized_volatility'],
            'win_rate': all_metrics['trading_metrics']['win_rate'],
            'profit_factor': all_metrics['trading_metrics']['profit_factor'],
            'directional_accuracy': all_metrics['trading_metrics']['directional_accuracy']
        }


class MetricsComparer:
    """
    Compare metrics across multiple models or strategies.
    """

    def __init__(self, metrics_dict: Dict[str, Dict]):
        """
        Initialize comparer.

        Args:
            metrics_dict: Dict of {model_name: metrics_dict}
        """
        self.metrics_dict = metrics_dict

    def compare_returns(self) -> pd.DataFrame:
        """Compare return metrics."""
        data = {}
        for name, metrics in self.metrics_dict.items():
            returns = metrics['return_metrics']
            data[name] = {
                'Total Return': returns['total_return'],
                'Annualized Return': returns['annualized_return'],
                'Sharpe Ratio': returns['sharpe_ratio']
            }
        return pd.DataFrame(data).T

    def compare_risk(self) -> pd.DataFrame:
        """Compare risk metrics."""
        data = {}
        for name, metrics in self.metrics_dict.items():
            risk = metrics['risk_metrics']
            data[name] = {
                'Volatility': risk['annualized_volatility'],
                'Max Drawdown': risk['max_drawdown'],
                'Sortino Ratio': risk['sortino_ratio'],
                'VaR 95%': risk['var_95']
            }
        return pd.DataFrame(data).T

    def compare_trading(self) -> pd.DataFrame:
        """Compare trading metrics."""
        data = {}
        for name, metrics in self.metrics_dict.items():
            trading = metrics['trading_metrics']
            data[name] = {
                'Win Rate': trading['win_rate'],
                'Profit Factor': trading['profit_factor'],
                'Accuracy': trading['directional_accuracy'],
                'Num Trades': trading['num_trades']
            }
        return pd.DataFrame(data).T

    def rank_by_sharpe(self) -> pd.Series:
        """Rank models by Sharpe ratio."""
        sharpe_ratios = {}
        for name, metrics in self.metrics_dict.items():
            sharpe_ratios[name] = metrics['return_metrics']['sharpe_ratio']
        return pd.Series(sharpe_ratios).sort_values(ascending=False)

    def rank_by_return(self) -> pd.Series:
        """Rank models by annualized return."""
        returns = {}
        for name, metrics in self.metrics_dict.items():
            returns[name] = metrics['return_metrics']['annualized_return']
        return pd.Series(returns).sort_values(ascending=False)

    def rank_by_risk_adjusted(self) -> pd.Series:
        """Rank models by risk-adjusted return (Sharpe/Sortino)."""
        scores = {}
        for name, metrics in self.metrics_dict.items():
            sharpe = metrics['return_metrics']['sharpe_ratio']
            sortino = metrics['risk_metrics']['sortino_ratio']
            scores[name] = (sharpe + sortino) / 2
        return pd.Series(scores).sort_values(ascending=False)


def calculate_cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative returns from daily returns.

    Args:
        returns: Array of daily returns

    Returns:
        Cumulative return at each timestep
    """
    return np.cumprod(1 + returns) - 1


def calculate_drawdown(returns: np.ndarray) -> np.ndarray:
    """
    Calculate drawdown from daily returns.

    Args:
        returns: Array of daily returns

    Returns:
        Drawdown at each timestep
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def calculate_rolling_sharpe(
    returns: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.02
) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Array of daily returns
        window: Rolling window size (default 252 = 1 year)
        risk_free_rate: Annual risk-free rate

    Returns:
        Rolling Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    rolling_sharpe = []

    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        mean_return = np.mean(window_returns)
        volatility = np.std(window_returns)

        if volatility > 0:
            sharpe = (mean_return - daily_rf) / volatility * np.sqrt(252)
        else:
            sharpe = 0

        rolling_sharpe.append(sharpe)

    return np.array(rolling_sharpe)
