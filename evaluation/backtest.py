"""
Backtesting Engine for Trading Strategies

Implements:
- Walk-forward backtesting
- Portfolio simulation
- Trade execution
- Performance tracking
- Position management
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Container for backtest results.
    """

    def __init__(
        self,
        returns: np.ndarray,
        predictions: np.ndarray,
        trades: List[Dict],
        positions: np.ndarray,
        equity_curve: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ):
        """
        Initialize backtest result.

        Args:
            returns: Array of returns
            predictions: Array of model predictions
            trades: List of trade dictionaries
            positions: Array of positions (1 for long, -1 for short, 0 for flat)
            equity_curve: Portfolio equity over time
            timestamps: Optional timestamps for returns
        """
        self.returns = returns
        self.predictions = predictions
        self.trades = trades
        self.positions = positions
        self.equity_curve = equity_curve
        self.timestamps = timestamps

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        df = pd.DataFrame({
            'returns': self.returns,
            'predictions': self.predictions,
            'positions': self.positions,
            'equity': self.equity_curve
        })

        if self.timestamps is not None:
            df['timestamp'] = self.timestamps

        return df


class Backtest:
    """
    Walk-forward backtesting engine.

    Simulates trading strategy with position management and slippage.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 1.0,
        max_positions: int = 1,
        transaction_cost: float = 0.001
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            position_size: Size of each position (0-1)
            max_positions: Maximum concurrent positions
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost

        self.equity = initial_capital
        self.trades = []
        self.equity_curve = [initial_capital]

    def run(
        self,
        returns: np.ndarray,
        predictions: np.ndarray,
        prediction_threshold: float = 0.0
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            returns: Array of asset returns
            predictions: Array of model predictions
            prediction_threshold: Threshold for position entry (default 0 = any positive)

        Returns:
            BacktestResult object
        """
        returns = returns.flatten()
        predictions = predictions.flatten()

        assert len(returns) == len(predictions), "Returns and predictions must have same length"

        positions = np.zeros(len(returns))
        equity_curve = [self.equity]
        trades = []
        current_position = 0

        logger.info("Starting backtest...")

        for i in range(len(returns)):
            # Position decision
            prediction = predictions[i]

            # Entry logic
            if current_position == 0:
                if prediction > prediction_threshold:
                    current_position = 1  # Go long
                    trades.append({
                        'timestamp': i,
                        'type': 'entry',
                        'direction': 'long',
                        'size': self.position_size
                    })
                elif prediction < -prediction_threshold:
                    current_position = -1  # Go short
                    trades.append({
                        'timestamp': i,
                        'type': 'entry',
                        'direction': 'short',
                        'size': self.position_size
                    })

            # Exit logic: exit if prediction changes sign or drops below threshold
            elif current_position == 1 and prediction < prediction_threshold:
                trades.append({
                    'timestamp': i,
                    'type': 'exit',
                    'direction': 'long',
                    'size': self.position_size
                })
                current_position = 0
            elif current_position == -1 and prediction > -prediction_threshold:
                trades.append({
                    'timestamp': i,
                    'type': 'exit',
                    'direction': 'short',
                    'size': self.position_size
                })
                current_position = 0

            # Calculate P&L
            position_return = current_position * returns[i]
            transaction_cost = self._calculate_transaction_cost(i, trades)

            # Update equity
            self.equity *= (1 + position_return - transaction_cost)
            equity_curve.append(self.equity)
            positions[i] = current_position

        logger.info(f"Backtest completed. Final equity: ${self.equity:,.2f}")

        return BacktestResult(
            returns=returns,
            predictions=predictions,
            trades=trades,
            positions=positions,
            equity_curve=np.array(equity_curve)
        )

    def _calculate_transaction_cost(self, timestamp: int, trades: List[Dict]) -> float:
        """Calculate transaction costs at current timestamp."""
        # Check if there's a trade at this timestamp
        if trades and trades[-1]['timestamp'] == timestamp:
            return self.transaction_cost * self.position_size
        return 0

    def reset(self):
        """Reset backtest state."""
        self.equity = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]


class WalkForwardBacktest:
    """
    Walk-forward backtesting framework.

    Tests model on sequential out-of-sample periods.
    """

    def __init__(
        self,
        model,
        data_loaders: List[Tuple],
        device: str = 'cpu',
        initial_capital: float = 100000
    ):
        """
        Initialize walk-forward backtest.

        Args:
            model: Trained neural network model
            data_loaders: List of (train_loader, val_loader) tuples
            device: Device to run model on
            initial_capital: Starting capital
        """
        self.model = model
        self.data_loaders = data_loaders
        self.device = device
        self.initial_capital = initial_capital
        self.results = []

    def run(self, prediction_threshold: float = 0.0) -> Dict:
        """
        Run walk-forward backtest.

        Args:
            prediction_threshold: Threshold for position entry

        Returns:
            Dictionary with results for each fold
        """
        logger.info(f"Starting walk-forward backtest with {len(self.data_loaders)} folds")

        for fold_idx, (train_loader, val_loader) in enumerate(self.data_loaders):
            logger.info(f"Fold {fold_idx + 1}/{len(self.data_loaders)}")

            # Get validation data
            predictions = []
            returns = []

            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device).float()

                    pred = self.model(X_batch)
                    predictions.extend(pred.cpu().numpy().flatten())
                    returns.extend(y_batch.cpu().numpy().flatten())

            predictions = np.array(predictions)
            returns = np.array(returns)

            # Run backtest
            backtest = Backtest(initial_capital=self.initial_capital)
            result = backtest.run(returns, predictions, prediction_threshold)

            self.results.append({
                'fold': fold_idx,
                'result': result,
                'final_equity': backtest.equity
            })

        return self._summarize_results()

    def _summarize_results(self) -> Dict:
        """Summarize results across all folds."""
        summary = {
            'num_folds': len(self.results),
            'folds': []
        }

        total_return = 1.0
        for result_dict in self.results:
            fold_result = result_dict['result']
            final_equity = result_dict['final_equity']
            fold_return = (final_equity - self.initial_capital) / self.initial_capital

            total_return *= (1 + fold_return)

            summary['folds'].append({
                'fold': result_dict['fold'],
                'final_equity': final_equity,
                'return': fold_return,
                'num_trades': len(fold_result.trades)
            })

        summary['total_return'] = total_return - 1
        summary['avg_fold_return'] = np.mean([f['return'] for f in summary['folds']])

        return summary


class BenchmarkComparison:
    """
    Compare strategy against benchmarks.
    """

    def __init__(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray):
        """
        Initialize benchmark comparison.

        Args:
            strategy_returns: Strategy daily returns
            benchmark_returns: Benchmark daily returns
        """
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns

    def excess_return(self) -> float:
        """Calculate excess return over benchmark."""
        strategy_total = np.sum(self.strategy_returns)
        benchmark_total = np.sum(self.benchmark_returns)
        return strategy_total - benchmark_total

    def information_ratio(self) -> float:
        """
        Calculate information ratio.

        IR = (strategy return - benchmark return) / tracking error
        """
        excess = self.strategy_returns - self.benchmark_returns
        mean_excess = np.mean(excess)
        std_excess = np.std(excess)

        if std_excess > 0:
            return mean_excess / std_excess * np.sqrt(252)
        return 0

    def batting_average(self) -> float:
        """
        Calculate percentage of periods beating benchmark.
        """
        wins = np.sum(self.strategy_returns > self.benchmark_returns)
        total = len(self.strategy_returns)
        return wins / total if total > 0 else 0

    def get_comparison(self) -> Dict:
        """Get full comparison."""
        return {
            'excess_return': self.excess_return(),
            'information_ratio': self.information_ratio(),
            'batting_average': self.batting_average(),
            'strategy_return': np.sum(self.strategy_returns),
            'benchmark_return': np.sum(self.benchmark_returns)
        }


def simulate_trading(
    predictions: np.ndarray,
    returns: np.ndarray,
    entry_threshold: float = 0.0,
    exit_threshold: float = 0.0,
    initial_capital: float = 100000
) -> Tuple[np.ndarray, float]:
    """
    Simulate trading strategy.

    Args:
        predictions: Model predictions
        returns: Actual returns
        entry_threshold: Threshold for entering position
        exit_threshold: Threshold for exiting position
        initial_capital: Starting capital

    Returns:
        Tuple of (equity_curve, final_equity)
    """
    equity = initial_capital
    equity_curve = [equity]
    position = 0

    for i in range(len(predictions)):
        # Entry/exit logic
        if position == 0 and predictions[i] > entry_threshold:
            position = 1
        elif position == 1 and predictions[i] < exit_threshold:
            position = 0

        # Calculate return
        period_return = position * returns[i]
        equity *= (1 + period_return)
        equity_curve.append(equity)

    return np.array(equity_curve), equity
