"""
Portfolio Optimization Module

Implements Modern Portfolio Theory (MPT) optimization including:
- Efficient frontier calculation
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Risk-constrained optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, LinearConstraint, Bounds

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for portfolio optimization results."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    allocation: Dict[str, float]
    optimization_method: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': self.weights.tolist(),
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'allocation': self.allocation,
            'optimization_method': self.optimization_method
        }


class EfficientFrontier:
    """
    Compute the efficient frontier using historical returns.

    Modern Portfolio Theory optimization for finding optimal asset allocations.
    """

    def __init__(
        self,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize efficient frontier.

        Args:
            returns: Asset returns (samples × assets)
            prices: Optional historical prices for alternative calculations
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns.astype(np.float32)
        self.prices = prices
        self.risk_free_rate = risk_free_rate
        self.num_assets = returns.shape[1]

        # Calculate statistics
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns.T)
        self.std_devs = np.std(returns, axis=0)

        logger.info(f"Efficient frontier initialized for {self.num_assets} assets")

    def portfolio_performance(
        self,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.

        Args:
            weights: Portfolio weights

        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        returns = np.sum(self.mean_returns * weights) * 252
        volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0

        return returns, volatility, sharpe

    def minimum_variance_portfolio(self) -> OptimizationResult:
        """
        Find the minimum variance portfolio.

        Returns:
            OptimizationResult with minimum variance allocation
        """
        def variance(w):
            return np.dot(w, np.dot(self.cov_matrix, w))

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            variance,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        allocation = {f'asset_{i}': w for i, w in enumerate(weights)}

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            allocation=allocation,
            optimization_method='minimum_variance'
        )

    def maximum_sharpe_portfolio(self) -> OptimizationResult:
        """
        Find the maximum Sharpe ratio portfolio.

        Returns:
            OptimizationResult with maximum Sharpe allocation
        """
        def neg_sharpe(w):
            ret, vol, sharpe = self.portfolio_performance(w)
            return -sharpe if vol > 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            neg_sharpe,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        allocation = {f'asset_{i}': w for i, w in enumerate(weights)}

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            allocation=allocation,
            optimization_method='maximum_sharpe'
        )

    def risk_parity_portfolio(self) -> OptimizationResult:
        """
        Find the risk parity portfolio (equal risk contribution).

        Returns:
            OptimizationResult with risk parity allocation
        """
        def risk_contribution(w):
            portfolio_vol = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
            marginal_contrib = np.dot(self.cov_matrix, w) / portfolio_vol
            contrib = w * marginal_contrib
            return contrib

        def rp_objective(w):
            contrib = risk_contribution(w)
            target = np.sum(contrib) / self.num_assets
            return np.sum((contrib - target) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            rp_objective,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        allocation = {f'asset_{i}': w for i, w in enumerate(weights)}

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            allocation=allocation,
            optimization_method='risk_parity'
        )

    def target_return_portfolio(
        self,
        target_return: float
    ) -> OptimizationResult:
        """
        Find minimum variance portfolio with target return.

        Args:
            target_return: Target annual return (0.1 = 10%)

        Returns:
            OptimizationResult with target return allocation
        """
        def variance(w):
            return np.dot(w, np.dot(self.cov_matrix, w))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(self.mean_returns * w) * 252 - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            variance,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Target return {target_return} may be infeasible")

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        allocation = {f'asset_{i}': w for i, w in enumerate(weights)}

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            allocation=allocation,
            optimization_method='target_return'
        )

    def target_volatility_portfolio(
        self,
        target_vol: float
    ) -> OptimizationResult:
        """
        Find maximum Sharpe portfolio with target volatility.

        Args:
            target_vol: Target annual volatility (0.15 = 15%)

        Returns:
            OptimizationResult with target volatility allocation
        """
        def neg_sharpe(w):
            ret, vol, sharpe = self.portfolio_performance(w)
            return -sharpe if vol > 0 else 0

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w, np.dot(self.cov_matrix, w))) * np.sqrt(252) - target_vol}
        ]
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1.0 / self.num_assets] * self.num_assets)

        result = minimize(
            neg_sharpe,
            initial,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Target volatility {target_vol} may be infeasible")

        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)

        allocation = {f'asset_{i}': w for i, w in enumerate(weights)}

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            allocation=allocation,
            optimization_method='target_volatility'
        )

    def efficient_frontier_points(
        self,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate efficient frontier points.

        Args:
            num_points: Number of points to generate

        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        min_ret = np.min(self.mean_returns * 252)
        max_ret = np.max(self.mean_returns * 252)

        target_returns = np.linspace(min_ret, max_ret, num_points)

        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []

        for target in target_returns:
            try:
                result = self.target_return_portfolio(target)
                frontier_returns.append(result.expected_return)
                frontier_vols.append(result.volatility)
                frontier_sharpes.append(result.sharpe_ratio)
            except Exception as e:
                logger.debug(f"Failed to compute point for return {target}: {e}")
                continue

        return (
            np.array(frontier_returns),
            np.array(frontier_vols),
            np.array(frontier_sharpes)
        )


class PortfolioOptimizer:
    """
    High-level portfolio optimization interface combining multiple strategies.
    """

    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.

        Args:
            returns: Asset returns (samples × assets)
            risk_free_rate: Annual risk-free rate
        """
        self.frontier = EfficientFrontier(returns, risk_free_rate=risk_free_rate)
        self.returns = returns
        self.risk_free_rate = risk_free_rate

    def optimize(
        self,
        method: str = 'maximum_sharpe',
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.

        Args:
            method: Optimization method ('maximum_sharpe', 'minimum_variance',
                    'risk_parity', 'target_return', 'target_volatility')
            **kwargs: Additional arguments for specific methods

        Returns:
            OptimizationResult with optimized allocation
        """
        if method == 'maximum_sharpe':
            return self.frontier.maximum_sharpe_portfolio()
        elif method == 'minimum_variance':
            return self.frontier.minimum_variance_portfolio()
        elif method == 'risk_parity':
            return self.frontier.risk_parity_portfolio()
        elif method == 'target_return':
            target_return = kwargs.get('target_return', 0.1)
            return self.frontier.target_return_portfolio(target_return)
        elif method == 'target_volatility':
            target_vol = kwargs.get('target_vol', 0.15)
            return self.frontier.target_volatility_portfolio(target_vol)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all optimization strategies.

        Returns:
            DataFrame with comparison of all strategies
        """
        results = {
            'Maximum Sharpe': self.frontier.maximum_sharpe_portfolio(),
            'Minimum Variance': self.frontier.minimum_variance_portfolio(),
            'Risk Parity': self.frontier.risk_parity_portfolio(),
        }

        comparison_data = []
        for strategy, result in results.items():
            comparison_data.append({
                'Strategy': strategy,
                'Return': result.expected_return,
                'Volatility': result.volatility,
                'Sharpe Ratio': result.sharpe_ratio
            })

        return pd.DataFrame(comparison_data)
