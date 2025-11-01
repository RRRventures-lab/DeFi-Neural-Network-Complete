"""
Portfolio Manager Module

Integrates all risk management components for comprehensive portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .portfolio_optimization import PortfolioOptimizer, OptimizationResult
from .risk_limits import (
    DrawdownLimit, ConcentrationLimit, VolatilityLimit, VaRLimit
)
from .position_sizing import AdaptivePositionSizer
from .hedging import HedgingManager

logger = logging.getLogger(__name__)


@dataclass
class RiskAdjustedAllocation:
    """Container for risk-adjusted allocation."""
    weights: np.ndarray
    allocation: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    position_sizes: Dict[str, float]
    risk_metrics: Dict
    hedges: Optional[Dict] = None
    confidence: float = 0.90

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': self.weights.tolist(),
            'allocation': self.allocation,
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'position_sizes': self.position_sizes,
            'risk_metrics': self.risk_metrics,
            'hedges': self.hedges,
            'confidence': self.confidence
        }


class PortfolioManager:
    """
    Comprehensive portfolio management system.

    Combines optimization, risk limits, position sizing, and hedging.
    """

    def __init__(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize portfolio manager.

        Args:
            returns: Historical returns (samples × assets)
            asset_names: Optional list of asset names
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.num_assets = returns.shape[1]

        if asset_names is None:
            self.asset_names = [f'asset_{i}' for i in range(self.num_assets)]
        else:
            self.asset_names = asset_names

        # Initialize components
        self.optimizer = PortfolioOptimizer(returns, risk_free_rate)
        self.position_sizer = AdaptivePositionSizer()
        self.hedging_manager = HedgingManager()

        # Initialize risk limits
        self.drawdown_limit = DrawdownLimit(max_drawdown=-0.25)
        self.concentration_limit = ConcentrationLimit(max_concentration=0.15)
        self.volatility_limit = VolatilityLimit(max_volatility=0.20)
        self.var_limit = VaRLimit(max_var=-0.05)

        logger.info(f"Portfolio manager initialized for {self.num_assets} assets")

    def get_optimal_allocation(
        self,
        method: str = 'maximum_sharpe',
        **kwargs
    ) -> RiskAdjustedAllocation:
        """
        Get optimal allocation.

        Args:
            method: Optimization method
            **kwargs: Additional arguments

        Returns:
            RiskAdjustedAllocation
        """
        # Optimize
        opt_result = self.optimizer.optimize(method=method, **kwargs)

        # Check concentration limits
        concentration_violations = self.concentration_limit.check_allocation(opt_result.weights)

        # Adjust if needed
        if concentration_violations:
            logger.warning(f"Concentration limit violated, adjusting allocation")
            weights = self._apply_concentration_limit(opt_result.weights)
        else:
            weights = opt_result.weights

        # Calculate position sizes
        account_equity = 100000  # Default account size
        position_sizes = self._calculate_position_sizes(weights, account_equity)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(self.returns, weights)

        allocation = {name: w for name, w in zip(self.asset_names, weights)}

        return RiskAdjustedAllocation(
            weights=weights,
            allocation=allocation,
            expected_return=opt_result.expected_return,
            volatility=opt_result.volatility,
            sharpe_ratio=opt_result.sharpe_ratio,
            position_sizes=position_sizes,
            risk_metrics=risk_metrics
        )

    def _apply_concentration_limit(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply concentration limit to weights.

        Args:
            weights: Original weights

        Returns:
            Adjusted weights
        """
        max_weight = self.concentration_limit.max_concentration

        adjusted = np.copy(weights)
        adjusted[adjusted > max_weight] = max_weight

        # Renormalize
        adjusted = adjusted / np.sum(adjusted)

        return adjusted

    def _calculate_position_sizes(
        self,
        weights: np.ndarray,
        account_equity: float
    ) -> Dict[str, float]:
        """
        Calculate position sizes in units.

        Args:
            weights: Portfolio weights
            account_equity: Total account equity

        Returns:
            Dict of {asset: position_size}
        """
        # Simple calculation: allocate equity proportionally
        # In practice, would consider actual prices
        position_sizes = {}

        for i, name in enumerate(self.asset_names):
            allocation_value = account_equity * weights[i]
            position_sizes[name] = allocation_value  # Assuming unit price = 1

        return position_sizes

    def _calculate_risk_metrics(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> Dict:
        """
        Calculate risk metrics for portfolio.

        Args:
            returns: Historical returns
            weights: Portfolio weights

        Returns:
            Dict with risk metrics
        """
        # Calculate portfolio returns
        portfolio_returns = returns @ weights

        return {
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'max_drawdown': self.drawdown_limit.calculate_drawdown(portfolio_returns).min(),
            'var': self.var_limit.calculate_var(portfolio_returns),
            'cvar': self.var_limit.calculate_cvar(portfolio_returns),
            'skewness': pd.Series(portfolio_returns).skew(),
            'kurtosis': pd.Series(portfolio_returns).kurtosis()
        }

    def analyze_risk(
        self,
        weights: np.ndarray
    ) -> Dict:
        """
        Comprehensive risk analysis.

        Args:
            weights: Portfolio weights

        Returns:
            Detailed risk analysis
        """
        portfolio_returns = self.returns @ weights

        analysis = {
            'summary': {
                'num_assets': self.num_assets,
                'num_positions': np.sum(weights > 0.001)
            },
            'concentration': {
                'top_3': self.concentration_limit.get_top_positions(weights, 3),
                'max_position': np.max(weights),
                'herfindahl': np.sum(weights ** 2)
            },
            'drawdown': {
                'max_drawdown': self.drawdown_limit.calculate_drawdown(portfolio_returns).min(),
                'remaining_budget': self.drawdown_limit.remaining_drawdown_budget(portfolio_returns)
            },
            'volatility': {
                'portfolio_vol': np.std(portfolio_returns) * np.sqrt(252),
                'remaining_budget': self.volatility_limit.get_volatility_budget(portfolio_returns)
            },
            'tail_risk': {
                'var': self.var_limit.calculate_var(portfolio_returns),
                'cvar': self.var_limit.calculate_cvar(portfolio_returns)
            }
        }

        return analysis

    def rebalance_portfolio(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_prices: np.ndarray
    ) -> Dict:
        """
        Calculate rebalancing trades.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_prices: Current asset prices

        Returns:
            Rebalancing specification
        """
        weight_drift = target_weights - current_weights

        rebalancing = {
            'trades': [],
            'total_rebalance_value': 0,
            'num_trades': 0
        }

        for i, drift in enumerate(weight_drift):
            if abs(drift) > 0.001:  # Minimum trade threshold
                trade_value = drift * 100000  # Assuming $100K portfolio
                trade_units = trade_value / current_prices[i]

                rebalancing['trades'].append({
                    'asset': self.asset_names[i],
                    'action': 'buy' if drift > 0 else 'sell',
                    'weight_drift': drift,
                    'value': trade_value,
                    'units': trade_units
                })

                rebalancing['total_rebalance_value'] += abs(trade_value)
                rebalancing['num_trades'] += 1

        return rebalancing

    def stress_test(
        self,
        weights: np.ndarray,
        scenarios: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Stress test portfolio under different scenarios.

        Args:
            weights: Portfolio weights
            scenarios: Dict of {scenario_name: returns_array}

        Returns:
            Stress test results
        """
        results = {}

        for scenario_name, scenario_returns in scenarios.items():
            portfolio_returns = scenario_returns @ weights

            results[scenario_name] = {
                'return': np.mean(portfolio_returns),
                'volatility': np.std(portfolio_returns) * np.sqrt(252),
                'worst_case': np.min(portfolio_returns),
                'var': np.percentile(portfolio_returns, 5),
                'cvar': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)])
            }

        return results

    def generate_report(
        self,
        weights: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate comprehensive portfolio report.

        Args:
            weights: Portfolio weights (uses optimal if not provided)

        Returns:
            Formatted report string
        """
        if weights is None:
            allocation = self.get_optimal_allocation()
            weights = allocation.weights
        else:
            allocation = None

        report = "=" * 70 + "\n"
        report += "PORTFOLIO RISK MANAGEMENT REPORT\n"
        report += "=" * 70 + "\n\n"

        # Allocation
        report += "OPTIMAL ALLOCATION\n"
        report += "-" * 70 + "\n"
        for name, weight in zip(self.asset_names, weights):
            report += f"{name:20} {weight:>10.2%}\n"

        if allocation:
            report += f"\nExpected Return:     {allocation.expected_return:>10.2%}\n"
            report += f"Volatility:          {allocation.volatility:>10.2%}\n"
            report += f"Sharpe Ratio:        {allocation.sharpe_ratio:>10.3f}\n"

        # Risk Analysis
        risk_analysis = self.analyze_risk(weights)
        report += "\n" + "=" * 70 + "\n"
        report += "RISK ANALYSIS\n"
        report += "-" * 70 + "\n"

        report += f"\nConcentration:\n"
        report += f"  Herfindahl Index: {risk_analysis['concentration']['herfindahl']:.4f}\n"
        report += f"  Top Position: {risk_analysis['concentration']['max_position']:.2%}\n"

        report += f"\nDrawdown:\n"
        report += f"  Max Drawdown: {risk_analysis['drawdown']['max_drawdown']:.2%}\n"
        report += f"  Remaining Budget: {risk_analysis['drawdown']['remaining_budget']:.2%}\n"

        report += f"\nVolatility:\n"
        report += f"  Portfolio Vol: {risk_analysis['volatility']['portfolio_vol']:.2%}\n"

        report += f"\nTail Risk:\n"
        report += f"  VaR (95%): {risk_analysis['tail_risk']['var']:.2%}\n"
        report += f"  CVaR (95%): {risk_analysis['tail_risk']['cvar']:.2%}\n"

        report += "\n" + "=" * 70 + "\n"

        return report
