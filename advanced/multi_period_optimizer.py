"""
Multi-Period Optimizer Module

Implements multi-period portfolio optimization:
- 2-period optimization
- Dynamic rebalancing
- Cash flow integration
- Path-dependent optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiPeriodAllocation:
    """Multi-period allocation result."""
    period_1_weights: np.ndarray
    period_2_weights: np.ndarray
    period_1_return: float
    period_2_expected_return: float
    total_expected_return: float
    period_1_variance: float
    period_2_variance: float
    rebalancing_trades: List[Dict]
    cash_flows: Dict[int, float]  # {period: amount}


class MultiPeriodOptimizer:
    """
    Multi-period portfolio optimization.
    """

    def __init__(self, num_periods: int = 2):
        """
        Initialize multi-period optimizer.

        Args:
            num_periods: Number of periods (default: 2)
        """
        self.num_periods = num_periods
        logger.info(f"Multi-period optimizer initialized for {num_periods} periods")

    def optimize_two_period(
        self,
        returns_mean_p1: np.ndarray,
        returns_cov_p1: np.ndarray,
        returns_mean_p2: np.ndarray,
        returns_cov_p2: np.ndarray,
        correlation_p1_p2: np.ndarray,
        risk_aversion: float = 2.0,
        cash_flow_p1: float = 0,
        cash_flow_p2: float = 0,
        min_weights: float = 0,
        max_weights: float = 1
    ) -> MultiPeriodAllocation:
        """
        Optimize allocation across two periods.

        Args:
            returns_mean_p1: Mean returns period 1
            returns_cov_p1: Covariance period 1
            returns_mean_p2: Expected mean returns period 2
            returns_cov_p2: Expected covariance period 2
            correlation_p1_p2: Correlation between periods
            risk_aversion: Risk aversion coefficient
            cash_flow_p1: Cash inflow period 1
            cash_flow_p2: Cash inflow period 2
            min_weights: Minimum weight per asset
            max_weights: Maximum weight per asset

        Returns:
            MultiPeriodAllocation
        """
        num_assets = len(returns_mean_p1)

        # Expected wealth after period 1
        # W_1 = W_0 * (1 + r_p1) + CF_p1

        # Objective: Maximize E[U(W_2)] = E[W_2] - (risk_aversion/2) * Var(W_2)

        # Simplified approach: Find optimal weights for each period
        # Period 1: Optimize based on period 1 metrics
        # Period 2: Optimize based on expected period 2 metrics

        # Period 1 optimization
        w1 = self._optimize_period(
            returns_mean_p1,
            returns_cov_p1,
            risk_aversion,
            min_weights,
            max_weights
        )

        # Expected portfolio return and variance period 1
        r1_expected = w1 @ returns_mean_p1
        v1 = w1 @ returns_cov_p1 @ w1

        # Period 2: Conditional on period 1 outcome
        # Assume period 2 means are adjusted based on period 1 returns
        w2 = self._optimize_period(
            returns_mean_p2,
            returns_cov_p2,
            risk_aversion,
            min_weights,
            max_weights
        )

        r2_expected = w2 @ returns_mean_p2
        v2 = w2 @ returns_cov_p2 @ w2

        # Calculate rebalancing trades
        rebalancing_trades = [
            {
                'asset': i,
                'period_1_weight': w1[i],
                'period_2_weight': w2[i],
                'trade': w2[i] - w1[i]
            }
            for i in range(num_assets)
        ]

        total_return = r1_expected + r2_expected
        total_variance = v1 + v2 + 2 * np.cov(w1 @ returns_cov_p1, w2 @ returns_cov_p2)

        return MultiPeriodAllocation(
            period_1_weights=w1,
            period_2_weights=w2,
            period_1_return=r1_expected,
            period_2_expected_return=r2_expected,
            total_expected_return=total_return,
            period_1_variance=v1,
            period_2_variance=v2,
            rebalancing_trades=rebalancing_trades,
            cash_flows={1: cash_flow_p1, 2: cash_flow_p2}
        )

    def _optimize_period(
        self,
        returns_mean: np.ndarray,
        returns_cov: np.ndarray,
        risk_aversion: float,
        min_weight: float,
        max_weight: float
    ) -> np.ndarray:
        """
        Optimize allocation for a single period using mean-variance.

        Args:
            returns_mean: Mean returns
            returns_cov: Covariance matrix
            risk_aversion: Risk aversion coefficient
            min_weight: Minimum weight
            max_weight: Maximum weight

        Returns:
            Optimal weights
        """
        num_assets = len(returns_mean)

        # Inverse covariance
        try:
            cov_inv = np.linalg.inv(returns_cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            cov_inv = np.linalg.pinv(returns_cov)

        # Raw optimal weights (no constraints)
        raw_weights = (1 / risk_aversion) * (cov_inv @ returns_mean)

        # Normalize
        raw_weights = raw_weights / np.sum(raw_weights)

        # Apply constraints
        weights = np.clip(raw_weights, min_weight, max_weight)
        weights = weights / np.sum(weights)  # Renormalize

        return weights

    def dynamic_rebalancing(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        returns_realized_p1: np.ndarray,
        expected_returns_p2: np.ndarray,
        volatilities_p2: np.ndarray,
        rebalance_threshold: float = 0.05
    ) -> Dict:
        """
        Calculate dynamic rebalancing after period 1.

        Args:
            current_weights: Current weights after period 1
            target_weights: Target weights for period 2
            returns_realized_p1: Realized returns in period 1
            expected_returns_p2: Expected returns period 2
            volatilities_p2: Expected volatilities period 2
            rebalance_threshold: Rebalance if drift exceeds threshold

        Returns:
            Dictionary with rebalancing decision
        """
        # Calculate drift from target
        drift = current_weights - target_weights
        max_drift = np.max(np.abs(drift))

        should_rebalance = max_drift > rebalance_threshold

        # Calculate opportunity cost of NOT rebalancing
        # Expected deviation from optimal path
        deviation_cost = np.sum(np.abs(drift) * volatilities_p2)

        return {
            'should_rebalance': should_rebalance,
            'max_drift': max_drift,
            'deviation_cost': deviation_cost,
            'trades_required': list(np.where(np.abs(drift) > rebalance_threshold)[0]),
            'current_weights': current_weights,
            'target_weights': target_weights,
            'drift_vector': drift
        }

    def path_dependent_optimization(
        self,
        returns_scenario_1: np.ndarray,
        returns_scenario_2: np.ndarray,
        probability_scenario: float = 0.5,
        utility_function: Optional[callable] = None
    ) -> Dict:
        """
        Optimize over multiple return scenarios (path-dependent).

        Args:
            returns_scenario_1: Returns under scenario 1
            returns_scenario_2: Returns under scenario 2
            probability_scenario: Probability of scenario 1
            utility_function: Custom utility function

        Returns:
            Path-dependent optimization result
        """
        if utility_function is None:
            # Default: Expected return - risk penalty
            utility_function = lambda r, v: np.mean(r) - 0.5 * np.var(r)

        # Scenario 1
        util_1 = utility_function(returns_scenario_1, np.var(returns_scenario_1))
        return_1 = np.mean(returns_scenario_1)
        var_1 = np.var(returns_scenario_1)

        # Scenario 2
        util_2 = utility_function(returns_scenario_2, np.var(returns_scenario_2))
        return_2 = np.mean(returns_scenario_2)
        var_2 = np.var(returns_scenario_2)

        # Expected utility
        expected_utility = probability_scenario * util_1 + (1 - probability_scenario) * util_2
        expected_return = probability_scenario * return_1 + (1 - probability_scenario) * return_2
        expected_variance = probability_scenario * var_1 + (1 - probability_scenario) * var_2

        return {
            'scenario_1': {
                'utility': util_1,
                'return': return_1,
                'variance': var_1,
                'probability': probability_scenario
            },
            'scenario_2': {
                'utility': util_2,
                'return': return_2,
                'variance': var_2,
                'probability': 1 - probability_scenario
            },
            'expected_utility': expected_utility,
            'expected_return': expected_return,
            'expected_variance': expected_variance,
            'better_scenario': 1 if util_1 > util_2 else 2
        }

    def calculate_conditional_var(
        self,
        portfolio_returns_p1: np.ndarray,
        portfolio_returns_p2_given_p1: Dict[int, np.ndarray],
        weights_p1: np.ndarray
    ) -> Dict:
        """
        Calculate Value at Risk conditional on period 1 outcomes.

        Args:
            portfolio_returns_p1: Period 1 returns
            portfolio_returns_p2_given_p1: Period 2 returns given each P1 outcome
            weights_p1: Period 1 weights

        Returns:
            Conditional VaR analysis
        """
        # P1 VaR
        var_p1 = np.percentile(portfolio_returns_p1, 5)

        # P2 VaR given bad P1 outcome
        bad_p1_index = np.argmin(portfolio_returns_p1)
        returns_p2_given_bad_p1 = portfolio_returns_p2_given_p1.get(bad_p1_index, np.array([0]))

        var_p2_given_bad_p1 = np.percentile(returns_p2_given_bad_p1, 5)

        # Two-period tail risk
        two_period_tail = var_p1 + var_p2_given_bad_p1

        return {
            'var_p1': var_p1,
            'var_p2_given_bad_p1': var_p2_given_bad_p1,
            'two_period_var': two_period_tail,
            'worst_case_return': var_p1 + var_p2_given_bad_p1
        }

    def get_rebalancing_schedule(
        self,
        target_weights_by_period: Dict[int, np.ndarray],
        drift_tolerance: float = 0.05
    ) -> Dict:
        """
        Generate rebalancing schedule over multiple periods.

        Args:
            target_weights_by_period: {period: weights}
            drift_tolerance: Tolerance for drift from target

        Returns:
            Rebalancing schedule
        """
        schedule = {}
        prev_weights = target_weights_by_period.get(1, None)

        for period in sorted(target_weights_by_period.keys()):
            target_weights = target_weights_by_period[period]

            if prev_weights is not None:
                drift = np.abs(target_weights - prev_weights)
                should_rebalance = np.any(drift > drift_tolerance)

                schedule[period] = {
                    'target_weights': target_weights,
                    'should_rebalance': should_rebalance,
                    'max_drift': np.max(drift),
                    'trades': list(np.where(drift > drift_tolerance)[0])
                }
            else:
                schedule[period] = {
                    'target_weights': target_weights,
                    'should_rebalance': True,
                    'max_drift': 1.0,
                    'trades': list(range(len(target_weights)))
                }

            prev_weights = target_weights

        return schedule
