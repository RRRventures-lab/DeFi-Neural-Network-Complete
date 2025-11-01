"""
Hedging Module

Implements hedging strategies for risk management:
- Correlation-based hedging
- VaR-based hedging
- Options-based hedging
- Dynamic hedging
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HedgingStrategy:
    """
    Base class for hedging strategies.
    """

    def calculate_hedge(self, **kwargs) -> Dict:
        """
        Calculate hedging requirements.

        Returns:
            Dictionary with hedge specifications
        """
        raise NotImplementedError


class CorrelationHedge(HedgingStrategy):
    """
    Hedge portfolio using negatively correlated assets.
    """

    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialize correlation hedge.

        Args:
            correlation_matrix: Asset correlation matrix
        """
        self.correlation_matrix = correlation_matrix
        self.num_assets = correlation_matrix.shape[0]

    def find_best_hedge(
        self,
        primary_asset: int,
        min_correlation: float = -0.5
    ) -> Optional[int]:
        """
        Find best hedge asset for primary position.

        Args:
            primary_asset: Index of primary asset to hedge
            min_correlation: Minimum correlation threshold

        Returns:
            Index of best hedge asset or None
        """
        correlations = self.correlation_matrix[primary_asset, :]

        # Find most negative correlation
        hedge_candidates = np.where(correlations < min_correlation)[0]

        if len(hedge_candidates) == 0:
            return None

        best_hedge_idx = hedge_candidates[np.argmin(correlations[hedge_candidates])]

        return best_hedge_idx

    def calculate_hedge_ratio(
        self,
        primary_volatility: float,
        hedge_volatility: float,
        correlation: float
    ) -> float:
        """
        Calculate hedge ratio (proportion of primary position to hedge).

        Hedge Ratio = (σ_primary / σ_hedge) * correlation

        Args:
            primary_volatility: Volatility of primary asset
            hedge_volatility: Volatility of hedge asset
            correlation: Correlation between assets

        Returns:
            Hedge ratio
        """
        if hedge_volatility <= 0:
            return 0

        ratio = (primary_volatility / hedge_volatility) * abs(correlation)
        return min(ratio, 1.0)  # Cap at 100% hedge

    def calculate_hedge(
        self,
        primary_asset: int,
        primary_position: float,
        primary_volatility: float,
        hedge_volatilities: Optional[Dict[int, float]] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate correlation-based hedge.

        Args:
            primary_asset: Index of primary asset
            primary_position: Size of primary position
            primary_volatility: Volatility of primary asset
            hedge_volatilities: Dict of {asset_idx: volatility}

        Returns:
            Hedge specification
        """
        best_hedge_idx = self.find_best_hedge(primary_asset)

        if best_hedge_idx is None:
            return {
                'primary_asset': primary_asset,
                'hedge_asset': None,
                'hedge_position': 0,
                'effectiveness': 0
            }

        correlation = self.correlation_matrix[primary_asset, best_hedge_idx]

        if hedge_volatilities is None:
            hedge_vol = 0.15
        else:
            hedge_vol = hedge_volatilities.get(best_hedge_idx, 0.15)

        hedge_ratio = self.calculate_hedge_ratio(
            primary_volatility,
            hedge_vol,
            correlation
        )

        hedge_position = primary_position * hedge_ratio

        return {
            'primary_asset': primary_asset,
            'primary_position': primary_position,
            'hedge_asset': best_hedge_idx,
            'hedge_position': hedge_position,
            'hedge_ratio': hedge_ratio,
            'correlation': correlation,
            'effectiveness': abs(correlation)  # Higher = more effective
        }


class VaRBasedHedge(HedgingStrategy):
    """
    Adjust positions based on Value at Risk calculations.
    """

    def __init__(self, target_var: float = -0.05):
        """
        Initialize VaR-based hedge.

        Args:
            target_var: Target Value at Risk level
        """
        self.target_var = target_var

    def calculate_var_reduction(
        self,
        current_var: float,
        hedge_var: float,
        hedge_correlation: float
    ) -> float:
        """
        Calculate VaR reduction from hedge.

        Args:
            current_var: Current portfolio VaR
            hedge_var: VaR of hedge instrument
            hedge_correlation: Correlation with portfolio

        Returns:
            Reduction in VaR
        """
        # Simplified VaR reduction calculation
        reduction = abs(current_var) * abs(hedge_correlation) * (abs(hedge_var) / 0.05)
        return min(reduction, abs(current_var))

    def calculate_hedge(
        self,
        portfolio_var: float,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Calculate VaR-based hedge position.

        Args:
            portfolio_var: Current portfolio VaR
            portfolio_returns: Portfolio returns array
            hedge_returns: Hedge instrument returns array

        Returns:
            Hedge specification
        """
        # Calculate correlation
        correlation = np.corrcoef(portfolio_returns, hedge_returns)[0, 1]

        # Calculate hedge VaR
        hedge_var = np.percentile(hedge_returns, 5)

        # Determine if hedging is needed
        needs_hedge = portfolio_var < self.target_var

        if needs_hedge and correlation < -0.3:
            # Calculate hedge size to reach target VaR
            var_gap = abs(portfolio_var) - abs(self.target_var)
            var_reduction = self.calculate_var_reduction(
                portfolio_var,
                hedge_var,
                correlation
            )

            hedge_size = var_gap / var_reduction if var_reduction > 0 else 0
        else:
            hedge_size = 0

        return {
            'current_var': portfolio_var,
            'target_var': self.target_var,
            'hedge_needed': needs_hedge,
            'hedge_size': max(0, min(hedge_size, 1.0)),  # Cap at 100%
            'var_reduction': self.calculate_var_reduction(
                portfolio_var,
                hedge_var,
                correlation
            ),
            'effectiveness': abs(correlation)
        }


class OptionsHedge(HedgingStrategy):
    """
    Calculate hedge requirements using options.
    """

    def __init__(self):
        """Initialize options hedge strategy."""
        pass

    def calculate_put_hedge(
        self,
        position_size: float,
        position_price: float,
        strike_price: float,
        put_premium: float,
        time_to_expiry: float = 30 / 365
    ) -> Dict:
        """
        Calculate protective put hedge.

        Args:
            position_size: Number of units held
            position_price: Current price of underlying
            strike_price: Strike price of put option
            put_premium: Cost of put option
            time_to_expiry: Time to expiration in years

        Returns:
            Hedge specification
        """
        # Calculate protection level
        protection = max(0, strike_price - position_price)
        cost = put_premium * position_size

        # Calculate effective price floor
        effective_floor = strike_price - put_premium

        # Calculate max loss
        max_loss = (position_price - effective_floor) * position_size

        return {
            'hedge_type': 'protective_put',
            'strike_price': strike_price,
            'put_premium': put_premium,
            'position_size': position_size,
            'protection': protection,
            'hedge_cost': cost,
            'cost_pct': (put_premium / position_price) * 100,
            'effective_floor': effective_floor,
            'max_loss': max_loss
        }

    def calculate_collar_hedge(
        self,
        position_size: float,
        position_price: float,
        put_strike: float,
        call_strike: float,
        put_premium: float,
        call_premium_received: float
    ) -> Dict:
        """
        Calculate collar hedge (put + short call).

        Args:
            position_size: Number of units held
            position_price: Current price
            put_strike: Strike of protective put
            call_strike: Strike of short call
            put_premium: Cost of put
            call_premium_received: Premium from short call

        Returns:
            Hedge specification
        """
        net_cost = (put_premium - call_premium_received) * position_size
        max_loss = (position_price - put_strike) * position_size
        max_gain = (call_strike - position_price) * position_size

        return {
            'hedge_type': 'collar',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'put_premium': put_premium,
            'call_premium_received': call_premium_received,
            'position_size': position_size,
            'net_cost': net_cost,
            'cost_pct': (net_cost / (position_price * position_size)) * 100,
            'max_loss': max_loss,
            'max_gain': max_gain
        }


class DynamicHedge(HedgingStrategy):
    """
    Dynamically adjust hedge based on market conditions.
    """

    def __init__(self, rebalance_threshold: float = 0.10):
        """
        Initialize dynamic hedge.

        Args:
            rebalance_threshold: Rebalance when delta exceeds threshold
        """
        self.rebalance_threshold = rebalance_threshold

    def should_rebalance(
        self,
        target_hedge_ratio: float,
        current_hedge_ratio: float
    ) -> bool:
        """
        Determine if hedge needs rebalancing.

        Args:
            target_hedge_ratio: Target hedge ratio
            current_hedge_ratio: Current hedge ratio

        Returns:
            True if rebalancing needed
        """
        drift = abs(target_hedge_ratio - current_hedge_ratio)
        return drift > self.rebalance_threshold

    def calculate_rebalance(
        self,
        target_hedge_ratio: float,
        current_hedge_ratio: float,
        current_position: float,
        position_price: float
    ) -> Dict:
        """
        Calculate rebalancing requirements.

        Args:
            target_hedge_ratio: Target hedge ratio
            current_hedge_ratio: Current hedge ratio
            current_position: Current position size
            position_price: Current price

        Returns:
            Rebalance specification
        """
        target_position = current_position * target_hedge_ratio
        current_position_value = current_position * position_price
        target_position_value = target_position * position_price

        rebalance_value = target_position_value - current_position_value
        rebalance_units = rebalance_value / position_price

        return {
            'should_rebalance': self.should_rebalance(
                target_hedge_ratio,
                current_hedge_ratio
            ),
            'current_ratio': current_hedge_ratio,
            'target_ratio': target_hedge_ratio,
            'drift': abs(target_hedge_ratio - current_hedge_ratio),
            'rebalance_units': rebalance_units,
            'rebalance_value': rebalance_value,
            'direction': 'buy' if rebalance_units > 0 else 'sell'
        }

    def calculate_hedge(
        self,
        portfolio_volatility: float,
        asset_volatility: float,
        target_portfolio_volatility: float = 0.10,
        **kwargs
    ) -> Dict:
        """
        Calculate dynamic hedge requirements.

        Args:
            portfolio_volatility: Current portfolio volatility
            asset_volatility: Asset volatility
            target_portfolio_volatility: Target portfolio volatility

        Returns:
            Hedge specification
        """
        # Calculate required hedge ratio
        vol_excess = portfolio_volatility - target_portfolio_volatility
        vol_excess = max(0, vol_excess)

        if asset_volatility > 0:
            target_hedge_ratio = vol_excess / asset_volatility
        else:
            target_hedge_ratio = 0

        target_hedge_ratio = min(target_hedge_ratio, 1.0)

        return {
            'current_volatility': portfolio_volatility,
            'target_volatility': target_portfolio_volatility,
            'vol_excess': vol_excess,
            'target_hedge_ratio': target_hedge_ratio,
            'rebalance_frequency': 'daily' if vol_excess > 0.05 else 'weekly'
        }


class HedgingManager:
    """
    Manage multiple hedging strategies.
    """

    def __init__(self):
        """Initialize hedging manager."""
        self.strategies: Dict[str, HedgingStrategy] = {}
        self.active_hedges: List[Dict] = []

    def register_strategy(self, name: str, strategy: HedgingStrategy) -> None:
        """
        Register a hedging strategy.

        Args:
            name: Strategy name
            strategy: HedgingStrategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Registered hedging strategy: {name}")

    def get_hedge_recommendation(
        self,
        primary_asset: int,
        market_data: Dict,
        **kwargs
    ) -> Dict:
        """
        Get hedging recommendation from all strategies.

        Args:
            primary_asset: Asset to hedge
            market_data: Market data
            **kwargs: Additional parameters

        Returns:
            Consolidated hedge recommendation
        """
        recommendations = {}

        for name, strategy in self.strategies.items():
            try:
                rec = strategy.calculate_hedge(
                    primary_asset=primary_asset,
                    **market_data,
                    **kwargs
                )
                recommendations[name] = rec
            except Exception as e:
                logger.error(f"Error in {name} strategy: {e}")
                continue

        return recommendations

    def rank_hedges(
        self,
        recommendations: Dict
    ) -> List[Tuple[str, float]]:
        """
        Rank hedging strategies by effectiveness.

        Args:
            recommendations: Hedge recommendations

        Returns:
            List of (strategy_name, effectiveness_score) tuples
        """
        scores = []

        for name, rec in recommendations.items():
            effectiveness = rec.get('effectiveness', 0)
            scores.append((name, effectiveness))

        # Sort by effectiveness
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores
