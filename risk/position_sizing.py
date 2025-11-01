"""
Position Sizing Module

Implements various position sizing strategies:
- Kelly Criterion
- Fixed Fraction
- Volatility Targeting
- Risk Parity
- Drawdown-based sizing
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Base class for position sizing strategies.
    """

    def calculate_size(
        self,
        account_equity: float,
        position_price: float,
        **kwargs
    ) -> float:
        """
        Calculate position size.

        Args:
            account_equity: Total account equity
            position_price: Entry price for position
            **kwargs: Strategy-specific parameters

        Returns:
            Position size (number of units)
        """
        raise NotImplementedError


class KellyPositionSizer(PositionSizer):
    """
    Kelly Criterion based position sizing.

    Position size based on win rate and win/loss ratio.
    """

    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize Kelly position sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion percentage.

        Kelly % = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size (positive value)

        Returns:
            Kelly fraction (fraction of account to risk)
        """
        if avg_win <= 0 or avg_loss <= 0:
            return 0

        kelly = (
            (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        )

        # Apply kelly fraction (e.g., quarter Kelly for safety)
        return max(0, min(kelly * self.kelly_fraction, 0.25))

    def calculate_size(
        self,
        account_equity: float,
        position_price: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        max_loss: float = 0.02,
        **kwargs
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Args:
            account_equity: Total account equity
            position_price: Entry price
            win_rate: Estimated win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            max_loss: Maximum loss per trade (2% default)

        Returns:
            Position size (number of units)
        """
        kelly_pct = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        # Risk-based position sizing
        risk_amount = account_equity * kelly_pct
        position_size = risk_amount / (position_price * max_loss)

        return max(0, position_size)


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction position sizing.

    Risk a fixed percentage of account per trade.
    """

    def __init__(self, risk_per_trade: float = 0.02):
        """
        Initialize fixed fraction sizer.

        Args:
            risk_per_trade: Risk per trade as fraction of account (0.02 = 2%)
        """
        self.risk_per_trade = risk_per_trade

    def calculate_size(
        self,
        account_equity: float,
        position_price: float,
        stop_loss_pct: float = 0.05,
        **kwargs
    ) -> float:
        """
        Calculate position size using fixed fraction.

        Args:
            account_equity: Total account equity
            position_price: Entry price
            stop_loss_pct: Stop loss distance as percentage

        Returns:
            Position size (number of units)
        """
        risk_amount = account_equity * self.risk_per_trade
        position_size = risk_amount / (position_price * stop_loss_pct)

        return max(0, position_size)

    def get_stop_loss_price(
        self,
        entry_price: float,
        stop_loss_pct: float = 0.05
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss distance

        Returns:
            Stop loss price
        """
        return entry_price * (1 - stop_loss_pct)


class VolatilityTargetSizer(PositionSizer):
    """
    Volatility-targeted position sizing.

    Scale position inversely with volatility.
    """

    def __init__(self, target_volatility: float = 0.10):
        """
        Initialize volatility target sizer.

        Args:
            target_volatility: Target portfolio volatility (0.10 = 10%)
        """
        self.target_volatility = target_volatility

    def calculate_size(
        self,
        account_equity: float,
        position_price: float,
        volatility: float = 0.15,
        **kwargs
    ) -> float:
        """
        Calculate position size using volatility targeting.

        Args:
            account_equity: Total account equity
            position_price: Entry price
            volatility: Asset volatility

        Returns:
            Position size (number of units)
        """
        if volatility <= 0:
            return 0

        # Volatility scalar (inverse relationship)
        volatility_scalar = self.target_volatility / volatility

        # Base position
        base_position = account_equity / (position_price * self.target_volatility)

        # Scaled position
        position_size = base_position * volatility_scalar

        return max(0, position_size)

    def get_allocation(
        self,
        volatilities: np.ndarray
    ) -> np.ndarray:
        """
        Get allocation weights for multiple assets based on volatility.

        Args:
            volatilities: Asset volatilities

        Returns:
            Allocation weights (sum to 1)
        """
        # Inverse volatility weighting
        inverse_vols = 1.0 / volatilities
        weights = inverse_vols / np.sum(inverse_vols)

        return weights


class RiskParitySizer(PositionSizer):
    """
    Risk parity position sizing.

    Each position contributes equally to portfolio risk.
    """

    def __init__(self):
        """Initialize risk parity sizer."""
        pass

    def calculate_sizes(
        self,
        account_equity: float,
        positions: Dict[str, Dict],
        target_volatility: float = 0.15
    ) -> Dict[str, float]:
        """
        Calculate risk parity position sizes.

        Args:
            account_equity: Total account equity
            positions: Dict of {asset: {price, volatility, correlation}}
            target_volatility: Target portfolio volatility

        Returns:
            Dict of {asset: position_size}
        """
        assets = list(positions.keys())
        num_assets = len(assets)

        # Start with equal risk allocation
        risk_per_asset = target_volatility / num_assets

        sizes = {}

        for asset in assets:
            volatility = positions[asset].get('volatility', 0.15)
            price = positions[asset].get('price', 1.0)

            if volatility > 0:
                position_value = account_equity * (risk_per_asset / volatility)
                position_size = position_value / price
                sizes[asset] = max(0, position_size)
            else:
                sizes[asset] = 0

        return sizes

    def calculate_risk_contribution(
        self,
        sizes: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each position.

        Args:
            sizes: Position sizes
            volatilities: Asset volatilities
            correlation_matrix: Asset correlation matrix

        Returns:
            Dict of {asset: risk_contribution}
        """
        assets = list(sizes.keys())
        num_assets = len(assets)

        contributions = {}

        for i, asset in enumerate(assets):
            # Simple calculation (assumes independent assets)
            if correlation_matrix is None:
                vol = volatilities.get(asset, 0.15)
                contribution = sizes[asset] * vol
            else:
                # Account for correlations
                vol = volatilities.get(asset, 0.15)
                contribution = sizes[asset] * vol

            contributions[asset] = contribution

        return contributions


class DrawdownAdaptiveSizer(PositionSizer):
    """
    Reduce position size based on portfolio drawdown.

    Scale down positions as drawdown increases.
    """

    def __init__(self, max_position_scale: float = 1.0):
        """
        Initialize drawdown adaptive sizer.

        Args:
            max_position_scale: Maximum position scale factor
        """
        self.max_position_scale = max_position_scale

    def get_position_scale(
        self,
        current_drawdown: float,
        max_drawdown_limit: float = -0.20
    ) -> float:
        """
        Calculate position scale factor based on drawdown.

        Args:
            current_drawdown: Current portfolio drawdown (negative value)
            max_drawdown_limit: Maximum allowed drawdown

        Returns:
            Position scale factor (0-1)
        """
        if current_drawdown >= 0:
            return self.max_position_scale

        # Linear scaling: 100% scale at 0% drawdown, 0% at max drawdown
        scale = 1.0 + (current_drawdown / max_drawdown_limit)
        scale = max(0, min(scale, self.max_position_scale))

        return scale

    def calculate_size(
        self,
        account_equity: float,
        position_price: float,
        current_drawdown: float = 0,
        max_drawdown_limit: float = -0.20,
        base_size: float = 100,
        **kwargs
    ) -> float:
        """
        Calculate position size with drawdown scaling.

        Args:
            account_equity: Total account equity
            position_price: Entry price
            current_drawdown: Current portfolio drawdown
            max_drawdown_limit: Maximum allowed drawdown
            base_size: Base position size

        Returns:
            Scaled position size
        """
        scale = self.get_position_scale(current_drawdown, max_drawdown_limit)
        scaled_size = base_size * scale

        return max(0, scaled_size)


class AdaptivePositionSizer:
    """
    Adaptive position sizing combining multiple strategies.
    """

    def __init__(self):
        """Initialize adaptive sizer with multiple strategies."""
        self.kelly_sizer = KellyPositionSizer(kelly_fraction=0.25)
        self.fixed_sizer = FixedFractionSizer(risk_per_trade=0.02)
        self.vol_sizer = VolatilityTargetSizer(target_volatility=0.10)
        self.rp_sizer = RiskParitySizer()
        self.dd_sizer = DrawdownAdaptiveSizer()

    def calculate_optimal_size(
        self,
        account_equity: float,
        position_price: float,
        market_data: Dict,
        performance_data: Dict,
        strategy: str = 'adaptive'
    ) -> float:
        """
        Calculate optimal position size using adaptive strategy.

        Args:
            account_equity: Total account equity
            position_price: Entry price
            market_data: Market data (volatility, etc.)
            performance_data: Performance data (win rate, returns, etc.)
            strategy: Strategy to use ('adaptive', 'kelly', 'fixed', 'volatility')

        Returns:
            Position size
        """
        if strategy == 'kelly':
            return self.kelly_sizer.calculate_size(
                account_equity,
                position_price,
                **performance_data
            )
        elif strategy == 'fixed':
            return self.fixed_sizer.calculate_size(
                account_equity,
                position_price,
                **market_data
            )
        elif strategy == 'volatility':
            return self.vol_sizer.calculate_size(
                account_equity,
                position_price,
                **market_data
            )
        elif strategy == 'adaptive':
            # Combine strategies
            kelly_size = self.kelly_sizer.calculate_size(
                account_equity,
                position_price,
                **performance_data
            )
            fixed_size = self.fixed_sizer.calculate_size(
                account_equity,
                position_price,
                **market_data
            )
            vol_size = self.vol_sizer.calculate_size(
                account_equity,
                position_price,
                **market_data
            )

            # Average the strategies
            return np.mean([kelly_size, fixed_size, vol_size])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
