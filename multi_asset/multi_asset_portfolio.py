"""
Multi-Asset Portfolio Module

Implements cross-asset portfolio management:
- Asset class allocation
- Dynamic rebalancing
- Portfolio metrics
- Asset class monitoring
- Performance attribution
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssetClassAllocation:
    """Asset class allocation detail."""
    asset_class: str  # 'crypto', 'forex', 'derivatives', 'stocks'
    target_weight: float
    current_weight: float
    value: float
    drift: float  # Current - Target


@dataclass
class PortfolioHolding:
    """Individual holding in portfolio."""
    symbol: str
    asset_class: str
    quantity: float
    entry_price: float
    current_price: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics."""
    total_value: float
    total_invested: float
    total_gains: float
    portfolio_return_pct: float
    daily_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    num_positions: int
    asset_class_count: int
    diversification_score: float  # 0-1


@dataclass
class RebalancingTrade:
    """Rebalancing trade instruction."""
    symbol: str
    asset_class: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_weight: float
    current_weight: float
    reason: str


class MultiAssetPortfolio:
    """
    Manages portfolio across multiple asset classes.
    """

    def __init__(self, initial_value: float = 100000):
        """
        Initialize multi-asset portfolio.

        Args:
            initial_value: Initial portfolio value
        """
        self.initial_value = initial_value
        self.current_value = initial_value
        self.holdings: Dict[str, PortfolioHolding] = {}
        self.asset_class_allocations: Dict[str, AssetClassAllocation] = {}
        self.target_allocation: Dict[str, float] = {}  # {asset_class: weight}
        self.price_history: Dict[str, List[float]] = {}
        self.daily_returns: List[float] = []
        self.rebalancing_history: List[List[RebalancingTrade]] = []

        logger.info(f"MultiAssetPortfolio initialized: ${initial_value}")

    def set_target_allocation(self, allocation: Dict[str, float]) -> None:
        """
        Set target allocation by asset class.

        Args:
            allocation: {asset_class: target_weight}
        """
        if abs(sum(allocation.values()) - 1.0) > 0.01:
            logger.warning("Allocation weights do not sum to 1.0")

        self.target_allocation = allocation
        logger.info(f"Target allocation set: {allocation}")

    def add_holding(
        self,
        symbol: str,
        asset_class: str,
        quantity: float,
        entry_price: float,
        current_price: float,
    ) -> None:
        """
        Add holding to portfolio.

        Args:
            symbol: Asset symbol
            asset_class: Asset class
            quantity: Number of units/contracts
            entry_price: Entry price
            current_price: Current price
        """
        value = quantity * current_price
        invested = quantity * entry_price
        unrealized_pnl = value - invested
        unrealized_pnl_pct = unrealized_pnl / invested if invested > 0 else 0

        holding = PortfolioHolding(
            symbol=symbol,
            asset_class=asset_class,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            weight=value / self.current_value if self.current_value > 0 else 0,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )

        self.holdings[symbol] = holding
        logger.info(f"Added holding: {symbol} ({asset_class}), value: ${value:.2f}")

    def update_prices(self, prices: Dict[str, float]) -> float:
        """
        Update prices for holdings.

        Args:
            prices: {symbol: new_price}

        Returns:
            New portfolio value
        """
        old_value = self.current_value
        new_value = 0

        for symbol, price in prices.items():
            if symbol in self.holdings:
                holding = self.holdings[symbol]
                holding.current_price = price
                value = holding.quantity * price
                invested = holding.quantity * holding.entry_price
                holding.unrealized_pnl = value - invested
                holding.unrealized_pnl_pct = (
                    holding.unrealized_pnl / invested if invested > 0 else 0
                )
                new_value += value
            else:
                new_value += 0

        # Add cash value (if tracking)
        self.current_value = max(new_value, old_value)  # Don't decrease below initial

        # Calculate daily return
        if old_value > 0:
            daily_return = (self.current_value - old_value) / old_value
            self.daily_returns.append(daily_return)

        # Update weights
        for holding in self.holdings.values():
            holding.weight = (
                (holding.quantity * holding.current_price) / self.current_value
                if self.current_value > 0
                else 0
            )

        logger.debug(f"Updated prices, portfolio value: ${self.current_value:.2f}")

        return self.current_value

    def get_asset_class_allocation(self) -> Dict[str, AssetClassAllocation]:
        """Get allocation by asset class."""
        allocations = {}

        # Calculate current allocation
        class_values = {}
        for holding in self.holdings.values():
            ac = holding.asset_class
            if ac not in class_values:
                class_values[ac] = 0
            class_values[ac] += holding.quantity * holding.current_price

        # Create allocation objects
        for asset_class in set(
            list(class_values.keys()) + list(self.target_allocation.keys())
        ):
            value = class_values.get(asset_class, 0)
            current_weight = value / self.current_value if self.current_value > 0 else 0
            target_weight = self.target_allocation.get(asset_class, 0)
            drift = current_weight - target_weight

            allocations[asset_class] = AssetClassAllocation(
                asset_class=asset_class,
                target_weight=target_weight,
                current_weight=current_weight,
                value=value,
                drift=drift,
            )

        self.asset_class_allocations = allocations
        return allocations

    def calculate_rebalancing_trades(
        self, drift_threshold: float = 0.05
    ) -> List[RebalancingTrade]:
        """
        Calculate rebalancing trades.

        Args:
            drift_threshold: Tolerance for drift before rebalancing

        Returns:
            List of rebalancing trades
        """
        allocations = self.get_asset_class_allocation()
        trades = []

        for asset_class, alloc in allocations.items():
            if abs(alloc.drift) > drift_threshold:
                # Determine action
                if alloc.drift > 0:
                    # Overweight - sell
                    side = "sell"
                    reason = f"Reduce {asset_class} from {alloc.current_weight:.1%} to {alloc.target_weight:.1%}"
                else:
                    # Underweight - buy
                    side = "buy"
                    reason = f"Increase {asset_class} from {alloc.current_weight:.1%} to {alloc.target_weight:.1%}"

                # Find symbols in this asset class
                symbols = [
                    h.symbol for h in self.holdings.values()
                    if h.asset_class == asset_class
                ]

                for symbol in symbols:
                    trade = RebalancingTrade(
                        symbol=symbol,
                        asset_class=asset_class,
                        side=side,
                        quantity=0,  # To be calculated by trader
                        target_weight=alloc.target_weight,
                        current_weight=alloc.current_weight,
                        reason=reason,
                    )
                    trades.append(trade)

        self.rebalancing_history.append(trades)
        logger.info(f"Calculated {len(trades)} rebalancing trades")

        return trades

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics."""
        total_value = self.current_value
        total_invested = sum(
            h.quantity * h.entry_price for h in self.holdings.values()
        )
        total_gains = total_value - total_invested

        # Returns
        portfolio_return_pct = (
            (total_value - self.initial_value) / self.initial_value
            if self.initial_value > 0
            else 0
        )
        daily_return_pct = self.daily_returns[-1] if self.daily_returns else 0

        # Annualized return
        if len(self.daily_returns) > 0:
            annualized_return = (
                np.mean(self.daily_returns) * 252 if len(self.daily_returns) > 0 else 0
            )
        else:
            annualized_return = portfolio_return_pct / (1 / 252)

        # Volatility
        if len(self.daily_returns) > 1:
            volatility = np.std(self.daily_returns) * np.sqrt(252)
        else:
            volatility = 0

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility
            if volatility > 0
            else 0
        )

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Diversification score
        diversification_score = self._calculate_diversification_score()

        return PortfolioMetrics(
            total_value=total_value,
            total_invested=total_invested,
            total_gains=total_gains,
            portfolio_return_pct=portfolio_return_pct,
            daily_return_pct=daily_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            num_positions=len(self.holdings),
            asset_class_count=len(self.asset_class_allocations),
            diversification_score=diversification_score,
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.daily_returns) < 2:
            return 0

        cumulative_returns = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return max_drawdown

    def _calculate_diversification_score(self) -> float:
        """
        Calculate diversification score (0-1).
        Based on concentration and asset class distribution.
        """
        if not self.holdings:
            return 0

        # Position concentration
        weights = np.array([h.weight for h in self.holdings.values()])
        herfindahl = np.sum(weights**2)
        max_positions = len(self.holdings)
        concentration_score = 1 - (herfindahl / (1 / max_positions))

        # Asset class diversification
        allocations = self.get_asset_class_allocation()
        if allocations:
            class_weights = np.array(
                [a.current_weight for a in allocations.values()]
            )
            class_herfindahl = np.sum(class_weights**2)
            class_count = len(allocations)
            class_score = 1 - (class_herfindahl / (1 / class_count)) if class_count > 1 else 0.5
        else:
            class_score = 0

        # Combined score
        diversification_score = (concentration_score + class_score) / 2

        return max(0, min(1, diversification_score))

    def get_performance_attribution(self) -> Dict:
        """Get performance attribution by asset class and holding."""
        attribution = {
            "by_holding": {},
            "by_asset_class": {},
            "top_performers": [],
            "bottom_performers": [],
        }

        # By holding
        for symbol, holding in self.holdings.items():
            attribution["by_holding"][symbol] = {
                "value": holding.quantity * holding.current_price,
                "pnl": holding.unrealized_pnl,
                "pnl_pct": holding.unrealized_pnl_pct,
                "weight": holding.weight,
            }

        # By asset class
        allocations = self.get_asset_class_allocation()
        for asset_class, alloc in allocations.items():
            class_holdings = [
                h for h in self.holdings.values() if h.asset_class == asset_class
            ]
            total_pnl = sum(h.unrealized_pnl for h in class_holdings)

            attribution["by_asset_class"][asset_class] = {
                "value": alloc.value,
                "pnl": total_pnl,
                "weight": alloc.current_weight,
                "holding_count": len(class_holdings),
            }

        # Top and bottom performers
        sorted_holdings = sorted(
            self.holdings.items(),
            key=lambda x: x[1].unrealized_pnl_pct,
            reverse=True,
        )

        attribution["top_performers"] = [
            {"symbol": h[0], "pnl_pct": h[1].unrealized_pnl_pct}
            for h in sorted_holdings[:5]
        ]

        attribution["bottom_performers"] = [
            {"symbol": h[0], "pnl_pct": h[1].unrealized_pnl_pct}
            for h in sorted_holdings[-5:]
        ]

        return attribution

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        metrics = self.get_portfolio_metrics()
        allocations = self.get_asset_class_allocation()

        return {
            "total_value": metrics.total_value,
            "total_gains": metrics.total_gains,
            "portfolio_return_pct": metrics.portfolio_return_pct,
            "annualized_return": metrics.annualized_return,
            "volatility": metrics.volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "num_positions": metrics.num_positions,
            "asset_classes": {
                ac: {
                    "value": alloc.value,
                    "weight": alloc.current_weight,
                    "target_weight": alloc.target_weight,
                    "drift": alloc.drift,
                }
                for ac, alloc in allocations.items()
            },
            "diversification_score": metrics.diversification_score,
        }
