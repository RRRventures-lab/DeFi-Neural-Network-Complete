"""
Tax Optimizer Module

Implements tax-aware portfolio optimization:
- Tax-loss harvesting
- Capital gains management
- Tax-efficient rebalancing
- Wash sale enforcement
- Multiple tax rate handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaxLot:
    """Represents a tax lot (purchase batch) of a security."""
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: str
    current_price: float
    long_term: bool = False  # True if held > 1 year

    @property
    def unrealized_gain(self) -> float:
        """Unrealized gain in dollars."""
        return (self.current_price - self.purchase_price) * self.quantity

    @property
    def unrealized_gain_pct(self) -> float:
        """Unrealized gain as percentage."""
        if self.purchase_price == 0:
            return 0
        return ((self.current_price - self.purchase_price) / self.purchase_price)

    @property
    def gain_type(self) -> str:
        """Return 'gain', 'loss', or 'neutral'."""
        if self.unrealized_gain > 0.01:
            return 'gain'
        elif self.unrealized_gain < -0.01:
            return 'loss'
        else:
            return 'neutral'


@dataclass
class TaxHarvest:
    """Represents a tax-loss harvesting opportunity."""
    symbol: str
    quantity: float
    current_price: float
    realizable_loss: float  # Amount of loss that can be realized
    replacement_symbol: Optional[str] = None  # Replacement for avoiding wash sale
    days_until_wash_clear: int = 0  # Days until wash sale rule expires


@dataclass
class TaxOptimizationResult:
    """Result of tax optimization analysis."""
    current_taxes: float  # Current tax liability
    optimized_taxes: float  # Tax liability after optimization
    tax_savings: float  # Potential tax savings
    harvests: List[TaxHarvest] = field(default_factory=list)
    rebalancing_trades: List[Dict] = field(default_factory=list)
    realized_gains: float = 0
    realized_losses: float = 0
    net_gains: float = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'current_taxes': self.current_taxes,
            'optimized_taxes': self.optimized_taxes,
            'tax_savings': self.tax_savings,
            'num_harvest_opportunities': len(self.harvests),
            'num_trades': len(self.rebalancing_trades),
            'realized_gains': self.realized_gains,
            'realized_losses': self.realized_losses,
            'net_gains': self.net_gains
        }


class TaxOptimizer:
    """
    Tax-aware portfolio optimization system.
    """

    def __init__(
        self,
        short_term_rate: float = 0.37,
        long_term_rate: float = 0.15,
        wash_sale_days: int = 30
    ):
        """
        Initialize tax optimizer.

        Args:
            short_term_rate: Short-term capital gains tax rate
            long_term_rate: Long-term capital gains tax rate
            wash_sale_days: Number of days for wash sale rule
        """
        self.short_term_rate = short_term_rate
        self.long_term_rate = long_term_rate
        self.wash_sale_days = wash_sale_days
        self.tax_lots: Dict[str, List[TaxLot]] = {}
        self.sale_history: List[Dict] = []

        logger.info(f"Tax optimizer initialized")
        logger.info(f"  Short-term rate: {short_term_rate:.1%}")
        logger.info(f"  Long-term rate: {long_term_rate:.1%}")
        logger.info(f"  Wash sale days: {wash_sale_days}")

    def add_tax_lot(self, lot: TaxLot) -> None:
        """
        Add a tax lot to tracking.

        Args:
            lot: TaxLot to add
        """
        if lot.symbol not in self.tax_lots:
            self.tax_lots[lot.symbol] = []

        self.tax_lots[lot.symbol].append(lot)
        logger.debug(f"Added tax lot: {lot.symbol} {lot.quantity}@{lot.purchase_price}")

    def calculate_tax_liability(self, gains_to_realize: Dict[str, float]) -> float:
        """
        Calculate tax liability for realizing gains.

        Args:
            gains_to_realize: Dict of {symbol: amount_to_realize}

        Returns:
            Total tax liability
        """
        total_tax = 0

        for symbol, amount in gains_to_realize.items():
            if symbol not in self.tax_lots:
                continue

            # Use FIFO method to select which lots to sell
            lots = self.tax_lots[symbol]
            remaining = amount

            for lot in lots:
                if remaining <= 0:
                    break

                sell_qty = min(lot.quantity, remaining / lot.current_price)
                gain = sell_qty * (lot.current_price - lot.purchase_price)

                # Apply appropriate tax rate
                tax_rate = self.long_term_rate if lot.long_term else self.short_term_rate
                total_tax += gain * tax_rate

                remaining -= sell_qty * lot.current_price

        return total_tax

    def identify_harvesting_opportunities(
        self,
        loss_threshold: float = -0.05,
        min_loss_amount: float = 100
    ) -> List[TaxHarvest]:
        """
        Identify tax-loss harvesting opportunities.

        Args:
            loss_threshold: Loss threshold percentage (-0.05 = -5%)
            min_loss_amount: Minimum loss amount in dollars

        Returns:
            List of TaxHarvest opportunities
        """
        harvests = []

        for symbol, lots in self.tax_lots.items():
            for lot in lots:
                # Check for unrealized loss
                if lot.unrealized_gain_pct < loss_threshold and lot.unrealized_gain < -min_loss_amount:
                    harvest = TaxHarvest(
                        symbol=symbol,
                        quantity=lot.quantity,
                        current_price=lot.current_price,
                        realizable_loss=abs(lot.unrealized_gain),
                        days_until_wash_clear=self._days_until_wash_clear(symbol)
                    )
                    harvests.append(harvest)

        return sorted(harvests, key=lambda x: x.realizable_loss, reverse=True)

    def _days_until_wash_clear(self, symbol: str) -> int:
        """
        Calculate days until wash sale rule expires for a symbol.

        Args:
            symbol: Security symbol

        Returns:
            Days until wash sale expires (0 if not in wash sale)
        """
        today = pd.Timestamp.now()

        # Check recent sales
        for sale in reversed(self.sale_history):
            if sale['symbol'] == symbol and sale['type'] == 'loss':
                sale_date = pd.Timestamp(sale['date'])
                days_elapsed = (today - sale_date).days
                days_remaining = self.wash_sale_days - days_elapsed

                if days_remaining > 0:
                    return days_remaining

        return 0

    def find_replacement_security(
        self,
        symbol: str,
        correlation_threshold: float = 0.7,
        correlation_matrix: Optional[np.ndarray] = None,
        symbols: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Find a replacement security to avoid wash sale.

        Args:
            symbol: Original symbol
            correlation_threshold: Minimum correlation for replacement
            correlation_matrix: Correlation matrix between symbols
            symbols: List of available symbols

        Returns:
            Replacement symbol or None
        """
        if correlation_matrix is None or symbols is None:
            return None

        symbol_idx = symbols.index(symbol) if symbol in symbols else -1
        if symbol_idx < 0:
            return None

        correlations = correlation_matrix[symbol_idx]

        # Find similar but not identical securities
        for i, corr in enumerate(correlations):
            if symbols[i] != symbol and corr >= correlation_threshold and corr < 0.95:
                return symbols[i]

        return None

    def optimize_rebalancing(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_prices: np.ndarray,
        asset_names: List[str],
        current_positions: Dict[str, float],
        tax_consideration: float = 0.5
    ) -> TaxOptimizationResult:
        """
        Optimize rebalancing with tax considerations.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_prices: Current prices
            asset_names: Asset names
            current_positions: Current positions {symbol: qty}
            tax_consideration: Weight for tax savings vs accuracy (0-1)

        Returns:
            TaxOptimizationResult
        """
        account_value = 100000  # Default

        # Calculate current tax liability
        gains = {}
        for i, name in enumerate(asset_names):
            if name in self.tax_lots and len(self.tax_lots[name]) > 0:
                total_gain = sum(lot.unrealized_gain for lot in self.tax_lots[name])
                gains[name] = total_gain

        current_taxes = self.calculate_tax_liability(gains)

        # Identify harvesting opportunities
        harvests = self.identify_harvesting_opportunities()

        # Calculate rebalancing trades
        weight_drift = target_weights - current_weights
        trades = []

        for i, drift in enumerate(weight_drift):
            if abs(drift) > 0.01:
                trade_value = drift * account_value
                trade_qty = trade_value / current_prices[i]

                trades.append({
                    'asset': asset_names[i],
                    'action': 'buy' if drift > 0 else 'sell',
                    'quantity': abs(trade_qty),
                    'value': abs(trade_value)
                })

        # Calculate optimized taxes
        sell_values = {t['asset']: t['value'] for t in trades if t['action'] == 'sell'}
        optimized_taxes = self.calculate_tax_liability(sell_values)

        tax_savings = max(0, current_taxes - optimized_taxes)

        result = TaxOptimizationResult(
            current_taxes=current_taxes,
            optimized_taxes=optimized_taxes,
            tax_savings=tax_savings,
            harvests=harvests,
            rebalancing_trades=trades,
            realized_gains=sum(g for g in gains.values() if g > 0),
            realized_losses=sum(g for g in gains.values() if g < 0)
        )

        result.net_gains = result.realized_gains + result.realized_losses

        return result

    def recommend_harvests(
        self,
        max_harvests: int = 5,
        loss_threshold: float = -0.05
    ) -> List[TaxHarvest]:
        """
        Get top tax-loss harvesting recommendations.

        Args:
            max_harvests: Maximum number to recommend
            loss_threshold: Minimum loss threshold

        Returns:
            List of recommended TaxHarvest operations
        """
        harvests = self.identify_harvesting_opportunities(loss_threshold=loss_threshold)
        return harvests[:max_harvests]

    def record_sale(
        self,
        symbol: str,
        quantity: float,
        price: float,
        gain_loss: float,
        sale_type: str = 'regular'
    ) -> None:
        """
        Record a sale in history.

        Args:
            symbol: Security symbol
            quantity: Quantity sold
            price: Sale price
            gain_loss: Realized gain/loss
            sale_type: 'regular' or 'loss' (for wash sale tracking)
        """
        self.sale_history.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'gain_loss': gain_loss,
            'type': sale_type,
            'date': datetime.now().isoformat()
        })

        # Remove sold lots from tracking
        if symbol in self.tax_lots:
            qty_to_remove = quantity
            for lot in self.tax_lots[symbol]:
                if qty_to_remove <= 0:
                    break
                remove = min(lot.quantity, qty_to_remove)
                lot.quantity -= remove
                qty_to_remove -= remove

            # Remove empty lots
            self.tax_lots[symbol] = [lot for lot in self.tax_lots[symbol] if lot.quantity > 0]

    def get_tax_summary(self) -> Dict:
        """
        Get tax summary for all positions.

        Returns:
            Dictionary with tax summary metrics
        """
        total_unrealized_gain = 0
        total_unrealized_loss = 0
        num_long_term = 0
        num_short_term = 0

        for symbol, lots in self.tax_lots.items():
            for lot in lots:
                if lot.unrealized_gain > 0:
                    total_unrealized_gain += lot.unrealized_gain
                    if lot.long_term:
                        num_long_term += 1
                    else:
                        num_short_term += 1
                else:
                    total_unrealized_loss += lot.unrealized_gain

        potential_tax_on_gains = (total_unrealized_gain * self.long_term_rate +
                                 total_unrealized_loss * self.long_term_rate)

        return {
            'total_unrealized_gain': total_unrealized_gain,
            'total_unrealized_loss': total_unrealized_loss,
            'net_unrealized': total_unrealized_gain + total_unrealized_loss,
            'long_term_positions': num_long_term,
            'short_term_positions': num_short_term,
            'potential_tax_liability': potential_tax_on_gains,
            'num_harvestable_losses': len(self.identify_harvesting_opportunities())
        }
