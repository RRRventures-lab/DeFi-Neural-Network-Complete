"""
Forex Trader Module

Implements forex currency pair trading:
- Currency pair handling
- Bid-ask spreads
- Pip calculations
- Leverage & margin management
- Currency conversion
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurrencyPair:
    """Currency pair specification."""
    base_currency: str  # USD, EUR, GBP, JPY
    quote_currency: str
    bid_price: float
    ask_price: float
    spread: float = 0.0002  # 2 pips
    pip_size: float = 0.0001  # For majors
    min_lot_size: float = 0.01  # Minimum 0.01 lots
    max_lot_size: float = 10.0  # Maximum 10 lots
    margin_requirement: float = 0.02  # 50:1 leverage

    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread_in_pips(self) -> float:
        """Spread in pips."""
        return self.spread / self.pip_size

    def get_pip_value(self, lot_size: float, account_currency: str = "USD") -> float:
        """Get value of one pip."""
        if account_currency == self.quote_currency:
            return lot_size * 100000 * self.pip_size
        else:
            return lot_size * 100000 * self.pip_size / self.ask_price


@dataclass
class ForexPosition:
    """Forex position tracking."""
    pair: str  # e.g., "EUR/USD"
    lot_size: float
    entry_price: float
    current_price: float
    leverage: float = 1.0
    entry_date: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def margin_used(self) -> float:
        """Margin used for position."""
        return (self.lot_size * 100000 * self.entry_price) / self.leverage

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in pips."""
        return (self.current_price - self.entry_price) / 0.0001

    @property
    def unrealized_pnl_usd(self) -> float:
        """Unrealized P&L in USD."""
        pip_value = self.lot_size * 100000 * 0.0001
        return self.unrealized_pnl * pip_value


class ForexTrader:
    """
    Forex currency pair trader.
    """

    def __init__(self, account_balance: float = 10000, leverage: float = 50):
        """
        Initialize forex trader.

        Args:
            account_balance: Starting account balance
            leverage: Maximum leverage (50:1 standard)
        """
        self.account_balance = account_balance
        self.available_margin = account_balance
        self.max_leverage = leverage
        self.positions: Dict[str, ForexPosition] = {}
        self.price_data: Dict[str, CurrencyPair] = {}
        self.order_history: List[Dict] = []

        logger.info(f"ForexTrader initialized: ${account_balance} balance, {leverage}:1 leverage")

    def add_currency_pair(self, pair: CurrencyPair) -> None:
        """Add currency pair for trading."""
        pair_name = f"{pair.base_currency}/{pair.quote_currency}"
        self.price_data[pair_name] = pair
        logger.debug(f"Added pair: {pair_name}")

    def update_prices(self, pair_name: str, bid: float, ask: float) -> None:
        """Update prices for a pair."""
        if pair_name in self.price_data:
            self.price_data[pair_name].bid_price = bid
            self.price_data[pair_name].ask_price = ask

    def open_position(
        self,
        pair: str,
        lot_size: float,
        entry_price: float,
        leverage: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Open a forex position.

        Args:
            pair: Currency pair (e.g., "EUR/USD")
            lot_size: Lot size (0.01 = 1000 units)
            entry_price: Entry price
            leverage: Leverage to use

        Returns:
            (success, position_id)
        """
        # Check margin requirements
        margin_required = (lot_size * 100000 * entry_price) / leverage

        if margin_required > self.available_margin:
            logger.warning(f"Insufficient margin for {pair}")
            return False, "insufficient_margin"

        # Create position
        position = ForexPosition(
            pair=pair,
            lot_size=lot_size,
            entry_price=entry_price,
            current_price=entry_price,
            leverage=leverage
        )

        self.positions[pair] = position
        self.available_margin -= margin_required

        logger.info(f"Opened {pair} position: {lot_size} lots @ {entry_price}")

        return True, f"pos_{len(self.positions)}"

    def close_position(self, pair: str, exit_price: float) -> Tuple[bool, float]:
        """
        Close a forex position.

        Args:
            pair: Currency pair
            exit_price: Exit price

        Returns:
            (success, pnl_usd)
        """
        if pair not in self.positions:
            return False, 0

        position = self.positions[pair]

        # Calculate P&L
        pip_diff = (exit_price - position.entry_price) / 0.0001
        pip_value = position.lot_size * 100000 * 0.0001
        pnl = pip_diff * pip_value

        # Release margin
        margin_used = (position.lot_size * 100000 * position.entry_price) / position.leverage
        self.available_margin += margin_used
        self.account_balance += pnl

        # Remove position
        del self.positions[pair]

        logger.info(f"Closed {pair} position: P&L = ${pnl:.2f}")

        return True, pnl

    def set_stop_loss(self, pair: str, stop_price: float) -> None:
        """Set stop-loss for position."""
        if pair in self.positions:
            self.positions[pair].stop_price = stop_price

    def set_take_profit(self, pair: str, tp_price: float) -> None:
        """Set take-profit for position."""
        if pair in self.positions:
            self.positions[pair].take_profit = tp_price

    def get_position(self, pair: str) -> Optional[ForexPosition]:
        """Get position details."""
        return self.positions.get(pair)

    def get_all_positions(self) -> Dict[str, ForexPosition]:
        """Get all open positions."""
        return self.positions.copy()

    def calculate_portfolio_equity(self) -> float:
        """Calculate total equity (balance + unrealized P&L)."""
        unrealized_pnl = sum(pos.unrealized_pnl_usd for pos in self.positions.values())
        return self.account_balance + unrealized_pnl

    def calculate_margin_level(self) -> float:
        """Calculate margin level (equity / margin used)."""
        if not self.positions:
            return float('inf')

        margin_used = sum(pos.margin_used for pos in self.positions.values())
        equity = self.calculate_portfolio_equity()

        if margin_used == 0:
            return float('inf')

        return (equity / margin_used) * 100

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        unrealized_loss = sum(
            min(0, pos.unrealized_pnl_usd) for pos in self.positions.values()
        )

        if self.account_balance == 0:
            return 0

        return abs(unrealized_loss) / self.account_balance

    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        exchange_rate: float
    ) -> float:
        """
        Convert between currencies.

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            exchange_rate: Exchange rate (to_currency per from_currency)

        Returns:
            Converted amount
        """
        return amount * exchange_rate

    def get_position_metrics(self, pair: str) -> Dict:
        """Get metrics for a position."""
        if pair not in self.positions:
            return {}

        position = self.positions[pair]

        return {
            'pair': pair,
            'lot_size': position.lot_size,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'unrealized_pnl_pips': position.unrealized_pnl,
            'unrealized_pnl_usd': position.unrealized_pnl_usd,
            'margin_used': position.margin_used,
            'leverage': position.leverage
        }

    def get_account_status(self) -> Dict:
        """Get account status."""
        return {
            'balance': self.account_balance,
            'equity': self.calculate_portfolio_equity(),
            'available_margin': self.available_margin,
            'margin_level': self.calculate_margin_level(),
            'drawdown_pct': self.calculate_drawdown(),
            'open_positions': len(self.positions),
            'max_leverage': self.max_leverage
        }

    def check_margin_call(self) -> bool:
        """Check if account is in margin call (level < 100%)."""
        margin_level = self.calculate_margin_level()
        return margin_level < 100 if margin_level != float('inf') else False

    def check_stop_loss(self, pair: str, current_price: float) -> bool:
        """Check if stop-loss is hit."""
        if pair not in self.positions:
            return False

        position = self.positions[pair]
        if hasattr(position, 'stop_price'):
            return current_price <= position.stop_price

        return False

    def check_take_profit(self, pair: str, current_price: float) -> bool:
        """Check if take-profit is hit."""
        if pair not in self.positions:
            return False

        position = self.positions[pair]
        if hasattr(position, 'take_profit'):
            return current_price >= position.take_profit

        return False
