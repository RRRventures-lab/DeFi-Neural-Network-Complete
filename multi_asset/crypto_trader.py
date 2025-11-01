"""
Cryptocurrency Trader Module

Implements multi-exchange cryptocurrency trading:
- Exchange integration (Binance, Kraken, Coinbase)
- Order execution and management
- Position tracking
- Fee handling
- Portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CryptoOrder:
    """Cryptocurrency order specification."""
    symbol: str  # BTC, ETH, ADA, etc.
    order_type: str  # 'market', 'limit', 'stop_loss'
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop-loss
    fee_percent: float = 0.001  # Default 0.1% fee
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def calculate_total_cost(self) -> float:
        """Calculate total cost including fees."""
        if self.price is None:
            return 0
        base_cost = self.quantity * self.price
        fees = base_cost * self.fee_percent
        return base_cost + fees if self.side == 'buy' else base_cost - fees


@dataclass
class CryptoPosition:
    """Cryptocurrency position tracking."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_date: str
    fees_paid: float = 0

    @property
    def unrealized_gain(self) -> float:
        """Unrealized gain/loss in USD."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def roi(self) -> float:
        """Return on investment percentage."""
        return self.unrealized_pnl_pct - (self.fees_paid / (self.entry_price * self.quantity))


@dataclass
class CryptoPortfolio:
    """Cryptocurrency portfolio summary."""
    total_value: float
    total_invested: float
    total_gains: float
    positions: Dict[str, CryptoPosition]
    portfolio_pnl_pct: float
    allocation: Dict[str, float]  # {symbol: weight}


class CryptoExchange:
    """Base crypto exchange interface."""

    def __init__(self, exchange_name: str, api_key: str = "", api_secret: str = ""):
        """
        Initialize exchange.

        Args:
            exchange_name: Exchange name (binance, kraken, coinbase)
            api_key: API key
            api_secret: API secret
        """
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_connected = False

        logger.info(f"Initialized {exchange_name} exchange")

    def connect(self) -> bool:
        """Connect to exchange."""
        self.is_connected = True
        logger.info(f"Connected to {self.exchange_name}")
        return True

    def get_balance(self, symbol: str) -> float:
        """Get balance for symbol."""
        return 0.0

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        return {}

    def place_order(self, order: CryptoOrder) -> Dict:
        """Place an order."""
        return {}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return False

    def get_order_status(self, order_id: str) -> str:
        """Get order status."""
        return "unknown"


class CryptoTrader:
    """
    Multi-exchange cryptocurrency trader.
    """

    def __init__(self, exchanges: Optional[List[str]] = None):
        """
        Initialize crypto trader.

        Args:
            exchanges: List of exchange names to support
        """
        self.exchanges: Dict[str, CryptoExchange] = {}
        self.positions: Dict[str, CryptoPosition] = {}
        self.order_history: List[CryptoOrder] = []
        self.balances: Dict[str, float] = {}
        self.prices: Dict[str, float] = {}

        if exchanges:
            for exchange in exchanges:
                self.add_exchange(exchange)

        logger.info(f"CryptoTrader initialized with {len(self.exchanges)} exchanges")

    def add_exchange(self, exchange_name: str, api_key: str = "", api_secret: str = "") -> CryptoExchange:
        """
        Add an exchange.

        Args:
            exchange_name: Exchange name
            api_key: API key
            api_secret: API secret

        Returns:
            CryptoExchange instance
        """
        exchange = CryptoExchange(exchange_name, api_key, api_secret)
        exchange.connect()
        self.exchanges[exchange_name] = exchange

        logger.info(f"Added exchange: {exchange_name}")

        return exchange

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for symbols.

        Args:
            prices: Dict of {symbol: current_price}
        """
        self.prices.update(prices)

        # Update position prices
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

        logger.debug(f"Updated prices for {len(prices)} symbols")

    def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        fee_percent: float = 0.001,
        exchange: str = "binance"
    ) -> Tuple[bool, str]:
        """
        Execute market order.

        Args:
            symbol: Cryptocurrency symbol
            side: 'buy' or 'sell'
            quantity: Quantity to trade
            current_price: Current market price
            fee_percent: Trading fee percentage
            exchange: Exchange to use

        Returns:
            (success, order_id)
        """
        order = CryptoOrder(
            symbol=symbol,
            order_type='market',
            side=side,
            quantity=quantity,
            price=current_price,
            fee_percent=fee_percent
        )

        # Execute order
        if side == 'buy':
            self._buy(symbol, quantity, current_price, fee_percent)
        elif side == 'sell':
            self._sell(symbol, quantity, current_price, fee_percent)
        else:
            return False, "invalid_side"

        self.order_history.append(order)

        logger.info(f"Executed {side} order: {quantity} {symbol} @ ${current_price}")

        return True, f"order_{len(self.order_history)}"

    def _buy(self, symbol: str, quantity: float, price: float, fee_percent: float) -> None:
        """Internal buy implementation."""
        if symbol not in self.positions:
            self.positions[symbol] = CryptoPosition(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                entry_date=datetime.now().isoformat(),
                fees_paid=quantity * price * fee_percent
            )
        else:
            # Average price calculation
            old_pos = self.positions[symbol]
            new_avg_price = (
                (old_pos.quantity * old_pos.entry_price + quantity * price) /
                (old_pos.quantity + quantity)
            )
            old_pos.quantity += quantity
            old_pos.entry_price = new_avg_price
            old_pos.fees_paid += quantity * price * fee_percent

    def _sell(self, symbol: str, quantity: float, price: float, fee_percent: float) -> None:
        """Internal sell implementation."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.quantity -= quantity
            pos.fees_paid += quantity * price * fee_percent

            if pos.quantity <= 0:
                del self.positions[symbol]

    def get_position(self, symbol: str) -> Optional[CryptoPosition]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, CryptoPosition]:
        """Get all positions."""
        return self.positions.copy()

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total = 0
        for pos in self.positions.values():
            total += pos.quantity * pos.current_price
        return total

    def calculate_total_invested(self) -> float:
        """Calculate total invested amount."""
        total = 0
        for pos in self.positions.values():
            total += pos.quantity * pos.entry_price + pos.fees_paid
        return total

    def get_portfolio_summary(self) -> CryptoPortfolio:
        """Get complete portfolio summary."""
        total_value = self.calculate_portfolio_value()
        total_invested = self.calculate_total_invested()
        total_gains = total_value - total_invested

        # Calculate allocation
        allocation = {}
        for symbol, pos in self.positions.items():
            allocation[symbol] = (pos.quantity * pos.current_price) / total_value if total_value > 0 else 0

        portfolio_pnl = (total_value - total_invested) / total_invested if total_invested > 0 else 0

        return CryptoPortfolio(
            total_value=total_value,
            total_invested=total_invested,
            total_gains=total_gains,
            positions=self.positions.copy(),
            portfolio_pnl_pct=portfolio_pnl,
            allocation=allocation
        )

    def rebalance_portfolio(
        self,
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Tuple[str, str, float]]:
        """
        Calculate rebalancing trades.

        Args:
            target_allocation: {symbol: target_weight}
            current_prices: {symbol: current_price}

        Returns:
            List of (symbol, side, quantity) trades
        """
        portfolio = self.get_portfolio_summary()
        trades = []

        for symbol, target_weight in target_allocation.items():
            current_weight = portfolio.allocation.get(symbol, 0)
            drift = target_weight - current_weight

            if abs(drift) > 0.01:  # Only trade if drift > 1%
                portfolio_value = portfolio.total_value
                target_value = portfolio_value * target_weight
                current_value = portfolio_value * current_weight
                trade_value = target_value - current_value

                if trade_value > 0:
                    # Buy
                    quantity = trade_value / current_prices[symbol]
                    trades.append((symbol, 'buy', quantity))
                else:
                    # Sell
                    quantity = abs(trade_value) / current_prices[symbol]
                    trades.append((symbol, 'sell', quantity))

        return trades

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[CryptoOrder]:
        """Get order history."""
        orders = self.order_history[-limit:]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from closed positions."""
        # This would require tracking closed positions
        # Simplified version just returns 0
        return 0.0

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        portfolio = self.get_portfolio_summary()

        return {
            'total_value': portfolio.total_value,
            'total_invested': portfolio.total_invested,
            'total_gains': portfolio.total_gains,
            'portfolio_pnl_pct': portfolio.portfolio_pnl_pct,
            'num_positions': len(portfolio.positions),
            'largest_position': max(portfolio.allocation.values()) if portfolio.allocation else 0
        }
