"""
Execution Manager

Handles order execution, routing, and trade management:
- Order management and lifecycle
- Trade execution
- Slippage calculation
- Execution statistics
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trade order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str = "market"  # market, limit, stop
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: str = field(default_factory=lambda: datetime.now().isoformat())
    filled_quantity: float = 0
    filled_price: float = 0
    slippage: float = 0


@dataclass
class Trade:
    """Executed trade."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: str = field(default_factory=lambda: datetime.now().isoformat())
    exit_time: Optional[str] = None
    pnl: float = 0
    pnl_pct: float = 0
    slippage: float = 0
    execution_quality: float = 1.0  # 0-1, 1 is perfect


@dataclass
class ExecutionStats:
    """Execution statistics."""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    avg_slippage: float = 0
    avg_fill_price_difference: float = 0
    total_commissions: float = 0
    execution_quality_score: float = 0


class ExecutionManager:
    """
    Manages order execution and trade management.
    """

    def __init__(self, broker: str = "paper", slippage_pct: float = 0.001):
        """
        Initialize execution manager.

        Args:
            broker: Broker name (paper, live, etc.)
            slippage_pct: Default slippage percentage
        """
        self.broker = broker
        self.slippage_pct = slippage_pct
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.execution_stats = ExecutionStats()
        self.order_counter = 0

        logger.info(f"ExecutionManager initialized: {broker} trading")

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """
        Create new order.

        Args:
            symbol: Asset symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market', 'limit', 'stop'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            Created Order object
        """
        order_id = f"{symbol}_{self.order_counter}"
        self.order_counter += 1

        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        self.orders[order_id] = order
        logger.info(f"Created order {order_id}: {side} {quantity} {symbol} ({order_type})")

        return order

    def execute_order(self, order_id: str, market_price: float) -> Optional[Trade]:
        """
        Execute order at market price.

        Args:
            order_id: Order ID to execute
            market_price: Current market price

        Returns:
            Executed Trade or None if order not found
        """
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        # Calculate slippage
        slippage = market_price * self.slippage_pct
        execution_price = market_price + slippage if order.side == "buy" else market_price - slippage

        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            entry_price=execution_price,
            slippage=slippage,
        )

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.slippage = slippage

        # Update statistics
        self.execution_stats.total_orders += 1
        self.execution_stats.filled_orders += 1
        self.execution_stats.avg_slippage = (
            (self.execution_stats.avg_slippage * (self.execution_stats.filled_orders - 1) + abs(slippage))
            / self.execution_stats.filled_orders
        )

        self.trades.append(trade)

        logger.info(
            f"Executed order {order_id}: {order.side} {order.quantity} "
            f"{order.symbol} @ ${execution_price:.2f} (slippage: ${slippage:.4f})"
        )

        return trade

    def close_trade(self, trade: Trade, exit_price: float) -> None:
        """
        Close an open trade.

        Args:
            trade: Trade to close
            exit_price: Exit price
        """
        # Calculate slippage on exit
        slippage = exit_price * self.slippage_pct if trade.side == "buy" else exit_price * self.slippage_pct
        adjusted_exit = exit_price - slippage if trade.side == "buy" else exit_price + slippage

        # Calculate P&L
        if trade.side == "buy":
            pnl = (adjusted_exit - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - adjusted_exit) * trade.quantity

        pnl_pct = pnl / (trade.entry_price * trade.quantity) if trade.entry_price > 0 else 0

        # Update trade
        trade.exit_price = adjusted_exit
        trade.exit_time = datetime.now().isoformat()
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.slippage += slippage

        logger.info(
            f"Closed trade: {trade.symbol} P&L: ${pnl:.2f} ({pnl_pct:.2%})"
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.CANCELLED
        self.execution_stats.total_orders += 1
        self.execution_stats.cancelled_orders += 1

        logger.info(f"Cancelled order {order_id}")

        return True

    def get_execution_quality(self) -> float:
        """Calculate execution quality score (0-1)."""
        if not self.trades:
            return 1.0

        # Quality based on slippage
        max_slippage = max(t.slippage for t in self.trades) if self.trades else 0
        avg_slippage = np.mean([t.slippage for t in self.trades])

        quality = 1.0 - (avg_slippage * 100)  # Slippage as percentage of price
        quality = max(0, min(1, quality))

        self.execution_stats.execution_quality_score = quality

        return quality

    def get_trade_analysis(self) -> Dict:
        """Get comprehensive trade analysis."""
        if not self.trades:
            return {"total_trades": 0}

        completed_trades = [t for t in self.trades if t.exit_price is not None]
        open_trades = [t for t in self.trades if t.exit_price is None]

        total_pnl = sum(t.pnl for t in completed_trades)
        winning_trades = len([t for t in completed_trades if t.pnl > 0])
        losing_trades = len([t for t in completed_trades if t.pnl < 0])

        return {
            "total_trades": len(self.trades),
            "completed_trades": len(completed_trades),
            "open_trades": len(open_trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / len(completed_trades) if completed_trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": total_pnl / len(completed_trades) if completed_trades else 0,
            "avg_slippage": self.execution_stats.avg_slippage,
            "execution_quality": self.get_execution_quality(),
        }

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders filtered by status."""
        return [o for o in self.orders.values() if o.status == status]

    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history."""
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "quantity": o.quantity,
                "filled_price": o.filled_price,
                "status": o.status,
                "created_time": o.created_time,
            }
            for o in list(self.orders.values())[-limit:]
        ]
