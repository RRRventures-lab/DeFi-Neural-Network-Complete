"""
Core Trading Engine

Orchestrates all trading components:
- Signal generation
- Position management
- Risk control
- Order execution
- Performance tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Trading engine configuration."""
    initial_capital: float = 100000
    max_positions: int = 20
    max_position_size_pct: float = 0.05
    max_leverage: float = 2.0
    risk_per_trade_pct: float = 0.02
    max_drawdown_pct: float = 0.15
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    enable_risk_limits: bool = True
    enable_live_trading: bool = False
    enable_paper_trading: bool = True


@dataclass
class EngineState:
    """Current state of trading engine."""
    is_running: bool = False
    is_connected: bool = False
    current_capital: float = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0
    peak_equity: float = 0
    current_drawdown: float = 0
    risk_violations: List[str] = field(default_factory=list)
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


class TradingEngine:
    """
    Integrated trading engine orchestrating all system components.
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize trading engine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self.state = EngineState(current_capital=config.initial_capital, peak_equity=config.initial_capital)
        self.positions: Dict[str, Dict] = {}
        self.pending_orders: List[Dict] = []
        self.executed_trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.signal_queue: List[Dict] = []

        logger.info(f"TradingEngine initialized: ${config.initial_capital} capital")

    def start(self) -> bool:
        """Start trading engine."""
        self.state.is_running = True
        logger.info("Trading engine started")
        return True

    def stop(self) -> bool:
        """Stop trading engine."""
        self.state.is_running = False
        logger.info("Trading engine stopped")
        return True

    def connect_data_source(self, source_name: str) -> bool:
        """Connect to market data source."""
        logger.info(f"Connected to {source_name}")
        self.state.is_connected = True
        return True

    def add_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        asset_class: str = "stock",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add trading signal to queue.

        Args:
            symbol: Asset symbol
            signal_type: 'buy', 'sell', 'hold'
            strength: Signal strength (0-1)
            asset_class: Asset class (stock, crypto, forex, etc.)
            metadata: Additional signal metadata
        """
        signal = {
            "symbol": symbol,
            "type": signal_type,
            "strength": strength,
            "asset_class": asset_class,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.signal_queue.append(signal)
        logger.debug(f"Added signal: {symbol} {signal_type} (strength: {strength:.2f})")

    def process_signals(self) -> List[Dict]:
        """
        Process pending signals and generate orders.

        Returns:
            List of generated orders
        """
        orders = []

        for signal in self.signal_queue:
            if signal["type"] == "buy":
                order = self._create_buy_order(signal)
            elif signal["type"] == "sell":
                order = self._create_sell_order(signal)
            else:
                continue

            if order and self._validate_order(order):
                orders.append(order)
                self.pending_orders.append(order)

        self.signal_queue.clear()
        logger.info(f"Processed {len(orders)} orders from signals")

        return orders

    def _create_buy_order(self, signal: Dict) -> Optional[Dict]:
        """Create buy order from signal."""
        symbol = signal["symbol"]
        strength = signal["strength"]

        # Position sizing based on signal strength
        risk_amount = self.state.current_capital * self.config.risk_per_trade_pct
        position_size = risk_amount * strength

        return {
            "id": f"order_{len(self.pending_orders)}",
            "symbol": symbol,
            "side": "buy",
            "size": position_size,
            "asset_class": signal.get("asset_class", "stock"),
            "timestamp": datetime.now().isoformat(),
            "signal_strength": strength,
            "status": "pending",
        }

    def _create_sell_order(self, signal: Dict) -> Optional[Dict]:
        """Create sell order from signal."""
        symbol = signal["symbol"]

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]

        return {
            "id": f"order_{len(self.pending_orders)}",
            "symbol": symbol,
            "side": "sell",
            "size": position["quantity"],
            "asset_class": signal.get("asset_class", "stock"),
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }

    def _validate_order(self, order: Dict) -> bool:
        """Validate order against risk limits."""
        if not self.config.enable_risk_limits:
            return True

        # Check position count
        if len(self.positions) >= self.config.max_positions and order["side"] == "buy":
            logger.warning(f"Max positions ({self.config.max_positions}) reached")
            self.state.risk_violations.append("Max positions exceeded")
            return False

        # Check position size
        max_size = self.state.current_capital * self.config.max_position_size_pct
        if order["size"] > max_size:
            logger.warning(f"Position size {order['size']} exceeds max {max_size}")
            self.state.risk_violations.append("Position size exceeded")
            return False

        # Check capital availability
        if order["side"] == "buy" and order["size"] > self.state.current_capital * 0.3:
            logger.warning("Insufficient capital for order")
            self.state.risk_violations.append("Insufficient capital")
            return False

        return True

    def execute_order(self, order: Dict, execution_price: float) -> bool:
        """
        Execute pending order.

        Args:
            order: Order to execute
            execution_price: Execution price

        Returns:
            Success status
        """
        symbol = order["symbol"]

        if order["side"] == "buy":
            # Open or add to position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": order["size"],
                    "entry_price": execution_price,
                    "entry_date": datetime.now().isoformat(),
                    "asset_class": order.get("asset_class", "stock"),
                }
            else:
                # Average price
                pos = self.positions[symbol]
                total_quantity = pos["quantity"] + order["size"]
                pos["entry_price"] = (
                    (pos["quantity"] * pos["entry_price"] + order["size"] * execution_price)
                    / total_quantity
                )
                pos["quantity"] = total_quantity

            # Deduct from capital
            self.state.current_capital -= order["size"]

        elif order["side"] == "sell":
            if symbol in self.positions:
                position = self.positions[symbol]
                pnl = (execution_price - position["entry_price"]) * order["size"]

                # Record trade
                trade = {
                    "symbol": symbol,
                    "entry_price": position["entry_price"],
                    "exit_price": execution_price,
                    "quantity": order["size"],
                    "pnl": pnl,
                    "pnl_pct": pnl / (position["entry_price"] * order["size"]) if position["entry_price"] > 0 else 0,
                    "entry_date": position["entry_date"],
                    "exit_date": datetime.now().isoformat(),
                }

                self.executed_trades.append(trade)
                self.state.total_trades += 1

                if pnl > 0:
                    self.state.winning_trades += 1
                else:
                    self.state.losing_trades += 1

                self.state.total_pnl += pnl

                # Add proceeds to capital
                self.state.current_capital += execution_price * order["size"]

                # Remove position
                del self.positions[symbol]

        # Update state
        order["status"] = "executed"
        order["execution_price"] = execution_price
        order["execution_time"] = datetime.now().isoformat()

        logger.info(
            f"Executed {order['side']} order: {symbol} "
            f"{order['size']} @ ${execution_price:.2f}"
        )

        return True

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices and calculate unrealized P&L.

        Args:
            prices: {symbol: current_price}
        """
        total_unrealized_pnl = 0

        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                unrealized_pnl = (current_price - position["entry_price"]) * position["quantity"]
                position["current_price"] = current_price
                position["unrealized_pnl"] = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl

        # Calculate equity and drawdown
        equity = self.state.current_capital + total_unrealized_pnl
        drawdown = (self.state.peak_equity - equity) / self.state.peak_equity if self.state.peak_equity > 0 else 0

        self.state.current_drawdown = max(0, drawdown)

        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # Check drawdown limit
        if self.state.current_drawdown > self.config.max_drawdown_pct:
            self.state.risk_violations.append(f"Drawdown exceeded: {self.state.current_drawdown:.2%}")
            logger.warning(f"Drawdown limit exceeded: {self.state.current_drawdown:.2%}")

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        unrealized_pnl = sum(
            pos.get("unrealized_pnl", 0) for pos in self.positions.values()
        )
        return self.state.current_capital + unrealized_pnl

    def get_engine_summary(self) -> Dict:
        """Get comprehensive engine summary."""
        total_trades = self.state.total_trades if self.state.total_trades > 0 else 1
        win_rate = self.state.winning_trades / total_trades

        return {
            "state": {
                "is_running": self.state.is_running,
                "is_connected": self.state.is_connected,
                "current_capital": self.state.current_capital,
                "portfolio_value": self.get_portfolio_value(),
                "peak_equity": self.state.peak_equity,
                "current_drawdown": self.state.current_drawdown,
            },
            "performance": {
                "total_trades": self.state.total_trades,
                "winning_trades": self.state.winning_trades,
                "losing_trades": self.state.losing_trades,
                "win_rate": win_rate,
                "total_pnl": self.state.total_pnl,
                "avg_pnl_per_trade": self.state.total_pnl / total_trades,
            },
            "positions": {
                "open_count": len(self.positions),
                "open_positions": {
                    symbol: {
                        "quantity": pos["quantity"],
                        "entry_price": pos["entry_price"],
                        "current_price": pos.get("current_price", 0),
                        "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    }
                    for symbol, pos in self.positions.items()
                },
            },
            "orders": {
                "pending": len(self.pending_orders),
                "executed": len(self.executed_trades),
            },
            "risk": {
                "violations": len(self.state.risk_violations),
                "latest_violations": self.state.risk_violations[-5:],
            },
        }

    def reset_state(self) -> None:
        """Reset engine state for backtesting."""
        self.positions.clear()
        self.pending_orders.clear()
        self.executed_trades.clear()
        self.signal_queue.clear()
        self.portfolio_history.clear()
        self.state = EngineState(
            current_capital=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
        )
        logger.info("Engine state reset")
