"""
Trading Operations Router

Handles:
- Engine start/stop
- Signal processing
- Order execution
- Position management
- Portfolio monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock engine state (in production, would connect to real trading_engine module)
engine_state = {
    "running": False,
    "initial_capital": 100000,
    "current_value": 100000,
    "open_positions": [],
    "total_pnl": 0,
    "trading_mode": "paper",
    "signals_processed": 0,
    "orders_executed": 0,
}


@router.get("/status")
async def get_engine_status():
    """Get current trading engine status."""
    return {
        "running": engine_state["running"],
        "trading_mode": engine_state["trading_mode"],
        "initial_capital": engine_state["initial_capital"],
        "current_value": engine_state["current_value"],
        "total_pnl": engine_state["total_pnl"],
        "pnl_percent": (engine_state["total_pnl"] / engine_state["initial_capital"]) * 100,
        "open_positions_count": len(engine_state["open_positions"]),
        "signals_processed": engine_state["signals_processed"],
        "orders_executed": engine_state["orders_executed"],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/start")
async def start_engine(mode: str = "paper"):
    """
    Start the trading engine.

    Args:
        mode: Trading mode - 'paper' or 'live' (default: 'paper')

    Returns:
        Engine started status
    """
    if engine_state["running"]:
        raise HTTPException(status_code=400, detail="Engine already running")

    if mode not in ["paper", "live"]:
        raise HTTPException(status_code=400, detail="Invalid trading mode")

    engine_state["running"] = True
    engine_state["trading_mode"] = mode

    logger.info(f"Trading engine started in {mode} mode")

    return {
        "status": "started",
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/stop")
async def stop_engine():
    """Stop the trading engine."""
    if not engine_state["running"]:
        raise HTTPException(status_code=400, detail="Engine not running")

    engine_state["running"] = False

    logger.info("Trading engine stopped")

    return {
        "status": "stopped",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/positions")
async def get_positions():
    """Get all open positions."""
    return {
        "positions": engine_state["open_positions"],
        "count": len(engine_state["open_positions"]),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/signal")
async def process_signal(
    symbol: str,
    signal_type: str,
    strength: float,
    metadata: Optional[Dict] = None,
):
    """
    Process a trading signal.

    Args:
        symbol: Asset symbol (e.g., 'BTC', 'ETH')
        signal_type: Signal type - 'buy', 'sell', 'hold'
        strength: Signal strength (0-1)
        metadata: Additional signal metadata

    Returns:
        Signal processing result
    """
    if not engine_state["running"]:
        raise HTTPException(status_code=400, detail="Engine not running")

    if signal_type not in ["buy", "sell", "hold"]:
        raise HTTPException(status_code=400, detail="Invalid signal type")

    if not 0 <= strength <= 1:
        raise HTTPException(status_code=400, detail="Signal strength must be 0-1")

    engine_state["signals_processed"] += 1

    logger.info(f"Signal processed: {signal_type} {symbol} (strength: {strength})")

    return {
        "signal_id": f"sig_{engine_state['signals_processed']}",
        "symbol": symbol,
        "type": signal_type,
        "strength": strength,
        "status": "processed",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/order")
async def create_order(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    price: Optional[float] = None,
):
    """
    Create and execute a trading order.

    Args:
        symbol: Asset symbol
        side: Order side - 'buy' or 'sell'
        quantity: Order quantity
        order_type: Order type - 'market', 'limit', 'stop' (default: 'market')
        price: Limit/stop price for non-market orders

    Returns:
        Order execution result
    """
    if not engine_state["running"]:
        raise HTTPException(status_code=400, detail="Engine not running")

    if side not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Invalid order side")

    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    if order_type not in ["market", "limit", "stop"]:
        raise HTTPException(status_code=400, detail="Invalid order type")

    engine_state["orders_executed"] += 1

    # Mock position update
    if side == "buy":
        engine_state["open_positions"].append({
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": price or 1000,
            "entry_time": datetime.now().isoformat(),
        })

    logger.info(f"Order executed: {side} {quantity} {symbol} @{order_type}")

    return {
        "order_id": f"ord_{engine_state['orders_executed']}",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "type": order_type,
        "status": "executed",
        "filled_at": datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
    }


@router.delete("/positions/{symbol}")
async def close_position(symbol: str):
    """
    Close a specific position.

    Args:
        symbol: Symbol of position to close

    Returns:
        Position close result
    """
    if not engine_state["running"]:
        raise HTTPException(status_code=400, detail="Engine not running")

    position = next(
        (p for p in engine_state["open_positions"] if p["symbol"] == symbol),
        None,
    )

    if not position:
        raise HTTPException(status_code=404, detail=f"Position {symbol} not found")

    engine_state["open_positions"].remove(position)

    logger.info(f"Position closed: {symbol}")

    return {
        "symbol": symbol,
        "closed_quantity": position["quantity"],
        "closed_at": datetime.now().isoformat(),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/orders")
async def get_order_history(limit: int = 100):
    """
    Get recent order history.

    Args:
        limit: Number of recent orders to return

    Returns:
        List of recent orders
    """
    return {
        "orders": [
            {
                "order_id": f"ord_{i}",
                "symbol": "BTC",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": 0.5,
                "type": "market",
                "executed_at": datetime.now().isoformat(),
            }
            for i in range(min(limit, 10))
        ],
        "total": engine_state["orders_executed"],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/mode")
async def set_trading_mode(mode: str):
    """
    Switch trading mode.

    Args:
        mode: Trading mode - 'paper' or 'live'

    Returns:
        Mode change result
    """
    if mode not in ["paper", "live"]:
        raise HTTPException(status_code=400, detail="Invalid trading mode")

    engine_state["trading_mode"] = mode

    logger.info(f"Trading mode changed to: {mode}")

    return {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/portfolio")
async def get_portfolio_summary():
    """Get portfolio summary and allocation."""
    return {
        "initial_capital": engine_state["initial_capital"],
        "current_value": engine_state["current_value"],
        "total_pnl": engine_state["total_pnl"],
        "pnl_percent": (engine_state["total_pnl"] / engine_state["initial_capital"]) * 100,
        "positions": engine_state["open_positions"],
        "cash_available": engine_state["current_value"] - sum(p.get("quantity", 0) * p.get("entry_price", 0) for p in engine_state["open_positions"]),
        "allocation": {
            "stocks": 0.4,
            "crypto": 0.3,
            "forex": 0.2,
            "cash": 0.1,
        },
        "timestamp": datetime.now().isoformat(),
    }
