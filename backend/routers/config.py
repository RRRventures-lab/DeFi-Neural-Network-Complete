"""
Configuration & Settings Router

Handles:
- Trading configuration
- Risk limits
- Symbol watchlist
- API status
- System settings
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock configuration state
config_state = {
    "trading_config": {
        "initial_capital": 100000,
        "max_position_size": 0.05,
        "max_positions": 20,
        "max_leverage": 1.0,
        "use_stop_loss": True,
        "stop_loss_percent": 0.02,
        "use_take_profit": True,
        "take_profit_percent": 0.05,
        "rebalance_frequency": "daily",
        "enable_shorting": False,
    },
    "risk_limits": {
        "max_drawdown_percent": 0.15,
        "max_daily_loss_percent": 0.02,
        "max_position_concentration": 0.1,
        "min_position_size": 100,
        "var_confidence_level": 0.95,
        "max_sector_allocation": 0.3,
    },
    "watchlist": [
        "BTC",
        "ETH",
        "AAPL",
        "GOOGL",
        "MSFT",
    ],
    "data_sources": {
        "primary": "polygon.io",
        "backup": "yfinance",
    },
}


@router.get("/trading")
async def get_trading_config():
    """Get current trading configuration."""
    return {
        "config": config_state["trading_config"],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/trading")
async def update_trading_config(updates: Dict):
    """
    Update trading configuration.

    Args:
        updates: Configuration updates to apply

    Returns:
        Updated configuration
    """
    # Validate updates
    for key, value in updates.items():
        if key in config_state["trading_config"]:
            config_state["trading_config"][key] = value

    logger.info(f"Trading config updated: {updates}")

    return {
        "status": "updated",
        "config": config_state["trading_config"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/risk-limits")
async def get_risk_limits():
    """Get current risk limit configuration."""
    return {
        "limits": config_state["risk_limits"],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/risk-limits")
async def update_risk_limits(updates: Dict):
    """
    Update risk limit configuration.

    Args:
        updates: Risk limit updates to apply

    Returns:
        Updated risk limits
    """
    for key, value in updates.items():
        if key in config_state["risk_limits"]:
            if 0 <= value <= 1:  # Validate percentage values
                config_state["risk_limits"][key] = value

    logger.info(f"Risk limits updated: {updates}")

    return {
        "status": "updated",
        "limits": config_state["risk_limits"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/watchlist")
async def get_watchlist():
    """Get asset watchlist."""
    return {
        "watchlist": config_state["watchlist"],
        "count": len(config_state["watchlist"]),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/watchlist")
async def add_to_watchlist(symbol: str):
    """
    Add symbol to watchlist.

    Args:
        symbol: Symbol to add (e.g., 'BTC', 'AAPL')

    Returns:
        Updated watchlist
    """
    if symbol.upper() not in config_state["watchlist"]:
        config_state["watchlist"].append(symbol.upper())
        logger.info(f"Added {symbol} to watchlist")

    return {
        "status": "added",
        "symbol": symbol.upper(),
        "watchlist": config_state["watchlist"],
        "timestamp": datetime.now().isoformat(),
    }


@router.delete("/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    """
    Remove symbol from watchlist.

    Args:
        symbol: Symbol to remove

    Returns:
        Updated watchlist
    """
    if symbol.upper() in config_state["watchlist"]:
        config_state["watchlist"].remove(symbol.upper())
        logger.info(f"Removed {symbol} from watchlist")

    return {
        "status": "removed",
        "symbol": symbol.upper(),
        "watchlist": config_state["watchlist"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/api-status")
async def get_api_status():
    """Get status of all API connections."""
    return {
        "polygon_io": {
            "status": "connected",
            "latency_ms": 45,
            "last_update": datetime.now().isoformat(),
        },
        "broker_api": {
            "status": "connected",
            "latency_ms": 120,
            "last_update": datetime.now().isoformat(),
        },
        "data_cache": {
            "status": "healthy",
            "cached_symbols": 50,
            "cache_hit_rate": 0.78,
        },
        "websocket": {
            "status": "connected",
            "active_connections": 5,
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/api-reconnect/{api_name}")
async def reconnect_api(api_name: str):
    """
    Reconnect to a specific API.

    Args:
        api_name: Name of API to reconnect to

    Returns:
        Reconnection result
    """
    logger.info(f"Reconnecting to {api_name}")

    return {
        "status": "reconnecting",
        "api": api_name,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/data-sources")
async def get_data_sources():
    """Get configured data sources."""
    return {
        "data_sources": config_state["data_sources"],
        "available_sources": [
            {
                "name": "polygon.io",
                "type": "REST API",
                "rate_limit": "5 calls/minute",
                "coverage": ["stocks", "crypto", "forex"],
            },
            {
                "name": "yfinance",
                "type": "Web Scraper",
                "rate_limit": "Unlimited",
                "coverage": ["stocks", "crypto"],
            },
            {
                "name": "alpaca",
                "type": "WebSocket",
                "rate_limit": "Unlimited",
                "coverage": ["stocks"],
            },
        ],
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/data-sources")
async def update_data_source(source_name: str, is_primary: bool = True):
    """
    Change data source configuration.

    Args:
        source_name: Name of data source
        is_primary: Whether to use as primary source

    Returns:
        Updated data source config
    """
    if is_primary:
        config_state["data_sources"]["primary"] = source_name
    else:
        config_state["data_sources"]["backup"] = source_name

    logger.info(f"Data source updated: {source_name}")

    return {
        "status": "updated",
        "data_sources": config_state["data_sources"],
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/system-info")
async def get_system_info():
    """Get system information and status."""
    return {
        "system": {
            "version": "1.0.0",
            "python_version": "3.10",
            "uptime_seconds": 3600,
        },
        "performance": {
            "cpu_usage_percent": 25.5,
            "memory_usage_percent": 42.3,
            "memory_available_gb": 4.5,
        },
        "trading": {
            "positions_open": 5,
            "orders_pending": 2,
            "signals_today": 12,
            "trades_today": 8,
        },
        "cache": {
            "cached_items": 256,
            "cache_size_mb": 125,
            "hit_rate": 0.78,
        },
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/password-change")
async def change_password(current_password: str, new_password: str):
    """
    Change admin password.

    Args:
        current_password: Current password for verification
        new_password: New password to set

    Returns:
        Password change confirmation
    """
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # In production, would verify current password and update hash
    logger.info("Password change requested")

    return {
        "status": "updated",
        "message": "Password changed successfully",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/backup-config")
async def backup_configuration():
    """
    Export current configuration as backup.

    Returns:
        Configuration backup
    """
    backup = {
        "timestamp": datetime.now().isoformat(),
        "trading_config": config_state["trading_config"],
        "risk_limits": config_state["risk_limits"],
        "watchlist": config_state["watchlist"],
    }

    logger.info("Configuration backup created")

    return backup


@router.post("/restore-config")
async def restore_configuration(backup: Dict):
    """
    Restore configuration from backup.

    Args:
        backup: Configuration backup to restore

    Returns:
        Restoration result
    """
    try:
        config_state["trading_config"] = backup.get("trading_config", config_state["trading_config"])
        config_state["risk_limits"] = backup.get("risk_limits", config_state["risk_limits"])
        config_state["watchlist"] = backup.get("watchlist", config_state["watchlist"])

        logger.info("Configuration restored from backup")

        return {
            "status": "restored",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Configuration restore failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to restore configuration")
