"""
API Routers Package

Contains FastAPI routers for:
- Trading operations
- Performance analytics
- Model management
- Configuration
"""

from . import trading, performance, models, config

__all__ = ["trading", "performance", "models", "config"]
