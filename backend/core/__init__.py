"""
Core Module Integration

Wraps existing trading engine modules for API access.
Provides unified interface to:
- Trading engine
- Performance monitor
- Risk management
- Model inference
"""

from .trading_engine_wrapper import TradingEngineWrapper
from .performance_wrapper import PerformanceMonitorWrapper
from .models_wrapper import ModelsWrapper

__all__ = ["TradingEngineWrapper", "PerformanceMonitorWrapper", "ModelsWrapper"]
