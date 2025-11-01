"""
Trading Engine Module

Integrated trading system combining all stages:
- Data pipeline (Stage 1)
- Feature engineering (Stage 2)
- Signal generation (Stage 3)
- Backtesting (Stage 4)
- Live trading (Stage 5)
- Risk management (Stage 6)
- Advanced features (Stage 7)
- Multi-asset trading (Stage 8)
- Integrated execution (Stage 9)
"""

from .trading_engine import (
    TradingEngine,
    EngineConfig,
    EngineState,
)

from .execution_manager import (
    ExecutionManager,
    Order,
    OrderStatus,
    Trade,
    ExecutionStats,
)

from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    DailyReturn,
    TradeAnalysis,
)

from .deployment_manager import (
    DeploymentManager,
    DeploymentConfig,
    SystemStatus,
    HealthCheck,
)

from .trading_agent import (
    TradingAgent,
    AgentDecision,
    AgentState,
)

__all__ = [
    # Engine
    "TradingEngine",
    "EngineConfig",
    "EngineState",
    # Execution
    "ExecutionManager",
    "Order",
    "OrderStatus",
    "Trade",
    "ExecutionStats",
    # Performance
    "PerformanceMonitor",
    "PerformanceMetrics",
    "DailyReturn",
    "TradeAnalysis",
    # Deployment
    "DeploymentManager",
    "DeploymentConfig",
    "SystemStatus",
    "HealthCheck",
    # Agent
    "TradingAgent",
    "AgentDecision",
    "AgentState",
]
