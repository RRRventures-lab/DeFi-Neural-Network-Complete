"""
Risk Management Module

Implements portfolio optimization, risk limits, position sizing,
and hedging strategies for the DeFi Neural Network trading system.
"""

from .portfolio_optimization import (
    EfficientFrontier,
    PortfolioOptimizer,
    OptimizationResult
)
from .risk_limits import (
    RiskLimits,
    DrawdownLimit,
    ConcentrationLimit,
    VolatilityLimit,
    VaRLimit
)
from .position_sizing import (
    PositionSizer,
    KellyPositionSizer,
    FixedFractionSizer,
    VolatilityTargetSizer,
    RiskParitySizer
)
from .hedging import (
    HedgingStrategy,
    CorrelationHedge,
    VaRBasedHedge,
    OptionsHedge,
    DynamicHedge
)
from .portfolio_manager import (
    PortfolioManager,
    RiskAdjustedAllocation
)

__all__ = [
    'EfficientFrontier',
    'PortfolioOptimizer',
    'OptimizationResult',
    'RiskLimits',
    'DrawdownLimit',
    'ConcentrationLimit',
    'VolatilityLimit',
    'VaRLimit',
    'PositionSizer',
    'KellyPositionSizer',
    'FixedFractionSizer',
    'VolatilityTargetSizer',
    'RiskParitySizer',
    'HedgingStrategy',
    'CorrelationHedge',
    'VaRBasedHedge',
    'OptionsHedge',
    'DynamicHedge',
    'PortfolioManager',
    'RiskAdjustedAllocation',
]
