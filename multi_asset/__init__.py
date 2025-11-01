"""
Multi-Asset Trading Module

Implements trading across multiple asset classes:
- Cryptocurrency (multi-exchange)
- Forex (currency pairs)
- Derivatives (futures, options)
- Cross-asset correlation analysis
- Multi-asset portfolio management
- Systemic risk monitoring
"""

from .crypto_trader import (
    CryptoOrder,
    CryptoPosition,
    CryptoPortfolio,
    CryptoExchange,
    CryptoTrader,
)

from .forex_trader import (
    CurrencyPair,
    ForexPosition,
    ForexTrader,
)

from .derivatives_trader import (
    FuturesContract,
    FuturesPosition,
    DerivativesTrader,
)

from .asset_correlation import (
    CorrelationMetric,
    BetaMetric,
    DiversificationMetric,
    AssetCorrelationAnalyzer,
)

from .multi_asset_portfolio import (
    AssetClassAllocation,
    PortfolioHolding,
    PortfolioMetrics,
    RebalancingTrade,
    MultiAssetPortfolio,
)

from .multi_asset_risk import (
    SystemicRiskMetric,
    ConcentrationRiskMetric,
    StressTestResult,
    RiskLimitViolation,
    MultiAssetRiskManager,
)

__all__ = [
    # Crypto
    "CryptoOrder",
    "CryptoPosition",
    "CryptoPortfolio",
    "CryptoExchange",
    "CryptoTrader",
    # Forex
    "CurrencyPair",
    "ForexPosition",
    "ForexTrader",
    # Derivatives
    "FuturesContract",
    "FuturesPosition",
    "DerivativesTrader",
    # Correlation
    "CorrelationMetric",
    "BetaMetric",
    "DiversificationMetric",
    "AssetCorrelationAnalyzer",
    # Portfolio
    "AssetClassAllocation",
    "PortfolioHolding",
    "PortfolioMetrics",
    "RebalancingTrade",
    "MultiAssetPortfolio",
    # Risk
    "SystemicRiskMetric",
    "ConcentrationRiskMetric",
    "StressTestResult",
    "RiskLimitViolation",
    "MultiAssetRiskManager",
]
