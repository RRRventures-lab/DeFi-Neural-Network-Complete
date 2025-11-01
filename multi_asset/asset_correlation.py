"""
Asset Correlation Module

Implements cross-asset correlation analysis:
- Correlation matrix calculation
- Rolling correlations
- Correlation breakdown detection
- Beta and covariance analysis
- Diversification metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetric:
    """Correlation metric snapshot."""
    timestamp: str
    asset_pair: Tuple[str, str]
    correlation: float
    rolling_avg: float
    volatility_asset1: float
    volatility_asset2: float
    covariance: float


@dataclass
class BetaMetric:
    """Beta metric for asset against market."""
    symbol: str
    beta: float
    r_squared: float
    alpha: float
    market_sensitivity: float


@dataclass
class DiversificationMetric:
    """Diversification metrics."""
    portfolio_variance: float
    weighted_individual_variance: float
    correlation_benefit: float
    diversification_ratio: float
    effective_n_assets: float  # Effective number of uncorrelated assets


class AssetCorrelationAnalyzer:
    """
    Analyzes correlation between assets.
    """

    def __init__(self, lookback_period: int = 252, rolling_window: int = 30):
        """
        Initialize correlation analyzer.

        Args:
            lookback_period: Number of days for historical analysis
            rolling_window: Window for rolling correlation
        """
        self.lookback_period = lookback_period
        self.rolling_window = rolling_window
        self.price_history: Dict[str, List[Tuple[str, float]]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.rolling_correlations: Dict[str, List[CorrelationMetric]] = {}

        logger.info(f"AssetCorrelationAnalyzer initialized: {lookback_period} day lookback")

    def add_price_history(self, symbol: str, prices: List[Tuple[str, float]]) -> None:
        """
        Add price history for asset.

        Args:
            symbol: Asset symbol
            prices: List of (timestamp, price) tuples
        """
        self.price_history[symbol] = prices
        logger.debug(f"Added price history for {symbol}: {len(prices)} data points")

    def calculate_returns(self, symbol: str) -> np.ndarray:
        """Calculate log returns from prices."""
        if symbol not in self.price_history:
            return np.array([])

        prices = np.array([p[1] for p in self.price_history[symbol]])
        if len(prices) < 2:
            return np.array([])

        return np.diff(np.log(prices))

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for all assets.

        Returns:
            Correlation matrix as DataFrame
        """
        returns_dict = {}

        for symbol in self.price_history.keys():
            returns = self.calculate_returns(symbol)
            if len(returns) > 0:
                returns_dict[symbol] = returns

        # Ensure all return series have same length
        min_length = min(len(r) for r in returns_dict.values())
        for symbol in returns_dict:
            returns_dict[symbol] = returns_dict[symbol][-min_length:]

        df = pd.DataFrame(returns_dict)
        self.correlation_matrix = df.corr()

        logger.info(f"Calculated correlation matrix for {len(returns_dict)} assets")

        return self.correlation_matrix

    def calculate_rolling_correlation(
        self, symbol1: str, symbol2: str
    ) -> List[CorrelationMetric]:
        """
        Calculate rolling correlation between two assets.

        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol

        Returns:
            List of rolling correlation metrics
        """
        returns1 = self.calculate_returns(symbol1)
        returns2 = self.calculate_returns(symbol2)

        if len(returns1) < self.rolling_window or len(returns2) < self.rolling_window:
            return []

        rolling_corr = []
        min_length = min(len(returns1), len(returns2))

        for i in range(min_length - self.rolling_window + 1):
            window1 = returns1[i : i + self.rolling_window]
            window2 = returns2[i : i + self.rolling_window]

            corr = np.corrcoef(window1, window2)[0, 1]
            rolling_avg = np.mean(
                [m.correlation for m in rolling_corr[-5:]] + [corr]
            )

            vol1 = np.std(window1) * np.sqrt(252)  # Annualized
            vol2 = np.std(window2) * np.sqrt(252)
            cov = np.cov(window1, window2)[0, 1] * 252

            metric = CorrelationMetric(
                timestamp=datetime.now().isoformat(),
                asset_pair=(symbol1, symbol2),
                correlation=corr if not np.isnan(corr) else 0,
                rolling_avg=rolling_avg if not np.isnan(rolling_avg) else 0,
                volatility_asset1=vol1,
                volatility_asset2=vol2,
                covariance=cov,
            )

            rolling_corr.append(metric)

        self.rolling_correlations[(symbol1, symbol2)] = rolling_corr
        logger.debug(
            f"Calculated {len(rolling_corr)} rolling correlation points for {symbol1}-{symbol2}"
        )

        return rolling_corr

    def detect_correlation_breakdown(
        self, symbol1: str, symbol2: str, threshold: float = 0.3
    ) -> Dict:
        """
        Detect correlation breakdown between assets.

        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            threshold: Breakdown threshold (drop in correlation)

        Returns:
            Dictionary with breakdown information
        """
        rolling = self.calculate_rolling_correlation(symbol1, symbol2)

        if len(rolling) < 10:
            return {"breakdown_detected": False}

        recent_corr = rolling[-1].correlation
        historical_corr = np.mean([m.correlation for m in rolling[:-1]])
        breakdown = historical_corr - recent_corr

        return {
            "breakdown_detected": breakdown > threshold,
            "breakdown_magnitude": breakdown,
            "historical_correlation": historical_corr,
            "recent_correlation": recent_corr,
            "asset_pair": (symbol1, symbol2),
        }

    def calculate_beta(
        self, symbol: str, market_returns: np.ndarray
    ) -> BetaMetric:
        """
        Calculate beta for asset against market.

        Args:
            symbol: Asset symbol
            market_returns: Market return series

        Returns:
            BetaMetric with beta, alpha, and R-squared
        """
        asset_returns = self.calculate_returns(symbol)

        if len(asset_returns) < len(market_returns):
            market_returns = market_returns[-len(asset_returns) :]
        elif len(market_returns) < len(asset_returns):
            asset_returns = asset_returns[-len(market_returns) :]

        # Remove NaN values
        valid_idx = ~(np.isnan(asset_returns) | np.isnan(market_returns))
        asset_returns = asset_returns[valid_idx]
        market_returns = market_returns[valid_idx]

        if len(asset_returns) < 2:
            return BetaMetric(
                symbol=symbol, beta=0, r_squared=0, alpha=0, market_sensitivity=0
            )

        # Calculate beta via covariance
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        beta = covariance / market_variance if market_variance > 0 else 0

        # Calculate R-squared
        correlation = np.corrcoef(asset_returns, market_returns)[0, 1]
        r_squared = correlation**2 if not np.isnan(correlation) else 0

        # Calculate alpha (excess return)
        alpha = np.mean(asset_returns) - beta * np.mean(market_returns)

        return BetaMetric(
            symbol=symbol,
            beta=beta,
            r_squared=r_squared,
            alpha=alpha,
            market_sensitivity=abs(beta),
        )

    def calculate_diversification_metrics(
        self, weights: Dict[str, float], covariance_matrix: Optional[np.ndarray] = None
    ) -> DiversificationMetric:
        """
        Calculate diversification metrics for portfolio.

        Args:
            weights: Portfolio weights {symbol: weight}
            covariance_matrix: Optional covariance matrix

        Returns:
            DiversificationMetric
        """
        if covariance_matrix is None:
            # Calculate from returns
            returns_dict = {}
            for symbol in weights.keys():
                returns = self.calculate_returns(symbol)
                if len(returns) > 0:
                    returns_dict[symbol] = returns

            min_length = min(len(r) for r in returns_dict.values())
            for symbol in returns_dict:
                returns_dict[symbol] = returns_dict[symbol][-min_length:]

            df = pd.DataFrame(returns_dict)
            covariance_matrix = df.cov().values * 252  # Annualize

        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Portfolio variance
        portfolio_variance = w @ covariance_matrix @ w

        # Weighted individual variance
        individual_vars = np.diag(covariance_matrix)
        weighted_individual_variance = np.sum(w**2 * individual_vars)

        # Correlation benefit
        correlation_benefit = weighted_individual_variance - portfolio_variance

        # Diversification ratio
        individual_volatilities = np.sqrt(individual_vars)
        numerator = np.sum(w * individual_volatilities)
        denominator = np.sqrt(portfolio_variance)

        diversification_ratio = (
            numerator / denominator if denominator > 0 else 1.0
        )

        # Effective number of uncorrelated assets
        # Herfindahl index approach
        herfindahl = np.sum(w**2)
        effective_n = 1 / herfindahl if herfindahl > 0 else len(symbols)

        return DiversificationMetric(
            portfolio_variance=portfolio_variance,
            weighted_individual_variance=weighted_individual_variance,
            correlation_benefit=correlation_benefit,
            diversification_ratio=diversification_ratio,
            effective_n_assets=effective_n,
        )

    def get_correlation_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Get asset pairs with correlation above threshold.

        Args:
            threshold: Correlation threshold

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()

        pairs = []
        symbols = self.correlation_matrix.index.tolist()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    pairs.append((symbols[i], symbols[j], corr))

        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def calculate_systemic_risk(self, weights: Dict[str, float]) -> float:
        """
        Calculate systemic risk (beta-weighted concentration).

        Args:
            weights: Portfolio weights

        Returns:
            Systemic risk score (0-1)
        """
        if not self.correlation_matrix:
            self.calculate_correlation_matrix()

        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Get betas from correlation with market proxy (average of all assets)
        betas = []
        for symbol in symbols:
            returns = self.calculate_returns(symbol)
            avg_returns = np.mean(
                [self.calculate_returns(s) for s in symbols if s in self.price_history],
                axis=0,
            )

            if len(returns) > 0 and len(avg_returns) > 0:
                beta = np.corrcoef(returns[-min(len(returns), len(avg_returns)) :],
                                   avg_returns[-min(len(returns), len(avg_returns)) :])
                betas.append(abs(beta[0, 1]))
            else:
                betas.append(0)

        # Systemic risk = sum of weighted betas
        systemic_risk = np.sum(w * np.array(betas))

        return min(systemic_risk, 1.0)  # Normalize to [0, 1]
