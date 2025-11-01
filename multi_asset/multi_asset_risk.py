"""
Multi-Asset Risk Module

Implements cross-asset risk management:
- Systemic risk detection
- Concentration risk monitoring
- Stress testing across assets
- Risk limit enforcement
- Correlation risk metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemicRiskMetric:
    """Systemic risk metric."""
    timestamp: str
    systemic_risk_score: float  # 0-1, higher = more risk
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    market_stress_indicator: float
    asset_class_exposures: Dict[str, float]


@dataclass
class ConcentrationRiskMetric:
    """Concentration risk details."""
    largest_position_weight: float
    top_5_weight: float
    top_10_weight: float
    herfindahl_index: float
    effective_n_assets: float
    risk_level: str  # 'low', 'medium', 'high'


@dataclass
class StressTestResult:
    """Stress test result."""
    scenario_name: str
    portfolio_loss_pct: float
    worst_asset: str
    worst_asset_loss: float
    var_95: float
    cvar_95: float
    recovery_time_days: int


@dataclass
class RiskLimitViolation:
    """Risk limit violation."""
    limit_name: str
    limit_value: float
    current_value: float
    violation_severity: str  # 'warning', 'critical'
    timestamp: str
    asset_class: Optional[str] = None


class MultiAssetRiskManager:
    """
    Manages risk across multiple asset classes.
    """

    def __init__(self):
        """Initialize multi-asset risk manager."""
        self.risk_limits: Dict[str, float] = {
            "max_concentration": 0.20,  # Max 20% in single position
            "max_sector_concentration": 0.30,  # Max 30% in sector
            "max_drawdown": 0.15,  # Max 15% drawdown
            "max_volatility": 0.25,  # Max 25% volatility
            "min_diversification_score": 0.5,  # Min diversification
            "max_systemic_risk": 0.70,  # Max systemic risk score
        }
        self.violations: List[RiskLimitViolation] = []
        self.stress_test_results: List[StressTestResult] = []

        logger.info("MultiAssetRiskManager initialized")

    def set_risk_limits(self, limits: Dict[str, float]) -> None:
        """
        Set risk limits.

        Args:
            limits: Dictionary of limit names and values
        """
        self.risk_limits.update(limits)
        logger.info(f"Risk limits updated: {limits}")

    def assess_concentration_risk(
        self, holdings: Dict[str, Dict]
    ) -> ConcentrationRiskMetric:
        """
        Assess concentration risk in portfolio.

        Args:
            holdings: {symbol: {value, asset_class, ...}}

        Returns:
            ConcentrationRiskMetric
        """
        total_value = sum(h["value"] for h in holdings.values())
        weights = [h["value"] / total_value for h in holdings.values()]
        weights = np.array(sorted(weights, reverse=True))

        # Largest position
        largest_position_weight = weights[0] if len(weights) > 0 else 0

        # Top 5 and Top 10
        top_5_weight = np.sum(weights[:5]) if len(weights) >= 5 else np.sum(weights)
        top_10_weight = np.sum(weights[:10]) if len(weights) >= 10 else np.sum(weights)

        # Herfindahl index
        herfindahl = np.sum(weights**2)

        # Effective number of assets
        effective_n = 1 / herfindahl if herfindahl > 0 else len(holdings)

        # Determine risk level
        if largest_position_weight > 0.30 or herfindahl > 0.25:
            risk_level = "high"
        elif largest_position_weight > 0.15 or herfindahl > 0.15:
            risk_level = "medium"
        else:
            risk_level = "low"

        metric = ConcentrationRiskMetric(
            largest_position_weight=largest_position_weight,
            top_5_weight=top_5_weight,
            top_10_weight=top_10_weight,
            herfindahl_index=herfindahl,
            effective_n_assets=effective_n,
            risk_level=risk_level,
        )

        logger.info(f"Concentration risk: {risk_level} - Largest: {largest_position_weight:.1%}")

        return metric

    def detect_systemic_risk(
        self,
        holdings: Dict[str, Dict],
        correlations: Dict[Tuple[str, str], float],
        market_stress: float = 0.0,
    ) -> SystemicRiskMetric:
        """
        Detect systemic risk across portfolio.

        Args:
            holdings: {symbol: {value, asset_class, ...}}
            correlations: {(symbol1, symbol2): correlation}
            market_stress: Market stress level (0-1)

        Returns:
            SystemicRiskMetric
        """
        total_value = sum(h["value"] for h in holdings.values())
        weights = {s: h["value"] / total_value for s, h in holdings.items()}

        # Asset class exposures
        asset_class_exposures = {}
        for symbol, holding in holdings.items():
            ac = holding.get("asset_class", "unknown")
            if ac not in asset_class_exposures:
                asset_class_exposures[ac] = 0
            asset_class_exposures[ac] += weights[symbol]

        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(correlations, weights)

        # Concentration risk
        concentration_metric = self.assess_concentration_risk(holdings)
        concentration_risk = min(1.0, concentration_metric.largest_position_weight * 2)

        # Liquidity risk (simplified - based on asset class)
        liquidity_risk = sum(
            weights.get(s, 0) * (0.3 if holdings[s].get("asset_class") == "crypto" else 0.1)
            for s in holdings.keys()
        )

        # Systemic risk score
        systemic_risk_score = (
            0.3 * correlation_risk
            + 0.3 * concentration_risk
            + 0.2 * liquidity_risk
            + 0.2 * market_stress
        )

        return SystemicRiskMetric(
            timestamp=datetime.now().isoformat(),
            systemic_risk_score=min(1.0, systemic_risk_score),
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            market_stress_indicator=market_stress,
            asset_class_exposures=asset_class_exposures,
        )

    def _calculate_correlation_risk(
        self, correlations: Dict[Tuple[str, str], float], weights: Dict[str, float]
    ) -> float:
        """Calculate correlation risk."""
        if not correlations:
            return 0

        # Weight correlations by position sizes
        weighted_corrs = []
        for (s1, s2), corr in correlations.items():
            if s1 in weights and s2 in weights:
                weight_product = weights[s1] * weights[s2]
                weighted_corrs.append(weight_product * corr)

        # Risk increases with high average correlation
        avg_weighted_corr = np.mean(weighted_corrs) if weighted_corrs else 0
        correlation_risk = max(0, avg_weighted_corr)

        return correlation_risk

    def check_risk_limits(
        self,
        portfolio_metrics: Dict,
        concentration_metric: ConcentrationRiskMetric,
        systemic_metric: SystemicRiskMetric,
    ) -> List[RiskLimitViolation]:
        """
        Check all risk limits.

        Args:
            portfolio_metrics: Portfolio metrics dictionary
            concentration_metric: Concentration risk metric
            systemic_metric: Systemic risk metric

        Returns:
            List of violations
        """
        violations = []

        # Check concentration limits
        if concentration_metric.largest_position_weight > self.risk_limits["max_concentration"]:
            violations.append(
                RiskLimitViolation(
                    limit_name="max_concentration",
                    limit_value=self.risk_limits["max_concentration"],
                    current_value=concentration_metric.largest_position_weight,
                    violation_severity="warning",
                    timestamp=datetime.now().isoformat(),
                )
            )

        # Check diversification limit
        if "diversification_score" in portfolio_metrics:
            if (portfolio_metrics["diversification_score"] <
                self.risk_limits["min_diversification_score"]):
                violations.append(
                    RiskLimitViolation(
                        limit_name="min_diversification_score",
                        limit_value=self.risk_limits["min_diversification_score"],
                        current_value=portfolio_metrics["diversification_score"],
                        violation_severity="warning",
                        timestamp=datetime.now().isoformat(),
                    )
                )

        # Check drawdown limit
        if "max_drawdown" in portfolio_metrics:
            drawdown = abs(portfolio_metrics["max_drawdown"])
            if drawdown > self.risk_limits["max_drawdown"]:
                violations.append(
                    RiskLimitViolation(
                        limit_name="max_drawdown",
                        limit_value=self.risk_limits["max_drawdown"],
                        current_value=drawdown,
                        violation_severity="critical",
                        timestamp=datetime.now().isoformat(),
                    )
                )

        # Check volatility limit
        if "volatility" in portfolio_metrics:
            if portfolio_metrics["volatility"] > self.risk_limits["max_volatility"]:
                violations.append(
                    RiskLimitViolation(
                        limit_name="max_volatility",
                        limit_value=self.risk_limits["max_volatility"],
                        current_value=portfolio_metrics["volatility"],
                        violation_severity="warning",
                        timestamp=datetime.now().isoformat(),
                    )
                )

        # Check systemic risk limit
        if systemic_metric.systemic_risk_score > self.risk_limits["max_systemic_risk"]:
            violations.append(
                RiskLimitViolation(
                    limit_name="max_systemic_risk",
                    limit_value=self.risk_limits["max_systemic_risk"],
                    current_value=systemic_metric.systemic_risk_score,
                    violation_severity="critical",
                    timestamp=datetime.now().isoformat(),
                )
            )

        self.violations.extend(violations)

        if violations:
            logger.warning(f"Risk limit violations detected: {len(violations)}")

        return violations

    def stress_test_portfolio(
        self,
        holdings: Dict[str, Dict],
        scenarios: Dict[str, Dict[str, float]],
    ) -> List[StressTestResult]:
        """
        Stress test portfolio across scenarios.

        Args:
            holdings: {symbol: {value, ...}}
            scenarios: {scenario_name: {symbol: return_pct}}

        Returns:
            List of StressTestResult
        """
        results = []
        total_value = sum(h["value"] for h in holdings.values())

        for scenario_name, returns in scenarios.items():
            portfolio_loss = 0
            worst_asset = ""
            worst_loss = 0

            for symbol, holding in holdings.items():
                symbol_return = returns.get(symbol, 0)
                position_loss = holding["value"] * symbol_return

                if position_loss < worst_loss:
                    worst_loss = position_loss
                    worst_asset = symbol

                portfolio_loss += position_loss

            portfolio_loss_pct = portfolio_loss / total_value if total_value > 0 else 0
            worst_asset_loss = worst_loss / holdings.get(worst_asset, {}).get("value", 1)

            # Estimate VaR and CVaR (simplified)
            var_95 = abs(portfolio_loss) * 1.645 / np.sqrt(252)
            cvar_95 = var_95 * 1.5

            # Estimate recovery time
            if portfolio_loss_pct < 0:
                recovery_time = int(abs(portfolio_loss_pct) * 365)
            else:
                recovery_time = 0

            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss_pct=portfolio_loss_pct,
                worst_asset=worst_asset,
                worst_asset_loss=worst_asset_loss,
                var_95=var_95,
                cvar_95=cvar_95,
                recovery_time_days=recovery_time,
            )

            results.append(result)

        self.stress_test_results = results
        logger.info(f"Stress tested {len(scenarios)} scenarios")

        return results

    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Portfolio return series
            confidence: Confidence level (0-1)

        Returns:
            VaR at specified confidence level
        """
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        return var

    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk.

        Args:
            returns: Portfolio return series
            confidence: Confidence level (0-1)

        Returns:
            CVaR at specified confidence level
        """
        returns_array = np.array(returns)
        var = self.calculate_var(returns, confidence)
        cvar = returns_array[returns_array <= var].mean()
        return cvar

    def get_risk_dashboard(
        self,
        portfolio_metrics: Dict,
        concentration_metric: ConcentrationRiskMetric,
        systemic_metric: SystemicRiskMetric,
    ) -> Dict:
        """Get comprehensive risk dashboard."""
        violations = self.check_risk_limits(
            portfolio_metrics, concentration_metric, systemic_metric
        )

        return {
            "concentration": {
                "largest_position": concentration_metric.largest_position_weight,
                "top_5_weight": concentration_metric.top_5_weight,
                "herfindahl_index": concentration_metric.herfindahl_index,
                "risk_level": concentration_metric.risk_level,
                "effective_n_assets": concentration_metric.effective_n_assets,
            },
            "systemic": {
                "systemic_risk_score": systemic_metric.systemic_risk_score,
                "correlation_risk": systemic_metric.correlation_risk,
                "concentration_risk": systemic_metric.concentration_risk,
                "liquidity_risk": systemic_metric.liquidity_risk,
                "market_stress": systemic_metric.market_stress_indicator,
            },
            "portfolio": {
                "volatility": portfolio_metrics.get("volatility", 0),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0),
                "max_drawdown": portfolio_metrics.get("max_drawdown", 0),
                "diversification_score": portfolio_metrics.get("diversification_score", 0),
            },
            "violations": [
                {
                    "limit": v.limit_name,
                    "limit_value": v.limit_value,
                    "current_value": v.current_value,
                    "severity": v.violation_severity,
                }
                for v in violations
            ],
            "violation_count": len(violations),
            "critical_violations": len([v for v in violations if v.violation_severity == "critical"]),
        }
