"""
Risk Limits Module

Implements risk limit enforcement for portfolio management:
- Maximum drawdown limits
- Position concentration limits
- Volatility limits
- Value at Risk (VaR) limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskViolation:
    """Container for risk limit violations."""
    limit_name: str
    current_value: float
    limit_value: float
    exceeded_by: float
    severity: str  # 'warning', 'critical'
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'limit_name': self.limit_name,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'exceeded_by': self.exceeded_by,
            'severity': self.severity,
            'timestamp': self.timestamp
        }


class RiskLimits:
    """
    Comprehensive risk limit enforcement system.
    """

    def __init__(self):
        """Initialize risk limits container."""
        self.limits: Dict[str, float] = {}
        self.violations: List[RiskViolation] = []
        self.thresholds = {
            'warning': 0.9,  # 90% of limit
            'critical': 0.95  # 95% of limit
        }

    def set_limit(self, name: str, value: float) -> None:
        """
        Set a risk limit.

        Args:
            name: Limit name
            value: Limit value
        """
        self.limits[name] = value
        logger.info(f"Set risk limit {name} = {value}")

    def check_limit(
        self,
        name: str,
        current_value: float,
        asset: Optional[str] = None
    ) -> Optional[RiskViolation]:
        """
        Check if current value violates limit.

        Args:
            name: Limit name
            current_value: Current value
            asset: Optional asset identifier

        Returns:
            RiskViolation if exceeded, None otherwise
        """
        if name not in self.limits:
            return None

        limit = self.limits[name]
        exceeded_by = current_value - limit

        if exceeded_by > 0:
            # Determine severity
            ratio = current_value / limit
            severity = 'critical' if ratio > self.thresholds['critical'] else 'warning'

            violation = RiskViolation(
                limit_name=name,
                current_value=current_value,
                limit_value=limit,
                exceeded_by=exceeded_by,
                severity=severity
            )

            self.violations.append(violation)
            logger.warning(f"Risk limit violation: {name} - {severity}")

            return violation

        return None

    def get_violations(self) -> List[RiskViolation]:
        """Get all recorded violations."""
        return self.violations

    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violations = []


class DrawdownLimit:
    """
    Maximum drawdown limit enforcement.
    """

    def __init__(self, max_drawdown: float = -0.25):
        """
        Initialize drawdown limit.

        Args:
            max_drawdown: Maximum allowed drawdown (-0.25 = -25%)
        """
        self.max_drawdown = max_drawdown
        self.risk_limits = RiskLimits()
        self.risk_limits.set_limit('max_drawdown', abs(max_drawdown))

    def calculate_drawdown(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown from returns.

        Args:
            returns: Daily returns

        Returns:
            Drawdown at each timestep
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    def check_drawdown(self, returns: np.ndarray) -> Optional[RiskViolation]:
        """
        Check if current drawdown violates limit.

        Args:
            returns: Daily returns

        Returns:
            RiskViolation if exceeded, None otherwise
        """
        drawdown = self.calculate_drawdown(returns)
        current_dd = np.min(drawdown)

        return self.risk_limits.check_limit(
            'max_drawdown',
            abs(current_dd)
        )

    def remaining_drawdown_budget(self, returns: np.ndarray) -> float:
        """
        Calculate remaining drawdown budget.

        Args:
            returns: Daily returns

        Returns:
            Remaining drawdown budget (0-1)
        """
        drawdown = self.calculate_drawdown(returns)
        current_dd = np.min(drawdown)
        remaining = abs(self.max_drawdown) - abs(current_dd)
        return max(0, remaining)


class ConcentrationLimit:
    """
    Position concentration limit enforcement.
    """

    def __init__(self, max_concentration: float = 0.15):
        """
        Initialize concentration limit.

        Args:
            max_concentration: Maximum weight per position (0.15 = 15%)
        """
        self.max_concentration = max_concentration
        self.risk_limits = RiskLimits()
        self.risk_limits.set_limit('position_concentration', max_concentration)

    def check_allocation(
        self,
        weights: np.ndarray
    ) -> List[RiskViolation]:
        """
        Check if any position exceeds concentration limit.

        Args:
            weights: Portfolio weights

        Returns:
            List of RiskViolations for exceeded positions
        """
        violations = []

        for i, weight in enumerate(weights):
            if weight > self.max_concentration:
                violation = self.risk_limits.check_limit(
                    'position_concentration',
                    weight,
                    asset=f'asset_{i}'
                )
                if violation:
                    violations.append(violation)

        return violations

    def get_top_positions(
        self,
        weights: np.ndarray,
        num_top: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top N positions by weight.

        Args:
            weights: Portfolio weights
            num_top: Number of top positions

        Returns:
            List of (asset_index, weight) tuples
        """
        indices = np.argsort(-weights)[:num_top]
        return [(idx, weights[idx]) for idx in indices]


class VolatilityLimit:
    """
    Portfolio volatility limit enforcement.
    """

    def __init__(self, max_volatility: float = 0.20):
        """
        Initialize volatility limit.

        Args:
            max_volatility: Maximum annual volatility (0.20 = 20%)
        """
        self.max_volatility = max_volatility
        self.risk_limits = RiskLimits()
        self.risk_limits.set_limit('volatility', max_volatility)

    def calculate_volatility(
        self,
        returns: np.ndarray
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Daily returns

        Returns:
            Annualized volatility
        """
        return np.std(returns) * np.sqrt(252)

    def check_volatility(
        self,
        returns: np.ndarray
    ) -> Optional[RiskViolation]:
        """
        Check if portfolio volatility violates limit.

        Args:
            returns: Daily returns

        Returns:
            RiskViolation if exceeded, None otherwise
        """
        volatility = self.calculate_volatility(returns)

        return self.risk_limits.check_limit(
            'volatility',
            volatility
        )

    def get_volatility_budget(self, returns: np.ndarray) -> float:
        """
        Calculate remaining volatility budget.

        Args:
            returns: Daily returns

        Returns:
            Remaining volatility budget (0-1)
        """
        current_vol = self.calculate_volatility(returns)
        remaining = self.max_volatility - current_vol
        return max(0, remaining)


class VaRLimit:
    """
    Value at Risk (VaR) limit enforcement.
    """

    def __init__(
        self,
        max_var: float = -0.05,
        confidence_level: float = 0.95
    ):
        """
        Initialize VaR limit.

        Args:
            max_var: Maximum allowed VaR (-0.05 = -5% worst case)
            confidence_level: Confidence level for VaR calculation (0.95 = 95%)
        """
        self.max_var = max_var
        self.confidence_level = confidence_level
        self.percentile = (1 - confidence_level) * 100
        self.risk_limits = RiskLimits()
        self.risk_limits.set_limit('var', abs(max_var))

    def calculate_var(self, returns: np.ndarray) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Daily returns

        Returns:
            VaR (negative value representing worst-case loss)
        """
        var = np.percentile(returns, self.percentile)
        return var

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Daily returns

        Returns:
            CVaR (expected loss in worst case scenarios)
        """
        var = self.calculate_var(returns)
        cvar = np.mean(returns[returns <= var])
        return cvar

    def check_var(
        self,
        returns: np.ndarray
    ) -> Optional[RiskViolation]:
        """
        Check if VaR violates limit.

        Args:
            returns: Daily returns

        Returns:
            RiskViolation if exceeded, None otherwise
        """
        var = self.calculate_var(returns)

        return self.risk_limits.check_limit(
            'var',
            abs(var)
        )

    def get_risk_summary(self, returns: np.ndarray) -> Dict:
        """
        Get comprehensive risk summary.

        Args:
            returns: Daily returns

        Returns:
            Dictionary with VaR metrics
        """
        var = self.calculate_var(returns)
        cvar = self.calculate_cvar(returns)

        return {
            'var': var,
            'cvar': cvar,
            'var_limit': self.max_var,
            'confidence_level': self.confidence_level,
            'var_exceeded': var < self.max_var
        }
