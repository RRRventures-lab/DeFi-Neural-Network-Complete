"""
Scenario Analyzer Module

Implements advanced scenario analysis:
- Monte Carlo simulation
- Historical scenarios
- Stress test scenarios
- Custom scenario builder
- Confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of scenario analysis."""
    scenario_name: str
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional VaR (expected shortfall)
    probability: float = 1.0
    num_simulations: int = 1000


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    mean_return: float
    std_return: float
    paths: np.ndarray  # Shape: (num_steps, num_paths, num_assets)
    confidence_intervals: Dict[float, Tuple[float, float]]  # {percentile: (lower, upper)}
    var_metrics: Dict[str, float]  # VaR at different confidence levels
    cvar_metrics: Dict[str, float]  # CVaR at different confidence levels

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mean_return': self.mean_return,
            'std_return': self.std_return,
            'confidence_intervals': self.confidence_intervals,
            'var_95': self.var_metrics.get('var_95', 0),
            'cvar_95': self.cvar_metrics.get('cvar_95', 0)
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio paths."""

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_paths(
        self,
        initial_prices: np.ndarray,
        returns_mean: np.ndarray,
        returns_cov: np.ndarray,
        num_steps: int = 252,
        num_paths: int = 1000,
        time_step: float = 1/252
    ) -> MonteCarloResults:
        """
        Simulate price paths using geometric Brownian motion.

        Args:
            initial_prices: Initial prices (num_assets,)
            returns_mean: Mean returns (num_assets,)
            returns_cov: Covariance matrix (num_assets, num_assets)
            num_steps: Number of time steps
            num_paths: Number of simulation paths
            time_step: Time step size (default: 1/252 for daily)

        Returns:
            MonteCarloResults
        """
        num_assets = len(initial_prices)
        paths = np.zeros((num_steps + 1, num_paths, num_assets))
        paths[0] = initial_prices

        # Cholesky decomposition for correlated returns
        L = np.linalg.cholesky(returns_cov)

        for t in range(1, num_steps + 1):
            # Generate correlated random normal returns
            Z = np.random.standard_normal((num_assets, num_paths))
            dW = L @ Z

            # GBM: dS = mu*S*dt + sigma*S*dW
            drift = returns_mean[:, np.newaxis] * time_step
            diffusion = (dW * np.sqrt(time_step)).T
            returns = drift.T + diffusion

            paths[t] = paths[t - 1] * np.exp(returns)

        # Calculate portfolio returns (assuming equal weight)
        portfolio_values = np.mean(paths, axis=2)
        portfolio_returns = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Calculate metrics
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        # Confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        confidence_intervals = {}
        for p in percentiles:
            lower = np.percentile(portfolio_returns, p)
            upper = np.percentile(portfolio_returns, 100 - p)
            confidence_intervals[p / 100] = (lower, upper)

        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

        var_metrics = {'var_95': var_95, 'var_99': np.percentile(portfolio_returns, 1)}
        cvar_metrics = {'cvar_95': cvar_95, 'cvar_99': np.mean(portfolio_returns[portfolio_returns <= var_metrics['var_99']])}

        return MonteCarloResults(
            mean_return=mean_return,
            std_return=std_return,
            paths=paths,
            confidence_intervals=confidence_intervals,
            var_metrics=var_metrics,
            cvar_metrics=cvar_metrics
        )


class ScenarioAnalyzer:
    """
    Advanced scenario analysis system.
    """

    def __init__(self):
        """Initialize scenario analyzer."""
        self.scenarios: Dict[str, np.ndarray] = {}
        self.scenario_weights: Dict[str, float] = {}
        self.simulator = MonteCarloSimulator()

        logger.info("Scenario analyzer initialized")

    def add_scenario(
        self,
        name: str,
        returns: np.ndarray,
        probability: float = 1.0
    ) -> None:
        """
        Add a custom scenario.

        Args:
            name: Scenario name
            returns: Returns array for scenario
            probability: Probability weight
        """
        self.scenarios[name] = returns
        self.scenario_weights[name] = probability
        logger.info(f"Added scenario: {name}")

    def add_monte_carlo_scenario(
        self,
        name: str,
        initial_prices: np.ndarray,
        returns_mean: np.ndarray,
        returns_cov: np.ndarray,
        num_steps: int = 252,
        num_paths: int = 1000
    ) -> MonteCarloResults:
        """
        Add Monte Carlo scenario.

        Args:
            name: Scenario name
            initial_prices: Initial prices
            returns_mean: Mean returns
            returns_cov: Covariance matrix
            num_steps: Number of steps
            num_paths: Number of paths

        Returns:
            MonteCarloResults
        """
        results = self.simulator.simulate_paths(
            initial_prices,
            returns_mean,
            returns_cov,
            num_steps,
            num_paths
        )

        # Store average path as scenario
        avg_path_returns = np.mean(results.paths, axis=1)
        portfolio_returns = (avg_path_returns[-1] - avg_path_returns[0]) / avg_path_returns[0]
        self.scenarios[name] = np.array([portfolio_returns])
        self.scenario_weights[name] = 1.0

        logger.info(f"Added Monte Carlo scenario: {name}")

        return results

    def add_historical_scenario(
        self,
        name: str,
        historical_returns: np.ndarray,
        scaling_factor: float = 1.0
    ) -> None:
        """
        Add historical scenario (replay historical returns).

        Args:
            name: Scenario name
            historical_returns: Historical returns to replay
            scaling_factor: Scale factor for returns
        """
        scaled_returns = historical_returns * scaling_factor
        self.scenarios[name] = scaled_returns
        self.scenario_weights[name] = 1.0

        logger.info(f"Added historical scenario: {name} (scale: {scaling_factor})")

    def add_stress_scenario(
        self,
        name: str,
        market_shock: float,
        volatility_spike: float = 1.5
    ) -> None:
        """
        Add stress test scenario.

        Args:
            name: Scenario name
            market_shock: Market return shock (e.g., -0.20 for -20%)
            volatility_spike: Volatility multiplier
        """
        # Simple stress scenario
        shock_returns = np.array([market_shock])
        self.scenarios[name] = shock_returns
        self.scenario_weights[name] = 1.0

        logger.info(f"Added stress scenario: {name} (shock: {market_shock:.1%}, vol spike: {volatility_spike}x)")

    def analyze_scenario(
        self,
        scenario_name: str,
        weights: np.ndarray
    ) -> ScenarioResult:
        """
        Analyze portfolio performance in scenario.

        Args:
            scenario_name: Scenario name
            weights: Portfolio weights

        Returns:
            ScenarioResult
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        returns = self.scenarios[scenario_name]

        # Calculate weighted portfolio returns
        if returns.ndim == 1:
            portfolio_returns = returns @ weights if len(weights) == len(returns) else np.mean(returns)
        else:
            portfolio_returns = returns @ weights

        # Convert to array if scalar
        if np.isscalar(portfolio_returns):
            portfolio_returns = np.array([portfolio_returns])

        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95]) if len(portfolio_returns) > 0 else var_95

        return ScenarioResult(
            scenario_name=scenario_name,
            mean_return=mean_return,
            std_return=std_return,
            min_return=np.min(portfolio_returns),
            max_return=np.max(portfolio_returns),
            var_95=var_95,
            cvar_95=cvar_95,
            probability=self.scenario_weights.get(scenario_name, 1.0),
            num_simulations=len(portfolio_returns)
        )

    def analyze_all_scenarios(
        self,
        weights: np.ndarray
    ) -> Dict[str, ScenarioResult]:
        """
        Analyze portfolio across all scenarios.

        Args:
            weights: Portfolio weights

        Returns:
            Dict of scenario results
        """
        results = {}

        for scenario_name in self.scenarios.keys():
            results[scenario_name] = self.analyze_scenario(scenario_name, weights)

        return results

    def calculate_expected_return(
        self,
        weights: np.ndarray
    ) -> float:
        """
        Calculate probability-weighted expected return across scenarios.

        Args:
            weights: Portfolio weights

        Returns:
            Expected return
        """
        total_weight = sum(self.scenario_weights.values())
        expected_return = 0

        for scenario_name, weight in self.scenario_weights.items():
            result = self.analyze_scenario(scenario_name, weights)
            expected_return += (result.mean_return * weight / total_weight)

        return expected_return

    def calculate_scenario_probability(
        self,
        scenario_name: str
    ) -> float:
        """
        Get scenario probability weight.

        Args:
            scenario_name: Scenario name

        Returns:
            Probability weight (0-1)
        """
        if scenario_name not in self.scenario_weights:
            return 0

        total_weight = sum(self.scenario_weights.values())
        return self.scenario_weights[scenario_name] / total_weight

    def get_scenario_summary(
        self,
        weights: np.ndarray
    ) -> pd.DataFrame:
        """
        Get summary of all scenarios.

        Args:
            weights: Portfolio weights

        Returns:
            DataFrame with scenario metrics
        """
        results = self.analyze_all_scenarios(weights)

        data = []
        for scenario_name, result in results.items():
            data.append({
                'Scenario': scenario_name,
                'Mean Return': result.mean_return,
                'Std Dev': result.std_return,
                'Min': result.min_return,
                'Max': result.max_return,
                'VaR(95%)': result.var_95,
                'CVaR(95%)': result.cvar_95,
                'Probability': result.probability
            })

        return pd.DataFrame(data)

    def get_distribution_stats(
        self,
        scenario_name: str
    ) -> Dict:
        """
        Get distribution statistics for a scenario.

        Args:
            scenario_name: Scenario name

        Returns:
            Dictionary with distribution stats
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        returns = self.scenarios[scenario_name]

        # Convert to 1D if needed
        if returns.ndim > 1:
            returns = returns.flatten()

        return {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns),
            'skew': pd.Series(returns).skew(),
            'kurtosis': pd.Series(returns).kurtosis(),
            'count': len(returns)
        }
