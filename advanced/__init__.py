"""
Advanced Features Module

Tax optimization, scenario analysis, options pricing,
multi-period optimization, and custom constraints.
"""

from .tax_optimizer import (
    TaxOptimizer,
    TaxLot,
    TaxHarvest,
    TaxOptimizationResult
)

from .scenario_analyzer import (
    ScenarioAnalyzer,
    MonteCarloSimulator,
    MonteCarloResults,
    ScenarioResult
)

from .options_pricer import (
    OptionsPricer,
    BlackScholesCalculator,
    OptionPrice
)

from .multi_period_optimizer import (
    MultiPeriodOptimizer,
    MultiPeriodAllocation
)

from .custom_constraints import (
    ConstraintBuilder,
    Constraint,
    CustomObjective
)

__all__ = [
    # Tax Optimizer
    'TaxOptimizer',
    'TaxLot',
    'TaxHarvest',
    'TaxOptimizationResult',

    # Scenario Analyzer
    'ScenarioAnalyzer',
    'MonteCarloSimulator',
    'MonteCarloResults',
    'ScenarioResult',

    # Options Pricer
    'OptionsPricer',
    'BlackScholesCalculator',
    'OptionPrice',

    # Multi-Period Optimizer
    'MultiPeriodOptimizer',
    'MultiPeriodAllocation',

    # Custom Constraints
    'ConstraintBuilder',
    'Constraint',
    'CustomObjective'
]
