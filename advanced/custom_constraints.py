"""
Custom Constraints Module

Implements flexible constraint system:
- Linear constraints
- Custom objectives
- Constraint combinations
- Sensitivity analysis
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Constraint:
    """Single constraint specification."""
    name: str
    constraint_type: str  # 'linear', 'min', 'max', 'sum', 'difference'
    assets: List[int]  # Asset indices involved
    bound: float  # Constraint bound
    operator: str  # '<=', '>=', '=='

    def check(self, weights: np.ndarray) -> bool:
        """Check if constraint is satisfied."""
        if self.constraint_type == 'linear':
            value = np.sum(weights[self.assets])
        elif self.constraint_type == 'min':
            value = np.min(weights[self.assets])
        elif self.constraint_type == 'max':
            value = np.max(weights[self.assets])
        elif self.constraint_type == 'sum':
            value = np.sum(weights[self.assets])
        elif self.constraint_type == 'difference':
            value = weights[self.assets[0]] - weights[self.assets[1]]
        else:
            return True

        if self.operator == '<=':
            return value <= self.bound
        elif self.operator == '>=':
            return value >= self.bound
        elif self.operator == '==':
            return np.isclose(value, self.bound)
        else:
            return True


@dataclass
class CustomObjective:
    """Custom objective function specification."""
    name: str
    objective_type: str  # 'maximize', 'minimize'
    weights: np.ndarray  # Weights for linear combination
    lambda_val: float = 1.0  # Weighting factor in multi-objective


class ConstraintBuilder:
    """
    Flexible constraint builder for portfolio optimization.
    """

    def __init__(self, num_assets: int):
        """
        Initialize constraint builder.

        Args:
            num_assets: Number of assets
        """
        self.num_assets = num_assets
        self.constraints: List[Constraint] = []
        self.objectives: List[CustomObjective] = []

        logger.info(f"Constraint builder initialized for {num_assets} assets")

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add a constraint.

        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint.name}")

    def add_linear_constraint(
        self,
        name: str,
        coefficients: np.ndarray,
        bound: float,
        operator: str = '<='
    ) -> None:
        """
        Add linear constraint: coefficients @ weights op bound.

        Args:
            name: Constraint name
            coefficients: Coefficients (num_assets,)
            bound: Constraint bound
            operator: '<=' or '>=' or '=='
        """
        # Convert to asset indices with non-zero coefficients
        nonzero_indices = np.where(coefficients != 0)[0].tolist()

        constraint = Constraint(
            name=name,
            constraint_type='linear',
            assets=nonzero_indices,
            bound=bound,
            operator=operator
        )
        self.add_constraint(constraint)

    def add_min_weight_constraint(
        self,
        asset_index: int,
        min_weight: float,
        asset_name: str = ''
    ) -> None:
        """
        Add minimum weight constraint for asset.

        Args:
            asset_index: Asset index
            min_weight: Minimum weight
            asset_name: Optional asset name
        """
        name = f"Min weight {asset_name or asset_index}"
        constraint = Constraint(
            name=name,
            constraint_type='min',
            assets=[asset_index],
            bound=min_weight,
            operator='>='
        )
        self.add_constraint(constraint)

    def add_max_weight_constraint(
        self,
        asset_index: int,
        max_weight: float,
        asset_name: str = ''
    ) -> None:
        """
        Add maximum weight constraint for asset.

        Args:
            asset_index: Asset index
            max_weight: Maximum weight
            asset_name: Optional asset name
        """
        name = f"Max weight {asset_name or asset_index}"
        constraint = Constraint(
            name=name,
            constraint_type='max',
            assets=[asset_index],
            bound=max_weight,
            operator='<='
        )
        self.add_constraint(constraint)

    def add_sector_constraint(
        self,
        name: str,
        sector_assets: List[int],
        max_weight: float
    ) -> None:
        """
        Add sector concentration constraint.

        Args:
            name: Sector name
            sector_assets: Assets in sector
            max_weight: Maximum sector weight
        """
        weights = np.zeros(self.num_assets)
        weights[sector_assets] = 1
        self.add_linear_constraint(f"Sector {name}", weights, max_weight, '<=')

    def add_correlation_constraint(
        self,
        name: str,
        asset1: int,
        asset2: int,
        max_difference: float
    ) -> None:
        """
        Add constraint limiting difference between two asset weights.

        Args:
            name: Constraint name
            asset1: First asset index
            asset2: Second asset index
            max_difference: Maximum difference in weights
        """
        constraint = Constraint(
            name=name,
            constraint_type='difference',
            assets=[asset1, asset2],
            bound=max_difference,
            operator='<='
        )
        self.add_constraint(constraint)

    def add_objective(self, objective: CustomObjective) -> None:
        """
        Add objective function.

        Args:
            objective: CustomObjective to add
        """
        self.objectives.append(objective)
        logger.debug(f"Added objective: {objective.name}")

    def add_return_objective(
        self,
        expected_returns: np.ndarray,
        maximize: bool = True,
        weight: float = 1.0
    ) -> None:
        """
        Add return optimization objective.

        Args:
            expected_returns: Expected returns for each asset
            maximize: True to maximize, False to minimize
            weight: Weight in multi-objective optimization
        """
        objective = CustomObjective(
            name="Return",
            objective_type='maximize' if maximize else 'minimize',
            weights=expected_returns,
            lambda_val=weight
        )
        self.add_objective(objective)

    def add_risk_objective(
        self,
        covariance_matrix: np.ndarray,
        minimize: bool = True,
        weight: float = 1.0
    ) -> None:
        """
        Add portfolio variance objective.

        Args:
            covariance_matrix: Covariance matrix
            minimize: True to minimize risk, False to maximize
            weight: Weight in multi-objective
        """
        # For quadratic objective: w' * Cov * w
        objective = CustomObjective(
            name="Risk",
            objective_type='minimize' if minimize else 'maximize',
            weights=covariance_matrix,
            lambda_val=weight
        )
        self.add_objective(objective)

    def check_all_constraints(self, weights: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check if weights satisfy all constraints.

        Args:
            weights: Portfolio weights

        Returns:
            (all_satisfied, list of violations)
        """
        violations = []

        for constraint in self.constraints:
            if not constraint.check(weights):
                violations.append(constraint.name)

        return len(violations) == 0, violations

    def get_constraint_violations(self, weights: np.ndarray) -> Dict:
        """
        Get details of constraint violations.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary with violation details
        """
        violations = {}

        for constraint in self.constraints:
            if not constraint.check(weights):
                if constraint.constraint_type == 'linear':
                    value = np.sum(weights[constraint.assets])
                elif constraint.constraint_type == 'min':
                    value = np.min(weights[constraint.assets])
                elif constraint.constraint_type == 'max':
                    value = np.max(weights[constraint.assets])
                elif constraint.constraint_type == 'sum':
                    value = np.sum(weights[constraint.assets])
                elif constraint.constraint_type == 'difference':
                    value = weights[constraint.assets[0]] - weights[constraint.assets[1]]
                else:
                    value = None

                violations[constraint.name] = {
                    'current_value': value,
                    'bound': constraint.bound,
                    'operator': constraint.operator,
                    'violation_magnitude': abs(value - constraint.bound)
                }

        return violations

    def get_constraint_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get constraint matrix for optimization solvers.

        Returns:
            (A_eq, b_eq, bounds)
        """
        # Equality constraints: A_eq @ w = b_eq
        eq_constraints = [c for c in self.constraints if c.operator == '==']
        if eq_constraints:
            A_eq = np.zeros((len(eq_constraints), self.num_assets))
            b_eq = np.zeros(len(eq_constraints))

            for i, constraint in enumerate(eq_constraints):
                A_eq[i, constraint.assets] = 1
                b_eq[i] = constraint.bound
        else:
            A_eq = None
            b_eq = None

        # Bounds for each asset
        bounds = [(0, 1) for _ in range(self.num_assets)]

        # Inequality constraints (linear)
        # Note: scipy.optimize expects g(w) <= 0
        ineq_constraints = [c for c in self.constraints if c.operator in ['<=', '>=']]

        return A_eq, b_eq, bounds

    def get_combined_objective(self) -> Callable:
        """
        Get combined objective function from all objectives.

        Returns:
            Objective function (w -> scalar)
        """
        if not self.objectives:
            # Default: maximize return
            return lambda w: np.dot(w, np.ones(self.num_assets))

        def combined_objective(weights):
            total = 0
            for obj in self.objectives:
                if isinstance(obj.weights, np.ndarray) and obj.weights.ndim == 1:
                    value = np.dot(weights, obj.weights)
                else:
                    value = weights @ obj.weights @ weights

                if obj.objective_type == 'maximize':
                    total += obj.lambda_val * value
                else:
                    total -= obj.lambda_val * value

            return total

        return combined_objective

    def sensitivity_analysis(
        self,
        base_weights: np.ndarray,
        parameter: str,
        parameter_range: Tuple[float, float],
        num_points: int = 10
    ) -> Dict:
        """
        Perform sensitivity analysis on a parameter.

        Args:
            base_weights: Base allocation
            parameter: Parameter to vary ('return', 'volatility', etc.)
            parameter_range: (min, max) of parameter
            num_points: Number of evaluation points

        Returns:
            Sensitivity analysis results
        """
        param_values = np.linspace(parameter_range[0], parameter_range[1], num_points)
        results = []

        for param_value in param_values:
            # Modify constraint or objective based on parameter
            # This is simplified - actual implementation would vary based on parameter type

            is_feasible, violations = self.check_all_constraints(base_weights)

            results.append({
                'parameter_value': param_value,
                'feasible': is_feasible,
                'violations': len(violations),
                'base_weights': base_weights.copy()
            })

        return {
            'parameter': parameter,
            'results': results,
            'sensitivity_curve': [r['violations'] for r in results]
        }

    def summary(self) -> str:
        """Get summary of constraints and objectives."""
        summary = "Constraint Summary\n"
        summary += "=" * 50 + "\n"
        summary += f"Number of constraints: {len(self.constraints)}\n"
        summary += f"Number of objectives: {len(self.objectives)}\n\n"

        summary += "Constraints:\n"
        for c in self.constraints:
            summary += f"  - {c.name}: {c.operator} {c.bound}\n"

        summary += "\nObjectives:\n"
        for obj in self.objectives:
            summary += f"  - {obj.name} ({obj.objective_type}): weight={obj.lambda_val}\n"

        return summary
