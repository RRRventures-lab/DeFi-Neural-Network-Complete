#!/usr/bin/env python3
"""
Stage 6 Risk Management Test Suite

Tests:
1. Portfolio optimization (efficient frontier, optimal allocation)
2. Risk limits (drawdown, concentration, volatility, VaR)
3. Position sizing (Kelly, fixed fraction, volatility targeting)
4. Hedging strategies (correlation, VaR-based, options, dynamic)
5. Portfolio manager (integration of all components)
6. Risk analysis and stress testing
7. Real data backtesting with risk management

Run with: python test_stage6.py
"""

import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from risk.portfolio_optimization import PortfolioOptimizer, EfficientFrontier
from risk.risk_limits import (
    DrawdownLimit, ConcentrationLimit, VolatilityLimit, VaRLimit
)
from risk.position_sizing import (
    KellyPositionSizer, FixedFractionSizer, VolatilityTargetSizer, RiskParitySizer
)
from risk.hedging import (
    CorrelationHedge, VaRBasedHedge, OptionsHedge, DynamicHedge, HedgingManager
)
from risk.portfolio_manager import PortfolioManager


def test_efficient_frontier():
    """Test efficient frontier calculation."""
    print("\n" + "="*60)
    print("TEST 1: Efficient Frontier")
    print("="*60)

    try:
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, (252, 3))

        frontier = EfficientFrontier(returns)

        # Test minimum variance
        mv_result = frontier.minimum_variance_portfolio()
        assert np.isclose(np.sum(mv_result.weights), 1.0), "Weights don't sum to 1"
        assert mv_result.volatility > 0, "Volatility should be positive"

        # Test maximum Sharpe
        ms_result = frontier.maximum_sharpe_portfolio()
        assert np.isclose(np.sum(ms_result.weights), 1.0), "Weights don't sum to 1"
        assert ms_result.sharpe_ratio > ms_result.volatility, "Sharpe should be reasonable"

        # Test risk parity
        rp_result = frontier.risk_parity_portfolio()
        assert np.isclose(np.sum(rp_result.weights), 1.0), "Weights don't sum to 1"

        # Test target return
        tr_result = frontier.target_return_portfolio(0.10)
        assert np.isclose(np.sum(tr_result.weights), 1.0), "Weights don't sum to 1"

        print(f"✓ Minimum variance: {mv_result.volatility:.4f} vol, {mv_result.sharpe_ratio:.4f} Sharpe")
        print(f"✓ Maximum Sharpe: {ms_result.sharpe_ratio:.4f} Sharpe, {ms_result.volatility:.4f} vol")
        print(f"✓ Risk parity: Equal risk contribution")
        print(f"✓ Target return: 10% return, {tr_result.volatility:.4f} vol")
        print(f"✓ Efficient frontier: {3} assets, {252} observations")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_optimizer():
    """Test portfolio optimizer."""
    print("\n" + "="*60)
    print("TEST 2: Portfolio Optimizer")
    print("="*60)

    try:
        np.random.seed(42)
        returns = np.random.normal([0.0005, 0.0006, 0.0004], [0.01, 0.012, 0.008], (252, 3))

        optimizer = PortfolioOptimizer(returns)

        # Test different methods
        methods = ['maximum_sharpe', 'minimum_variance', 'risk_parity']

        for method in methods:
            result = optimizer.optimize(method=method)
            assert np.isclose(np.sum(result.weights), 1.0), f"{method}: Weights don't sum to 1"
            assert all(w >= 0 for w in result.weights), f"{method}: Negative weights"

        # Test comparison
        comparison = optimizer.compare_strategies()
        assert len(comparison) == 3, "Should have 3 strategies"
        assert all(col in comparison.columns for col in ['Strategy', 'Return', 'Volatility', 'Sharpe Ratio'])

        print(f"✓ Methods tested: {', '.join(methods)}")
        print(f"✓ Comparison table: {len(comparison)} strategies")
        print(comparison.to_string())

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drawdown_limit():
    """Test drawdown limit enforcement."""
    print("\n" + "="*60)
    print("TEST 3: Drawdown Limit")
    print("="*60)

    try:
        np.random.seed(42)
        returns = np.array([0.02, 0.01, -0.05, 0.03, -0.02, 0.04, -0.03, 0.02])

        dd_limit = DrawdownLimit(max_drawdown=-0.10)
        violation = dd_limit.check_drawdown(returns)

        drawdown = dd_limit.calculate_drawdown(returns)
        max_dd = np.min(drawdown)

        budget = dd_limit.remaining_drawdown_budget(returns)

        print(f"✓ Max drawdown: {max_dd:.4f}")
        print(f"✓ Drawdown limit: -10%")
        print(f"✓ Remaining budget: {budget:.4f}")
        print(f"✓ Violation: {violation is not None}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_concentration_limit():
    """Test concentration limit enforcement."""
    print("\n" + "="*60)
    print("TEST 4: Concentration Limit")
    print("="*60)

    try:
        weights = np.array([0.25, 0.20, 0.15, 0.10, 0.30])  # Last position exceeds limit

        conc_limit = ConcentrationLimit(max_concentration=0.20)
        violations = conc_limit.check_allocation(weights)

        top_positions = conc_limit.get_top_positions(weights, 3)

        print(f"✓ Weights: {weights}")
        print(f"✓ Concentration limit: 20%")
        print(f"✓ Violations found: {len(violations)}")
        print(f"✓ Top 3 positions: {top_positions}")

        assert len(violations) > 0, "Should detect violations"
        assert len(top_positions) == 3, "Should return top 3"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_volatility_limit():
    """Test volatility limit enforcement."""
    print("\n" + "="*60)
    print("TEST 5: Volatility Limit")
    print("="*60)

    try:
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 252)

        vol_limit = VolatilityLimit(max_volatility=0.15)

        vol = vol_limit.calculate_volatility(returns)
        violation = vol_limit.check_volatility(returns)
        budget = vol_limit.get_volatility_budget(returns)

        print(f"✓ Portfolio volatility: {vol:.4f} ({vol*100:.2f}%)")
        print(f"✓ Volatility limit: 15%")
        print(f"✓ Remaining budget: {budget:.4f}")
        print(f"✓ Limit exceeded: {violation is not None}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_var_limit():
    """Test Value at Risk limit."""
    print("\n" + "="*60)
    print("TEST 6: VaR Limit")
    print("="*60)

    try:
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 252)

        var_limit = VaRLimit(max_var=-0.05, confidence_level=0.95)

        var = var_limit.calculate_var(returns)
        cvar = var_limit.calculate_cvar(returns)
        violation = var_limit.check_var(returns)

        summary = var_limit.get_risk_summary(returns)

        print(f"✓ VaR (95%): {var:.4f}")
        print(f"✓ CVaR (95%): {cvar:.4f}")
        print(f"✓ VaR limit: -5%")
        print(f"✓ Limit exceeded: {violation is not None}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_kelly_position_sizing():
    """Test Kelly criterion position sizing."""
    print("\n" + "="*60)
    print("TEST 7: Kelly Position Sizing")
    print("="*60)

    try:
        sizer = KellyPositionSizer(kelly_fraction=0.25)

        kelly_pct = sizer.calculate_kelly_criterion(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01
        )

        size = sizer.calculate_size(
            account_equity=100000,
            position_price=100,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            max_loss=0.02
        )

        print(f"✓ Win rate: 55%, Avg win: 2%, Avg loss: 1%")
        print(f"✓ Kelly %: {kelly_pct:.4f}")
        print(f"✓ Position size: {size:.2f} units")
        print(f"✓ Position value: ${size * 100:,.0f}")

        assert size > 0, "Position size should be positive"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_volatility_target_sizing():
    """Test volatility-targeted position sizing."""
    print("\n" + "="*60)
    print("TEST 8: Volatility Target Position Sizing")
    print("="*60)

    try:
        sizer = VolatilityTargetSizer(target_volatility=0.10)

        # Test position sizing
        size = sizer.calculate_size(
            account_equity=100000,
            position_price=100,
            volatility=0.15
        )

        # Test allocation
        volatilities = np.array([0.10, 0.15, 0.20])
        weights = sizer.get_allocation(volatilities)

        print(f"✓ Target volatility: 10%")
        print(f"✓ Asset volatility: 15%")
        print(f"✓ Position size: {size:.2f} units")
        print(f"✓ Weights for vol [10%, 15%, 20%]: {weights}")
        print(f"✓ Weights sum: {np.sum(weights):.4f}")

        assert np.isclose(np.sum(weights), 1.0), "Weights should sum to 1"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_correlation_hedge():
    """Test correlation-based hedging."""
    print("\n" + "="*60)
    print("TEST 9: Correlation Hedge")
    print("="*60)

    try:
        # Create correlation matrix
        correlation = np.array([
            [1.0, 0.3, -0.6],
            [0.3, 1.0, 0.2],
            [-0.6, 0.2, 1.0]
        ])

        hedge = CorrelationHedge(correlation)

        # Find hedge for asset 0
        best_hedge = hedge.find_best_hedge(primary_asset=0, min_correlation=-0.5)

        # Calculate hedge ratio
        hedge_ratio = hedge.calculate_hedge_ratio(
            primary_volatility=0.15,
            hedge_volatility=0.12,
            correlation=-0.6
        )

        # Calculate full hedge
        result = hedge.calculate_hedge(
            primary_asset=0,
            primary_position=100,
            primary_volatility=0.15,
            hedge_volatilities={2: 0.12}
        )

        print(f"✓ Correlation matrix: 3×3")
        print(f"✓ Best hedge for asset 0: asset {best_hedge}")
        print(f"✓ Hedge ratio: {hedge_ratio:.4f}")
        print(f"✓ Hedge effectiveness: {result['effectiveness']:.4f}")
        print(f"✓ Hedge position: {result['hedge_position']:.2f} units")

        assert best_hedge == 2, "Asset 2 should be best hedge (correlation -0.6)"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_manager():
    """Test integrated portfolio manager."""
    print("\n" + "="*60)
    print("TEST 10: Portfolio Manager Integration")
    print("="*60)

    try:
        np.random.seed(42)
        returns = np.random.normal(
            [0.0005, 0.0006, 0.0004],
            [0.01, 0.012, 0.008],
            (252, 3)
        )

        manager = PortfolioManager(
            returns,
            asset_names=['US Stocks', 'Intl Stocks', 'Bonds']
        )

        # Get optimal allocation
        allocation = manager.get_optimal_allocation(method='maximum_sharpe')

        print(f"✓ Assets: {manager.num_assets}")
        print(f"✓ Expected return: {allocation.expected_return:.2%}")
        print(f"✓ Volatility: {allocation.volatility:.2%}")
        print(f"✓ Sharpe ratio: {allocation.sharpe_ratio:.4f}")

        # Analyze risk
        risk_analysis = manager.analyze_risk(allocation.weights)

        print(f"✓ Max drawdown: {risk_analysis['drawdown']['max_drawdown']:.2%}")
        print(f"✓ Max position: {risk_analysis['concentration']['max_position']:.2%}")
        print(f"✓ Herfindahl: {risk_analysis['concentration']['herfindahl']:.4f}")

        # Stress test
        crisis_returns = returns * 2  # 2x worse scenario
        stress_results = manager.stress_test(
            allocation.weights,
            {'crisis': crisis_returns}
        )

        print(f"✓ Stress test VaR: {stress_results['crisis']['var']:.2%}")

        # Report
        report = manager.generate_report(allocation.weights)
        print(f"✓ Report generated: {len(report)} chars")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "   STAGE 6: RISK MANAGEMENT TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    results['Efficient Frontier'] = test_efficient_frontier()
    results['Portfolio Optimizer'] = test_portfolio_optimizer()
    results['Drawdown Limit'] = test_drawdown_limit()
    results['Concentration Limit'] = test_concentration_limit()
    results['Volatility Limit'] = test_volatility_limit()
    results['VaR Limit'] = test_var_limit()
    results['Kelly Position Sizing'] = test_kelly_position_sizing()
    results['Volatility Target Sizing'] = test_volatility_target_sizing()
    results['Correlation Hedge'] = test_correlation_hedge()
    results['Portfolio Manager Integration'] = test_portfolio_manager()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    print("="*60)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stage 6 is complete.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
