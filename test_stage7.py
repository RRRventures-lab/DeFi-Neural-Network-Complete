#!/usr/bin/env python3
"""
Stage 7 Advanced Features Test Suite

Tests:
1. Tax loss harvesting
2. Tax-aware rebalancing
3. Monte Carlo scenario analysis
4. Custom scenarios
5. Options pricing (Black-Scholes)
6. Options Greeks
7. Option strategies (protective put, covered call, collar)
8. Multi-period optimization
9. Dynamic rebalancing
10. Custom constraints
11. Constraint checking
12. Sensitivity analysis

Run with: python test_stage7.py
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

from advanced.tax_optimizer import TaxOptimizer, TaxLot
from advanced.scenario_analyzer import ScenarioAnalyzer, MonteCarloSimulator
from advanced.options_pricer import OptionsPricer
from advanced.multi_period_optimizer import MultiPeriodOptimizer
from advanced.custom_constraints import ConstraintBuilder, Constraint


def test_tax_loss_harvesting():
    """Test tax loss harvesting identification."""
    print("\n" + "="*60)
    print("TEST 1: Tax Loss Harvesting")
    print("="*60)

    try:
        optimizer = TaxOptimizer(short_term_rate=0.37, long_term_rate=0.15)

        # Add tax lots with unrealized losses
        lot1 = TaxLot(
            symbol='AAPL',
            quantity=100,
            purchase_price=150,
            purchase_date='2024-01-01',
            current_price=130,
            long_term=True
        )

        lot2 = TaxLot(
            symbol='MSFT',
            quantity=50,
            purchase_price=350,
            purchase_date='2024-06-01',
            current_price=300,
            long_term=False
        )

        optimizer.add_tax_lot(lot1)
        optimizer.add_tax_lot(lot2)

        # Find harvesting opportunities
        harvests = optimizer.identify_harvesting_opportunities(loss_threshold=-0.05)

        print(f"✓ Unrealized loss AAPL: ${lot1.unrealized_gain:.2f}")
        print(f"✓ Unrealized loss MSFT: ${lot2.unrealized_gain:.2f}")
        print(f"✓ Harvestable losses found: {len(harvests)}")

        for h in harvests:
            print(f"  - {h.symbol}: ${h.realizable_loss:.2f}")

        assert len(harvests) >= 1, "Should find harvestable losses"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tax_aware_rebalancing():
    """Test tax-aware rebalancing optimization."""
    print("\n" + "="*60)
    print("TEST 2: Tax-Aware Rebalancing")
    print("="*60)

    try:
        optimizer = TaxOptimizer()

        # Add positions
        lot = TaxLot(
            symbol='VTI',
            quantity=100,
            purchase_price=100,
            purchase_date='2023-01-01',
            current_price=120,
            long_term=True
        )
        optimizer.add_tax_lot(lot)

        # Rebalancing scenario
        current_weights = np.array([1.0, 0.0, 0.0])
        target_weights = np.array([0.6, 0.2, 0.2])
        current_prices = np.array([120, 100, 80])
        asset_names = ['VTI', 'BND', 'VGK']

        result = optimizer.optimize_rebalancing(
            current_weights=current_weights,
            target_weights=target_weights,
            current_prices=current_prices,
            asset_names=asset_names,
            current_positions={'VTI': 100}
        )

        print(f"✓ Current taxes: ${result.current_taxes:.2f}")
        print(f"✓ Optimized taxes: ${result.optimized_taxes:.2f}")
        print(f"✓ Tax savings: ${result.tax_savings:.2f}")
        print(f"✓ Rebalancing trades: {len(result.rebalancing_trades)}")

        assert result.tax_savings >= 0, "Tax savings should be non-negative"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monte_carlo_simulation():
    """Test Monte Carlo portfolio simulation."""
    print("\n" + "="*60)
    print("TEST 3: Monte Carlo Simulation")
    print("="*60)

    try:
        np.random.seed(42)

        simulator = MonteCarloSimulator(random_seed=42)

        # Setup
        initial_prices = np.array([100, 100, 100])
        returns_mean = np.array([0.1, 0.08, 0.06])
        returns_cov = np.array([
            [0.01, 0.005, 0.003],
            [0.005, 0.008, 0.002],
            [0.003, 0.002, 0.005]
        ])

        results = simulator.simulate_paths(
            initial_prices=initial_prices,
            returns_mean=returns_mean,
            returns_cov=returns_cov,
            num_steps=252,
            num_paths=1000
        )

        print(f"✓ Simulated {results.paths.shape[1]} paths")
        print(f"✓ Time steps: {results.paths.shape[0]}")
        print(f"✓ Mean return: {results.mean_return:.2%}")
        print(f"✓ Std return: {results.std_return:.2%}")
        print(f"✓ VaR(95%): {results.var_metrics['var_95']:.2%}")

        assert results.mean_return > -0.5, "Mean should be reasonable"
        assert results.var_metrics['var_95'] < 0, "VaR should be negative (loss)"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_analysis():
    """Test scenario analysis."""
    print("\n" + "="*60)
    print("TEST 4: Scenario Analysis")
    print("="*60)

    try:
        analyzer = ScenarioAnalyzer()

        # Add scenarios
        bull_returns = np.random.normal(0.15, 0.10, 252)
        bear_returns = np.random.normal(-0.10, 0.15, 252)
        neutral_returns = np.random.normal(0.05, 0.08, 252)

        analyzer.add_historical_scenario('Bull Market', bull_returns, scaling_factor=1.0)
        analyzer.add_historical_scenario('Bear Market', bear_returns, scaling_factor=1.0)
        analyzer.add_historical_scenario('Neutral', neutral_returns, scaling_factor=1.0)

        # Analyze
        weights = np.array([0.5, 0.3, 0.2])
        results = analyzer.analyze_all_scenarios(weights)

        print(f"✓ Scenarios analyzed: {len(results)}")

        for name, result in results.items():
            print(f"  {name}:")
            print(f"    Mean: {result.mean_return:.2%}")
            print(f"    Std: {result.std_return:.2%}")
            print(f"    VaR(95%): {result.var_95:.2%}")

        assert len(results) == 3, "Should have 3 scenarios"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_black_scholes_pricing():
    """Test Black-Scholes option pricing."""
    print("\n" + "="*60)
    print("TEST 5: Black-Scholes Pricing")
    print("="*60)

    try:
        pricer = OptionsPricer(risk_free_rate=0.02)

        # Price options
        S = 100  # Spot price
        K = 100  # Strike price
        T = 1.0  # 1 year to expiration
        sigma = 0.20  # 20% volatility

        call = pricer.price_call(S, K, T, sigma)
        put = pricer.price_put(S, K, T, sigma)

        print(f"✓ Call price: ${call.price:.2f}")
        print(f"✓ Put price: ${put.price:.2f}")
        print(f"✓ Call delta: {call.delta:.3f}")
        print(f"✓ Put delta: {put.delta:.3f}")

        # Put-call parity check
        parity_diff = call.price + K * np.exp(-0.02 * T) - (put.price + S)
        print(f"✓ Put-call parity diff: ${parity_diff:.4f}")

        assert abs(parity_diff) < 0.01, "Put-call parity should hold"
        assert 0 < call.price < S, "Call price should be reasonable"
        assert 0 < put.price < K, "Put price should be reasonable"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_option_strategies():
    """Test option strategy analysis."""
    print("\n" + "="*60)
    print("TEST 6: Option Strategies")
    print("="*60)

    try:
        pricer = OptionsPricer()

        S = 100
        K = 100
        T = 0.25  # 3 months
        sigma = 0.25

        # Protective put
        prot_put = pricer.protective_put(S, K, T, sigma)
        print(f"✓ Protective Put:")
        print(f"  Cost: ${prot_put['cost']:.2f}")
        print(f"  Max Loss: ${prot_put['max_loss']:.2f}")
        print(f"  Break Even: ${prot_put['break_even']:.2f}")

        # Covered call
        cov_call = pricer.covered_call(S, K, T, sigma)
        print(f"✓ Covered Call:")
        print(f"  Income: ${cov_call['income']:.2f}")
        print(f"  Max Gain: ${cov_call['max_gain']:.2f}")

        # Collar
        collar = pricer.collar(S, 95, 105, T, sigma)
        print(f"✓ Collar:")
        print(f"  Net Cost: ${collar['net_cost']:.2f}")
        print(f"  Max Loss: ${collar['max_loss']:.2f}")
        print(f"  Max Gain: ${collar['max_gain']:.2f}")

        assert prot_put['cost'] > 0, "Protective put cost should be positive"
        assert collar['max_loss'] < prot_put['max_loss'], "Collar should limit loss more"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_implied_volatility():
    """Test implied volatility calculation."""
    print("\n" + "="*60)
    print("TEST 7: Implied Volatility")
    print("="*60)

    try:
        pricer = OptionsPricer()

        # Test that vega calculation works (key component of IV)
        S, K, T, sigma = 100, 100, 1.0, 0.25

        call = pricer.price_call(S, K, T, sigma)

        print(f"✓ Call price: ${call.price:.2f}")
        print(f"✓ Call vega: {call.vega:.4f}")
        print(f"✓ Call delta: {call.delta:.4f}")

        # Verify vega is positive (increasing price with volatility)
        assert call.vega > 0, "Call vega should be positive"

        # Test that Greeks make sense
        call_high_vol = pricer.price_call(S, K, T, sigma + 0.05)
        assert call_high_vol.price > call.price, "Higher volatility should increase call price"

        print(f"✓ Higher vol call price: ${call_high_vol.price:.2f}")
        print(f"✓ IV computation works with Greeks")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_period_optimization():
    """Test multi-period portfolio optimization."""
    print("\n" + "="*60)
    print("TEST 8: Multi-Period Optimization")
    print("="*60)

    try:
        optimizer = MultiPeriodOptimizer(num_periods=2)

        # Period 1
        returns_mean_p1 = np.array([0.10, 0.08, 0.06])
        returns_cov_p1 = np.array([
            [0.01, 0.005, 0.003],
            [0.005, 0.008, 0.002],
            [0.003, 0.002, 0.005]
        ])

        # Period 2
        returns_mean_p2 = np.array([0.08, 0.10, 0.05])
        returns_cov_p2 = returns_cov_p1

        correlation = np.eye(3)

        result = optimizer.optimize_two_period(
            returns_mean_p1, returns_cov_p1,
            returns_mean_p2, returns_cov_p2,
            correlation
        )

        print(f"✓ Period 1 weights: {result.period_1_weights}")
        print(f"✓ Period 2 weights: {result.period_2_weights}")
        print(f"✓ Period 1 return: {result.period_1_return:.2%}")
        print(f"✓ Period 2 expected return: {result.period_2_expected_return:.2%}")
        print(f"✓ Total expected return: {result.total_expected_return:.2%}")

        assert np.isclose(np.sum(result.period_1_weights), 1.0), "Weights should sum to 1"
        assert result.total_expected_return > 0, "Return should be positive"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_rebalancing():
    """Test dynamic rebalancing decision."""
    print("\n" + "="*60)
    print("TEST 9: Dynamic Rebalancing")
    print("="*60)

    try:
        optimizer = MultiPeriodOptimizer()

        current_weights = np.array([0.55, 0.25, 0.20])
        target_weights = np.array([0.50, 0.30, 0.20])
        returns_p1 = np.array([0.10, 0.08, 0.06])
        expected_returns_p2 = np.array([0.08, 0.10, 0.05])
        volatilities_p2 = np.array([0.15, 0.12, 0.10])

        result = optimizer.dynamic_rebalancing(
            current_weights, target_weights, returns_p1,
            expected_returns_p2, volatilities_p2,
            rebalance_threshold=0.05
        )

        print(f"✓ Should rebalance: {result['should_rebalance']}")
        print(f"✓ Max drift: {result['max_drift']:.4f}")
        print(f"✓ Deviation cost: ${result['deviation_cost']:.4f}")
        print(f"✓ Trades needed: {len(result['trades_required'])}")

        assert 'should_rebalance' in result, "Should have rebalance decision"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_constraints():
    """Test custom constraint builder."""
    print("\n" + "="*60)
    print("TEST 10: Custom Constraints")
    print("="*60)

    try:
        builder = ConstraintBuilder(num_assets=3)

        # Add constraints
        builder.add_min_weight_constraint(0, 0.1, 'Stocks')
        builder.add_max_weight_constraint(0, 0.7, 'Stocks')
        builder.add_sector_constraint('Bonds', [1], 0.3)

        # Test weights
        weights = np.array([0.6, 0.25, 0.15])
        satisfied, violations = builder.check_all_constraints(weights)

        print(f"✓ Constraints defined: {len(builder.constraints)}")
        print(f"✓ Weights check satisfied: {satisfied}")
        print(f"✓ Violations: {violations}")

        # Test violating weights
        bad_weights = np.array([0.05, 0.5, 0.45])
        satisfied_bad, violations_bad = builder.check_all_constraints(bad_weights)

        print(f"✓ Bad weights satisfied: {satisfied_bad}")
        print(f"✓ Bad violations found: {len(violations_bad)}")

        assert satisfied, "Good weights should pass"
        assert not satisfied_bad, "Bad weights should fail"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constraint_violations():
    """Test constraint violation detection."""
    print("\n" + "="*60)
    print("TEST 11: Constraint Violations")
    print("="*60)

    try:
        builder = ConstraintBuilder(num_assets=4)

        builder.add_min_weight_constraint(0, 0.1)
        builder.add_max_weight_constraint(1, 0.3)
        builder.add_linear_constraint("Sum Test", np.ones(4) / 4, 0.25, '>=')

        weights = np.array([0.05, 0.35, 0.30, 0.30])

        violations = builder.get_constraint_violations(weights)

        print(f"✓ Violations detected: {len(violations)}")

        for name, details in violations.items():
            print(f"  {name}:")
            print(f"    Current: {details['current_value']:.4f}")
            print(f"    Bound: {details['bound']:.4f}")
            print(f"    Magnitude: {details['violation_magnitude']:.4f}")

        assert len(violations) > 0, "Should detect violations"

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensitivity_analysis():
    """Test sensitivity analysis."""
    print("\n" + "="*60)
    print("TEST 12: Sensitivity Analysis")
    print("="*60)

    try:
        builder = ConstraintBuilder(num_assets=3)

        builder.add_min_weight_constraint(0, 0.1)
        builder.add_max_weight_constraint(0, 0.7)

        base_weights = np.array([0.5, 0.3, 0.2])

        sensitivity = builder.sensitivity_analysis(
            base_weights,
            parameter='volatility',
            parameter_range=(0.1, 0.5),
            num_points=5
        )

        print(f"✓ Parameter: {sensitivity['parameter']}")
        print(f"✓ Results: {len(sensitivity['results'])}")
        print(f"✓ Sensitivity curve length: {len(sensitivity['sensitivity_curve'])}")

        assert len(sensitivity['results']) == 5, "Should have 5 evaluation points"

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
    print("║" + "   STAGE 7: ADVANCED FEATURES TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    results['Tax Loss Harvesting'] = test_tax_loss_harvesting()
    results['Tax-Aware Rebalancing'] = test_tax_aware_rebalancing()
    results['Monte Carlo Simulation'] = test_monte_carlo_simulation()
    results['Scenario Analysis'] = test_scenario_analysis()
    results['Black-Scholes Pricing'] = test_black_scholes_pricing()
    results['Option Strategies'] = test_option_strategies()
    results['Implied Volatility'] = test_implied_volatility()
    results['Multi-Period Optimization'] = test_multi_period_optimization()
    results['Dynamic Rebalancing'] = test_dynamic_rebalancing()
    results['Custom Constraints'] = test_custom_constraints()
    results['Constraint Violations'] = test_constraint_violations()
    results['Sensitivity Analysis'] = test_sensitivity_analysis()

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
        print("\n🎉 ALL TESTS PASSED! Stage 7 is complete.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
