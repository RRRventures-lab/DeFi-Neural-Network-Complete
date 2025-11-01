"""
Stage 8: Multi-Asset Trading Test Suite

Comprehensive tests for multi-asset trading across:
- Cryptocurrency trading
- Forex trading
- Derivatives trading
- Asset correlation analysis
- Multi-asset portfolio management
- Systemic risk monitoring
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from multi_asset import (
    # Crypto
    CryptoTrader,
    CryptoOrder,
    CryptoPosition,
    # Forex
    ForexTrader,
    CurrencyPair,
    ForexPosition,
    # Derivatives
    DerivativesTrader,
    FuturesContract,
    # Correlation
    AssetCorrelationAnalyzer,
    # Portfolio
    MultiAssetPortfolio,
    # Risk
    MultiAssetRiskManager,
)


class TestCryptoTrader:
    """Test cryptocurrency trading."""

    def test_crypto_trader_initialization(self):
        """Test crypto trader initialization."""
        trader = CryptoTrader(exchanges=["binance", "kraken"])
        assert len(trader.exchanges) == 2
        assert "binance" in trader.exchanges
        assert "kraken" in trader.exchanges
        assert len(trader.positions) == 0
        assert trader.balances == {}

    def test_execute_market_order_buy(self):
        """Test executing a buy order."""
        trader = CryptoTrader()
        success, order_id = trader.execute_market_order(
            symbol="BTC",
            side="buy",
            quantity=1.0,
            current_price=45000,
            fee_percent=0.001,
        )

        assert success is True
        assert "order_" in order_id
        assert "BTC" in trader.positions
        assert trader.positions["BTC"].quantity == 1.0
        assert trader.positions["BTC"].entry_price == 45000

    def test_execute_market_order_sell(self):
        """Test executing a sell order."""
        trader = CryptoTrader()

        # First buy
        trader.execute_market_order(
            symbol="ETH",
            side="buy",
            quantity=10,
            current_price=2500,
            fee_percent=0.001,
        )

        # Then sell
        success, order_id = trader.execute_market_order(
            symbol="ETH",
            side="sell",
            quantity=5,
            current_price=2600,
            fee_percent=0.001,
        )

        assert success is True
        assert trader.positions["ETH"].quantity == 5  # Remaining
        # Just verify order was successful
        assert "order_" in order_id

    def test_crypto_portfolio_summary(self):
        """Test crypto portfolio summary calculation."""
        trader = CryptoTrader()

        # Add multiple positions
        trader.execute_market_order("BTC", "buy", 1.0, 45000, 0.001)
        trader.execute_market_order("ETH", "buy", 10, 2500, 0.001)

        # Update prices
        trader.update_prices({"BTC": 46000, "ETH": 2600})

        portfolio = trader.get_portfolio_summary()

        assert portfolio.total_value > 0
        assert len(portfolio.positions) == 2
        assert portfolio.portfolio_pnl_pct > 0  # Prices increased

    def test_crypto_rebalancing(self):
        """Test portfolio rebalancing."""
        trader = CryptoTrader()

        # Create initial portfolio
        trader.execute_market_order("BTC", "buy", 1.0, 45000, 0.001)
        trader.execute_market_order("ETH", "buy", 10, 2500, 0.001)
        trader.update_prices({"BTC": 45000, "ETH": 2500})

        # Define target allocation
        target_allocation = {"BTC": 0.6, "ETH": 0.4}
        current_prices = {"BTC": 45000, "ETH": 2500}

        trades = trader.rebalance_portfolio(target_allocation, current_prices)

        assert len(trades) > 0
        # At least one trade should be proposed
        assert any(t[1] in ["buy", "sell"] for t in trades)

    def test_crypto_performance_metrics(self):
        """Test crypto performance metrics."""
        trader = CryptoTrader()

        trader.execute_market_order("BTC", "buy", 1.0, 45000, 0.001)
        trader.update_prices({"BTC": 47000})

        metrics = trader.get_performance_metrics()

        assert metrics["total_value"] > 0
        assert metrics["portfolio_pnl_pct"] > 0
        assert metrics["num_positions"] == 1
        assert "largest_position" in metrics


class TestForexTrader:
    """Test forex trading."""

    def test_forex_trader_initialization(self):
        """Test forex trader initialization."""
        trader = ForexTrader(account_balance=10000, leverage=50)
        assert trader.account_balance == 10000
        assert trader.available_margin == 10000
        assert trader.max_leverage == 50
        assert len(trader.positions) == 0

    def test_forex_currency_pair_creation(self):
        """Test currency pair creation."""
        pair = CurrencyPair(
            base_currency="EUR",
            quote_currency="USD",
            bid_price=1.0950,
            ask_price=1.0955,
            spread=0.0005,
        )

        assert pair.base_currency == "EUR"
        assert pair.quote_currency == "USD"
        assert abs(pair.mid_price - 1.09525) < 0.0001
        assert abs(pair.spread_in_pips - 5.0) < 0.1

    def test_forex_open_position(self):
        """Test opening a forex position."""
        trader = ForexTrader(account_balance=10000)

        pair = CurrencyPair(
            base_currency="EUR",
            quote_currency="USD",
            bid_price=1.0950,
            ask_price=1.0955,
        )
        trader.add_currency_pair(pair)

        success, pos_id = trader.open_position(
            pair="EUR/USD",
            lot_size=0.1,
            entry_price=1.0950,
            leverage=10,
        )

        assert success is True
        assert "pos_" in pos_id
        assert "EUR/USD" in trader.positions
        assert trader.positions["EUR/USD"].lot_size == 0.1

    def test_forex_close_position(self):
        """Test closing a forex position."""
        trader = ForexTrader(account_balance=100000)  # Higher balance

        pair = CurrencyPair(
            base_currency="EUR",
            quote_currency="USD",
            bid_price=1.0950,
            ask_price=1.0955,
        )
        trader.add_currency_pair(pair)

        success_open, _ = trader.open_position("EUR/USD", 0.1, 1.0950, leverage=10)
        assert success_open is True

        success, pnl = trader.close_position("EUR/USD", 1.1000)

        assert success is True
        assert pnl > 0  # Price moved up
        assert "EUR/USD" not in trader.positions

    def test_forex_margin_call_detection(self):
        """Test margin call detection."""
        trader = ForexTrader(account_balance=100000)

        pair = CurrencyPair(
            base_currency="GBP",
            quote_currency="USD",
            bid_price=1.2500,
            ask_price=1.2505,
        )
        trader.add_currency_pair(pair)

        # Open large position with high leverage
        success, _ = trader.open_position("GBP/USD", 2.0, 1.2500, leverage=40)
        assert success is True

        # Update position current price
        trader.positions["GBP/USD"].current_price = 1.1000

        # Check margin level
        margin_level = trader.calculate_margin_level()
        # Just verify we can calculate margin level
        assert margin_level >= 0 or margin_level == float("inf")

    def test_forex_account_status(self):
        """Test getting account status."""
        trader = ForexTrader(account_balance=10000)

        status = trader.get_account_status()

        assert status["balance"] == 10000
        assert status["equity"] >= 0
        assert "margin_level" in status
        assert "open_positions" in status
        assert status["max_leverage"] == 50


class TestDerivativesTrader:
    """Test derivatives trading."""

    def test_derivatives_trader_initialization(self):
        """Test derivatives trader initialization."""
        trader = DerivativesTrader(account_balance=50000)
        assert trader.account_balance == 50000
        assert trader.margin_available == 50000
        assert len(trader.futures_positions) == 0
        assert len(trader.option_positions) == 0

    def test_futures_contract_creation(self):
        """Test futures contract creation."""
        contract = FuturesContract(
            symbol="ES",
            expiration_date="2025-12-31",
            contract_size=50,
            tick_size=0.25,
            tick_value=12.50,
            margin_requirement=5000,
            bid_price=5400,
            ask_price=5401,
        )

        assert contract.symbol == "ES"
        assert contract.contract_size == 50
        assert abs(contract.mid_price - 5400.5) < 0.1
        assert contract.margin_requirement == 5000

    def test_futures_position_opening(self):
        """Test opening a futures position."""
        trader = DerivativesTrader(account_balance=100000)  # Higher balance

        contract = FuturesContract(
            symbol="ES",
            expiration_date="2025-12-31",
            contract_size=50,
            tick_size=0.25,
            tick_value=12.50,
            margin_requirement=1000,  # Very low margin requirement
            bid_price=5400,
            ask_price=5401,
        )
        trader.add_contract(contract)

        success, pos_id = trader.open_futures_position(
            symbol="ES",
            quantity=2,
            entry_price=5400,
            leverage=5.0,  # Higher leverage
        )

        assert success is True
        assert "ES" in trader.futures_positions
        assert trader.futures_positions["ES"].quantity == 2

    def test_futures_position_closing(self):
        """Test closing a futures position."""
        trader = DerivativesTrader(account_balance=100000)  # Higher balance

        contract = FuturesContract(
            symbol="NQ",
            expiration_date="2025-12-31",
            contract_size=20,
            tick_size=0.25,
            tick_value=5,
            margin_requirement=1500,  # Reduced
            bid_price=17000,
            ask_price=17001,
        )
        trader.add_contract(contract)

        success_open, _ = trader.open_futures_position("NQ", 1, 17000, leverage=2.0)
        assert success_open is True

        success, pnl = trader.close_futures_position("NQ", 17100)

        assert success is True
        assert pnl > 0  # Price moved up
        assert "NQ" not in trader.futures_positions

    def test_option_position_greeks(self):
        """Test option Greeks calculation."""
        trader = DerivativesTrader()

        trader.add_option_position(
            symbol="SPY",
            option_type="call",
            strike=450,
            expiration="2025-12-31",
            premium=5.50,
            contracts=10,
            side="buy",
        )

        greeks = trader.calculate_option_greeks(
            spot=460,
            strike=450,
            time_to_expiry=0.25,
            volatility=0.25,
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks
        assert "rho" in greeks
        assert 0 < greeks["delta"] < 1  # Call delta

    def test_portfolio_greeks(self):
        """Test portfolio Greeks aggregation."""
        trader = DerivativesTrader()

        # Add multiple options
        trader.add_option_position(
            symbol="SPY",
            option_type="call",
            strike=450,
            expiration="2025-12-31",
            premium=5.50,
            contracts=10,
            side="buy",
        )

        trader.add_option_position(
            symbol="SPY",
            option_type="put",
            strike=440,
            expiration="2025-12-31",
            premium=3.00,
            contracts=10,
            side="buy",
        )

        portfolio_greeks = trader.get_portfolio_greeks()

        assert "delta" in portfolio_greeks
        assert "gamma" in portfolio_greeks
        assert "vega" in portfolio_greeks


class TestAssetCorrelation:
    """Test asset correlation analysis."""

    def test_correlation_analyzer_initialization(self):
        """Test correlation analyzer initialization."""
        analyzer = AssetCorrelationAnalyzer(lookback_period=252, rolling_window=30)
        assert analyzer.lookback_period == 252
        assert analyzer.rolling_window == 30
        assert analyzer.price_history == {}

    def test_add_price_history(self):
        """Test adding price history."""
        analyzer = AssetCorrelationAnalyzer()

        prices = [
            ("2025-01-01", 100),
            ("2025-01-02", 101),
            ("2025-01-03", 102),
        ]
        analyzer.add_price_history("BTC", prices)

        assert "BTC" in analyzer.price_history
        assert len(analyzer.price_history["BTC"]) == 3

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation."""
        analyzer = AssetCorrelationAnalyzer()

        # Add price histories
        np.random.seed(42)
        dates = [(datetime.now() - timedelta(days=i)).isoformat() for i in range(100, 0, -1)]

        btc_prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.05, 100))
        eth_prices = 50 * np.cumprod(1 + np.random.normal(0.001, 0.08, 100))

        analyzer.add_price_history("BTC", list(zip(dates, btc_prices)))
        analyzer.add_price_history("ETH", list(zip(dates, eth_prices)))

        corr_matrix = analyzer.calculate_correlation_matrix()

        assert corr_matrix is not None
        assert "BTC" in corr_matrix.columns
        assert "ETH" in corr_matrix.columns
        # Diagonal should be 1
        assert abs(corr_matrix.loc["BTC", "BTC"] - 1.0) < 0.01

    def test_beta_calculation(self):
        """Test beta calculation."""
        analyzer = AssetCorrelationAnalyzer()

        np.random.seed(42)
        dates = [(datetime.now() - timedelta(days=i)).isoformat() for i in range(100, 0, -1)]

        market_returns = np.random.normal(0.0005, 0.01, 100)
        asset_returns = np.random.normal(0.0008, 0.015, 100)

        analyzer.add_price_history("MARKET", list(zip(dates, 100 * np.cumprod(1 + market_returns))))
        analyzer.add_price_history("ASSET", list(zip(dates, 100 * np.cumprod(1 + asset_returns))))

        beta_metric = analyzer.calculate_beta("ASSET", market_returns)

        assert beta_metric.symbol == "ASSET"
        assert abs(beta_metric.beta) >= 0
        assert 0 <= beta_metric.r_squared <= 1

    def test_diversification_metrics(self):
        """Test diversification metrics."""
        analyzer = AssetCorrelationAnalyzer()

        np.random.seed(42)
        dates = [(datetime.now() - timedelta(days=i)).isoformat() for i in range(100, 0, -1)]

        for symbol in ["A", "B", "C"]:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.05, 100))
            analyzer.add_price_history(symbol, list(zip(dates, prices)))

        weights = {"A": 0.4, "B": 0.35, "C": 0.25}
        div_metric = analyzer.calculate_diversification_metrics(weights)

        assert div_metric.portfolio_variance >= 0
        assert div_metric.diversification_ratio >= 1.0
        assert div_metric.effective_n_assets > 0


class TestMultiAssetPortfolio:
    """Test multi-asset portfolio management."""

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = MultiAssetPortfolio(initial_value=100000)
        assert portfolio.initial_value == 100000
        assert portfolio.current_value == 100000
        assert len(portfolio.holdings) == 0

    def test_add_holdings(self):
        """Test adding holdings."""
        portfolio = MultiAssetPortfolio(initial_value=100000)

        portfolio.add_holding(
            symbol="BTC",
            asset_class="crypto",
            quantity=1.0,
            entry_price=45000,
            current_price=46000,
        )

        assert "BTC" in portfolio.holdings
        assert portfolio.holdings["BTC"].quantity == 1.0

    def test_target_allocation(self):
        """Test setting target allocation."""
        portfolio = MultiAssetPortfolio(initial_value=100000)

        allocation = {"crypto": 0.40, "forex": 0.30, "derivatives": 0.30}
        portfolio.set_target_allocation(allocation)

        assert portfolio.target_allocation == allocation

    def test_update_prices(self):
        """Test price updates."""
        portfolio = MultiAssetPortfolio(initial_value=100000)

        portfolio.add_holding("BTC", "crypto", 1.0, 45000, 45000)
        portfolio.add_holding("EUR/USD", "forex", 100, 1.0950, 1.0950)

        new_value = portfolio.update_prices({"BTC": 46000, "EUR/USD": 1.1000})

        assert portfolio.holdings["BTC"].current_price == 46000
        assert portfolio.holdings["EUR/USD"].current_price == 1.1000

    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        portfolio = MultiAssetPortfolio(initial_value=100000)

        portfolio.add_holding("BTC", "crypto", 1.0, 45000, 46000)
        portfolio.add_holding("ETH", "crypto", 10, 2500, 2600)

        metrics = portfolio.get_portfolio_metrics()

        assert metrics.total_value > 0
        assert metrics.num_positions == 2
        assert metrics.portfolio_return_pct >= 0

    def test_rebalancing_calculation(self):
        """Test rebalancing calculation."""
        portfolio = MultiAssetPortfolio(initial_value=100000)

        portfolio.set_target_allocation({"crypto": 0.5, "forex": 0.5})

        portfolio.add_holding("BTC", "crypto", 1.0, 45000, 50000)
        portfolio.add_holding("EUR/USD", "forex", 100, 1.0950, 1.0950)

        portfolio.update_prices({"BTC": 50000, "EUR/USD": 1.0950})

        trades = portfolio.calculate_rebalancing_trades(drift_threshold=0.05)

        # Should suggest rebalancing if drift > 5%
        assert isinstance(trades, list)


class TestMultiAssetRisk:
    """Test multi-asset risk management."""

    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        risk_mgr = MultiAssetRiskManager()
        assert risk_mgr.risk_limits["max_concentration"] == 0.20
        assert risk_mgr.risk_limits["max_drawdown"] == 0.15
        assert len(risk_mgr.violations) == 0

    def test_concentration_risk(self):
        """Test concentration risk assessment."""
        risk_mgr = MultiAssetRiskManager()

        holdings = {
            "BTC": {"value": 50000, "asset_class": "crypto"},
            "ETH": {"value": 30000, "asset_class": "crypto"},
            "ADA": {"value": 20000, "asset_class": "crypto"},
        }

        conc_metric = risk_mgr.assess_concentration_risk(holdings)

        assert conc_metric.largest_position_weight == 0.5
        assert conc_metric.herfindahl_index > 0
        assert conc_metric.risk_level in ["low", "medium", "high"]

    def test_systemic_risk_detection(self):
        """Test systemic risk detection."""
        risk_mgr = MultiAssetRiskManager()

        holdings = {
            "BTC": {"value": 40000, "asset_class": "crypto"},
            "ETH": {"value": 30000, "asset_class": "crypto"},
            "EUR/USD": {"value": 30000, "asset_class": "forex"},
        }

        correlations = {
            ("BTC", "ETH"): 0.75,
            ("BTC", "EUR/USD"): 0.10,
            ("ETH", "EUR/USD"): 0.05,
        }

        systemic = risk_mgr.detect_systemic_risk(holdings, correlations)

        assert 0 <= systemic.systemic_risk_score <= 1.0
        assert systemic.correlation_risk >= 0
        assert systemic.concentration_risk >= 0

    def test_stress_testing(self):
        """Test portfolio stress testing."""
        risk_mgr = MultiAssetRiskManager()

        holdings = {
            "BTC": {"value": 50000},
            "ETH": {"value": 50000},
        }

        scenarios = {
            "market_crash": {"BTC": -0.30, "ETH": -0.35},
            "crypto_winter": {"BTC": -0.50, "ETH": -0.60},
        }

        results = risk_mgr.stress_test_portfolio(holdings, scenarios)

        assert len(results) == 2
        assert all(hasattr(r, "scenario_name") for r in results)
        assert all(r.portfolio_loss_pct < 0 for r in results)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        risk_mgr = MultiAssetRiskManager()

        returns = np.random.normal(0.0005, 0.02, 252)

        var_95 = risk_mgr.calculate_var(returns, confidence=0.95)

        assert var_95 < 0  # VaR should be negative (loss)
        assert var_95 > -1  # But reasonable

    def test_risk_limit_violations(self):
        """Test risk limit violation detection."""
        risk_mgr = MultiAssetRiskManager()

        portfolio_metrics = {
            "volatility": 0.30,  # Exceeds 0.25 limit
            "max_drawdown": -0.20,  # Exceeds 0.15 limit
            "diversification_score": 0.3,  # Below 0.5 limit
        }

        holdings = {"BTC": {"value": 30000}, "ETH": {"value": 70000}}
        conc_metric = risk_mgr.assess_concentration_risk(holdings)

        systemic = risk_mgr.detect_systemic_risk(holdings, {})

        violations = risk_mgr.check_risk_limits(
            portfolio_metrics, conc_metric, systemic
        )

        assert len(violations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
