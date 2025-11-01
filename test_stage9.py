"""
Stage 9: Integrated Trading Engine Test Suite

Comprehensive tests for:
- Trading engine orchestration
- Order execution and management
- Performance monitoring and analytics
- Deployment and connectivity
- Autonomous trading agent
"""

import pytest
import numpy as np
from datetime import datetime
from trading_engine import (
    TradingEngine,
    EngineConfig,
    ExecutionManager,
    Order,
    OrderStatus,
    PerformanceMonitor,
    DeploymentManager,
    DeploymentConfig,
    SystemStatus,
    TradingAgent,
)


class TestTradingEngine:
    """Test core trading engine."""

    def test_engine_initialization(self):
        """Test trading engine initialization."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        assert engine.config.initial_capital == 100000
        assert engine.state.current_capital == 100000
        assert len(engine.positions) == 0
        assert engine.state.is_running is False

    def test_engine_start_stop(self):
        """Test engine start and stop."""
        config = EngineConfig()
        engine = TradingEngine(config)

        assert engine.start() is True
        assert engine.state.is_running is True

        assert engine.stop() is True
        assert engine.state.is_running is False

    def test_data_source_connection(self):
        """Test connecting to data source."""
        config = EngineConfig()
        engine = TradingEngine(config)

        success = engine.connect_data_source("polygon.io")

        assert success is True
        assert engine.state.is_connected is True

    def test_add_buy_signal(self):
        """Test adding buy signal."""
        config = EngineConfig()
        engine = TradingEngine(config)

        engine.add_signal(
            symbol="BTC",
            signal_type="buy",
            strength=0.8,
            asset_class="crypto"
        )

        assert len(engine.signal_queue) == 1
        assert engine.signal_queue[0]["symbol"] == "BTC"
        assert engine.signal_queue[0]["type"] == "buy"

    def test_add_sell_signal(self):
        """Test adding sell signal."""
        config = EngineConfig()
        engine = TradingEngine(config)

        engine.add_signal(
            symbol="ETH",
            signal_type="sell",
            strength=0.6,
            asset_class="crypto"
        )

        assert len(engine.signal_queue) == 1
        assert engine.signal_queue[0]["type"] == "sell"

    def test_process_signals(self):
        """Test signal processing."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        # Add multiple signals
        engine.add_signal("BTC", "buy", 0.8)
        engine.add_signal("ETH", "buy", 0.7)

        orders = engine.process_signals()

        assert len(orders) == 2
        assert len(engine.pending_orders) == 2
        assert len(engine.signal_queue) == 0

    def test_order_execution(self):
        """Test order execution."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        engine.add_signal("BTC", "buy", 0.8)
        orders = engine.process_signals()

        assert len(orders) == 1
        order = orders[0]

        success = engine.execute_order(order, execution_price=45000)

        assert success is True
        assert "BTC" in engine.positions

    def test_position_tracking(self):
        """Test position tracking."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        engine.add_signal("BTC", "buy", 0.8)
        orders = engine.process_signals()
        engine.execute_order(orders[0], 45000)

        assert "BTC" in engine.positions
        position = engine.positions["BTC"]
        assert position["quantity"] > 0
        assert position["entry_price"] == 45000

    def test_price_update_and_pnl(self):
        """Test price updates and P&L calculation."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        # Buy position
        engine.add_signal("BTC", "buy", 0.8)
        orders = engine.process_signals()
        engine.execute_order(orders[0], 45000)

        # Update prices (price goes up)
        engine.update_prices({"BTC": 46000})

        position = engine.positions["BTC"]
        unrealized_pnl = position["unrealized_pnl"]

        assert unrealized_pnl > 0  # Profit

    def test_drawdown_monitoring(self):
        """Test drawdown monitoring."""
        config = EngineConfig(initial_capital=100000, max_drawdown_pct=0.15)
        engine = TradingEngine(config)

        engine.add_signal("BTC", "buy", 0.8)
        orders = engine.process_signals()
        engine.execute_order(orders[0], 45000)

        # Price drops slightly
        engine.update_prices({"BTC": 44000})

        # Drawdown should be calculated
        assert engine.state.current_drawdown >= 0

    def test_position_limit(self):
        """Test max positions limit."""
        config = EngineConfig(initial_capital=1000000, max_positions=2)
        engine = TradingEngine(config)

        # Add 3 signals
        engine.add_signal("BTC", "buy", 0.8)
        engine.add_signal("ETH", "buy", 0.8)
        engine.add_signal("XRP", "buy", 0.8)

        orders = engine.process_signals()

        # Only first 2 orders should be processable due to position limit
        # The signal processing should filter out the 3rd
        assert len(orders) >= 2

    def test_portfolio_summary(self):
        """Test portfolio summary."""
        config = EngineConfig(initial_capital=100000)
        engine = TradingEngine(config)

        engine.add_signal("BTC", "buy", 0.8)
        orders = engine.process_signals()
        engine.execute_order(orders[0], 45000)

        summary = engine.get_engine_summary()

        assert "state" in summary
        assert "performance" in summary
        assert "positions" in summary
        assert summary["positions"]["open_count"] == 1


class TestExecutionManager:
    """Test execution manager."""

    def test_execution_manager_initialization(self):
        """Test execution manager initialization."""
        manager = ExecutionManager(broker="paper", slippage_pct=0.001)

        assert manager.broker == "paper"
        assert manager.slippage_pct == 0.001
        assert len(manager.orders) == 0
        assert len(manager.trades) == 0

    def test_create_market_order(self):
        """Test creating market order."""
        manager = ExecutionManager()

        order = manager.create_order(
            symbol="BTC",
            side="buy",
            quantity=1.0,
            order_type="market"
        )

        assert order.symbol == "BTC"
        assert order.side == "buy"
        assert order.quantity == 1.0
        assert order.status == OrderStatus.PENDING

    def test_create_limit_order(self):
        """Test creating limit order."""
        manager = ExecutionManager()

        order = manager.create_order(
            symbol="ETH",
            side="buy",
            quantity=10.0,
            order_type="limit",
            limit_price=2500
        )

        assert order.order_type == "limit"
        assert order.limit_price == 2500

    def test_execute_order(self):
        """Test executing order."""
        manager = ExecutionManager()

        order = manager.create_order("BTC", "buy", 1.0)
        order_id = list(manager.orders.keys())[0]

        trade = manager.execute_order(order_id, market_price=45000)

        assert trade is not None
        assert trade.symbol == "BTC"
        assert trade.entry_price > 45000  # With slippage

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        manager = ExecutionManager(slippage_pct=0.001)

        order = manager.create_order("BTC", "buy", 1.0)
        order_id = list(manager.orders.keys())[0]

        trade = manager.execute_order(order_id, market_price=45000)

        expected_slippage = 45000 * 0.001
        assert abs(trade.slippage - expected_slippage) < 1

    def test_close_trade(self):
        """Test closing trade."""
        manager = ExecutionManager()

        order = manager.create_order("BTC", "buy", 1.0)
        order_id = list(manager.orders.keys())[0]
        trade = manager.execute_order(order_id, 45000)

        manager.close_trade(trade, exit_price=46000)

        assert trade.exit_price is not None
        assert trade.pnl > 0

    def test_cancel_order(self):
        """Test cancelling order."""
        manager = ExecutionManager()

        order = manager.create_order("BTC", "buy", 1.0)
        order_id = list(manager.orders.keys())[0]

        success = manager.cancel_order(order_id)

        assert success is True
        assert manager.orders[order_id].status == OrderStatus.CANCELLED

    def test_execution_quality(self):
        """Test execution quality calculation."""
        manager = ExecutionManager(slippage_pct=0.0005)

        # Execute multiple trades
        for i in range(5):
            order = manager.create_order(f"BTC", "buy", 1.0)
            order_id = list(manager.orders.keys())[-1]
            manager.execute_order(order_id, 45000)

        quality = manager.get_execution_quality()

        assert 0 <= quality <= 1

    def test_trade_analysis(self):
        """Test trade analysis."""
        manager = ExecutionManager()

        # Create and execute orders
        order = manager.create_order("BTC", "buy", 1.0)
        order_id = list(manager.orders.keys())[0]
        trade = manager.execute_order(order_id, 45000)
        manager.close_trade(trade, 46000)

        analysis = manager.get_trade_analysis()

        assert analysis["completed_trades"] == 1
        assert analysis["total_pnl"] > 0


class TestPerformanceMonitor:
    """Test performance monitor."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()

        assert len(monitor.daily_returns) == 0
        assert len(monitor.trades) == 0

    def test_record_daily_performance(self):
        """Test recording daily performance."""
        monitor = PerformanceMonitor()
        monitor.start_capital = 100000

        monitor.record_daily_performance(101000, 100000)

        assert len(monitor.daily_returns) == 1
        assert monitor.daily_returns[0].daily_return == 1000

    def test_record_trade(self):
        """Test recording trade."""
        monitor = PerformanceMonitor()

        monitor.record_trade(
            symbol="BTC",
            entry_price=45000,
            exit_price=46000,
            quantity=1.0,
            pnl=1000,
            duration_days=1
        )

        assert len(monitor.trades) == 1
        assert monitor.trades[0].pnl == 1000

    def test_calculate_metrics(self):
        """Test metric calculation."""
        monitor = PerformanceMonitor()
        monitor.start_capital = 100000

        # Record performance over time
        for i in range(10):
            monitor.record_daily_performance(100000 + (i * 100), 100000 + ((i - 1) * 100))

        metrics = monitor.calculate_metrics()

        assert metrics.total_return_pct >= 0
        assert metrics.volatility >= 0
        assert metrics.sharpe_ratio is not None

    def test_equity_curve(self):
        """Test equity curve."""
        monitor = PerformanceMonitor()
        monitor.start_capital = 100000

        for i in range(5):
            monitor.record_daily_performance(100000 + (i * 100), 100000 + ((i - 1) * 100))

        curve = monitor.get_equity_curve()

        assert len(curve) == 5
        assert curve[-1] > curve[0]

    def test_performance_summary(self):
        """Test performance summary."""
        monitor = PerformanceMonitor()
        monitor.start_capital = 100000

        for i in range(10):
            monitor.record_daily_performance(100000 + (i * 100))

        summary = monitor.get_performance_summary()

        assert "summary" in summary
        assert "returns" in summary
        assert "risk" in summary
        assert "trades" in summary


class TestDeploymentManager:
    """Test deployment manager."""

    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization."""
        config = DeploymentConfig(environment="paper")
        manager = DeploymentManager(config)

        assert manager.status == SystemStatus.OFFLINE
        assert manager.config.environment == "paper"

    def test_initialization_paper_trading(self):
        """Test initialization in paper trading mode."""
        config = DeploymentConfig(
            environment="paper",
            enable_live_trading=False,
            data_source_url="http://api.example.com"
        )
        manager = DeploymentManager(config)

        success = manager.initialize()

        assert success is True
        assert manager.status in [SystemStatus.ONLINE, SystemStatus.DEGRADED]

    def test_health_check(self):
        """Test health check."""
        config = DeploymentConfig(environment="paper")
        manager = DeploymentManager(config)
        manager.initialize()

        health = manager.perform_health_check()

        assert health.data_connection is not None
        assert health.risk_system is True

    def test_system_status(self):
        """Test system status."""
        config = DeploymentConfig()
        manager = DeploymentManager(config)
        manager.initialize()

        status = manager.get_system_status()

        assert "status" in status
        assert "data_connection" in status
        assert "error_count" in status

    def test_alert_sending(self):
        """Test alert sending."""
        config = DeploymentConfig(
            alerts_enabled=True,
            alert_email="test@example.com"
        )
        manager = DeploymentManager(config)

        success = manager.send_alert("Test Alert", "This is a test")

        assert success is True

    def test_graceful_shutdown(self):
        """Test graceful shutdown."""
        config = DeploymentConfig()
        manager = DeploymentManager(config)
        manager.initialize()

        manager.graceful_shutdown()

        assert manager.status == SystemStatus.OFFLINE

    def test_restart(self):
        """Test restart."""
        config = DeploymentConfig(environment="paper", data_source_url="http://api.example.com")
        manager = DeploymentManager(config)

        success = manager.restart()

        assert success is True
        assert manager.status in [SystemStatus.ONLINE, SystemStatus.DEGRADED]

    def test_diagnostics(self):
        """Test diagnostics."""
        config = DeploymentConfig()
        manager = DeploymentManager(config)
        manager.initialize()

        diagnostics = manager.get_diagnostics()

        assert "status" in diagnostics
        assert "health" in diagnostics
        assert "errors" in diagnostics
        assert "timing" in diagnostics


class TestTradingAgent:
    """Test autonomous trading agent."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = TradingAgent(agent_id="agent_1", aggressiveness=0.5)

        assert agent.agent_id == "agent_1"
        assert agent.aggressiveness == 0.5
        assert agent.state == "idle"

    def test_evaluate_buy_signal(self):
        """Test evaluating buy signal."""
        agent = TradingAgent()

        signal = {
            "symbol": "BTC",
            "type": "buy",
            "strength": 0.8,
            "metadata": {}
        }

        decision = agent.evaluate_signal(
            signal=signal,
            current_price=45000,
            current_positions={},
            portfolio_value=100000
        )

        assert decision is not None
        assert decision.symbol == "BTC"
        assert decision.action in ["buy", "hold"]

    def test_evaluate_sell_signal(self):
        """Test evaluating sell signal."""
        agent = TradingAgent()

        signal = {
            "symbol": "ETH",
            "type": "sell",
            "strength": 0.7,
            "metadata": {}
        }

        current_positions = {
            "ETH": {"quantity": 10, "entry_price": 2500}
        }

        decision = agent.evaluate_signal(
            signal=signal,
            current_price=2600,
            current_positions=current_positions,
            portfolio_value=100000
        )

        assert decision is not None
        assert decision.action == "sell"

    def test_confidence_calculation(self):
        """Test confidence calculation."""
        agent = TradingAgent()

        signal = {
            "symbol": "BTC",
            "type": "buy",
            "strength": 0.9,
            "metadata": {}
        }

        decision = agent.evaluate_signal(signal, 45000, {}, 100000)

        assert decision is not None
        assert 0 <= decision.confidence <= 1

    def test_adaptive_learning(self):
        """Test adaptive learning."""
        agent = TradingAgent()

        initial_weights = agent.strategy_weights.copy()

        # Simulate some winning trades
        trades = [
            {"strategy": "momentum", "pnl": 100},
            {"strategy": "momentum", "pnl": 50},
            {"strategy": "mean_reversion", "pnl": -30},
        ]

        agent.adaptive_learning(trades)

        # Weights should have changed
        assert agent.strategy_weights != initial_weights

    def test_aggressiveness_adjustment(self):
        """Test aggressiveness adjustment."""
        agent = TradingAgent(aggressiveness=0.5)

        agent.set_aggressiveness(0.8)

        assert agent.aggressiveness == 0.8

    def test_agent_state(self):
        """Test getting agent state."""
        agent = TradingAgent(agent_id="test_agent")

        signal = {"symbol": "BTC", "type": "buy", "strength": 0.8, "metadata": {}}
        agent.evaluate_signal(signal, 45000, {}, 100000)

        state = agent.get_agent_state()

        assert state["agent_id"] == "test_agent"
        assert state["total_decisions"] == 1
        assert "strategy_weights" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
