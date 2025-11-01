"""
Performance Monitor

Real-time performance tracking and analytics:
- Portfolio metrics
- Trade analysis
- Daily returns
- Risk metrics
- Performance attribution
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DailyReturn:
    """Daily return record."""
    date: str
    portfolio_value: float
    daily_return: float
    daily_return_pct: float
    cumulative_return: float


@dataclass
class TradeAnalysis:
    """Analysis of executed trades."""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_days: int
    profit_factor: float = 1.0
    risk_reward_ratio: float = 1.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    start_date: str
    current_date: str
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    information_ratio: float
    volatility: float
    beta: float = 0


class PerformanceMonitor:
    """
    Monitors and analyzes trading performance.
    """

    def __init__(self, benchmark: Optional[List[float]] = None):
        """
        Initialize performance monitor.

        Args:
            benchmark: Benchmark return series for comparison
        """
        self.daily_returns: List[DailyReturn] = []
        self.portfolio_values: List[float] = []
        self.trades: List[TradeAnalysis] = []
        self.benchmark = benchmark or []
        self.start_date: Optional[str] = None
        self.start_capital: float = 0

        logger.info("PerformanceMonitor initialized")

    def record_daily_performance(
        self,
        portfolio_value: float,
        previous_value: Optional[float] = None,
        date: Optional[str] = None
    ) -> None:
        """
        Record daily portfolio performance.

        Args:
            portfolio_value: Current portfolio value
            previous_value: Previous portfolio value
            date: Date of record
        """
        if date is None:
            date = datetime.now().isoformat().split("T")[0]

        if self.start_date is None:
            self.start_date = date

        # Calculate daily return
        if previous_value is None:
            if not self.portfolio_values:
                previous_value = self.start_capital or portfolio_value
            else:
                previous_value = self.portfolio_values[-1]

        daily_return = portfolio_value - previous_value
        daily_return_pct = daily_return / previous_value if previous_value > 0 else 0

        # Calculate cumulative return
        if self.start_capital == 0:
            self.start_capital = previous_value if previous_value > 0 else portfolio_value

        cumulative_return = (portfolio_value - self.start_capital) / self.start_capital

        record = DailyReturn(
            date=date,
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            daily_return_pct=daily_return_pct,
            cumulative_return=cumulative_return,
        )

        self.daily_returns.append(record)
        self.portfolio_values.append(portfolio_value)

        logger.debug(
            f"Recorded performance: {date} "
            f"Value: ${portfolio_value:.2f} "
            f"Daily: {daily_return_pct:.2%}"
        )

    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        duration_days: int = 1
    ) -> None:
        """
        Record completed trade.

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            pnl: Profit/loss
            duration_days: Trade duration
        """
        pnl_pct = pnl / (entry_price * quantity) if entry_price > 0 else 0

        trade = TradeAnalysis(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_days=duration_days,
        )

        self.trades.append(trade)

        logger.info(
            f"Recorded trade: {symbol} P&L: ${pnl:.2f} ({pnl_pct:.2%})"
        )

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            PerformanceMetrics object
        """
        if not self.daily_returns or not self.portfolio_values:
            return PerformanceMetrics(
                start_date=self.start_date or datetime.now().isoformat(),
                current_date=datetime.now().isoformat(),
                total_return_pct=0,
                annualized_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                calmar_ratio=0,
                win_rate=0,
                profit_factor=0,
                information_ratio=0,
                volatility=0,
            )

        # Calculate returns
        returns = np.array([r.daily_return_pct for r in self.daily_returns])
        total_return = (self.portfolio_values[-1] - self.start_capital) / self.start_capital

        # Annualized metrics
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if self.trades:
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = len([t for t in self.trades if t.pnl < 0])
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0)
        else:
            win_rate = 0
            profit_factor = 0

        # Information ratio (vs benchmark)
        information_ratio = 0
        if self.benchmark and len(self.benchmark) == len(returns):
            active_returns = returns - np.array(self.benchmark)
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = np.mean(active_returns) * 252 / tracking_error if tracking_error > 0 else 0

        return PerformanceMetrics(
            start_date=self.start_date or datetime.now().isoformat(),
            current_date=datetime.now().isoformat(),
            total_return_pct=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            information_ratio=information_ratio,
            volatility=volatility,
        )

    def get_equity_curve(self) -> List[float]:
        """Get portfolio equity curve."""
        return self.portfolio_values.copy()

    def get_daily_returns_array(self) -> np.ndarray:
        """Get daily returns as numpy array."""
        return np.array([r.daily_return_pct for r in self.daily_returns])

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        metrics = self.calculate_metrics()

        return {
            "summary": {
                "start_date": metrics.start_date,
                "current_date": metrics.current_date,
                "trading_period_days": len(self.daily_returns),
            },
            "returns": {
                "total_return_pct": metrics.total_return_pct,
                "annualized_return": metrics.annualized_return,
                "daily_avg_return": np.mean(self.get_daily_returns_array()) if self.daily_returns else 0,
            },
            "risk": {
                "volatility": metrics.volatility,
                "max_drawdown": metrics.max_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
            },
            "trades": {
                "total_trades": len(self.trades),
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_pnl": np.mean([t.pnl for t in self.trades]) if self.trades else 0,
            },
            "comparison": {
                "information_ratio": metrics.information_ratio,
            },
        }

    def generate_report(self) -> str:
        """Generate performance report."""
        metrics = self.calculate_metrics()
        summary = self.get_performance_summary()

        report = f"""
TRADING PERFORMANCE REPORT
{'=' * 60}

Period: {metrics.start_date} to {metrics.current_date}
Trading Days: {len(self.daily_returns)}

RETURNS
{'-' * 60}
Total Return:          {summary['returns']['total_return_pct']:>10.2%}
Annualized Return:     {summary['returns']['annualized_return']:>10.2%}
Daily Avg Return:      {summary['returns']['daily_avg_return']:>10.4%}

RISK METRICS
{'-' * 60}
Volatility:            {summary['risk']['volatility']:>10.2%}
Max Drawdown:          {summary['risk']['max_drawdown']:>10.2%}
Sharpe Ratio:          {summary['risk']['sharpe_ratio']:>10.2f}
Sortino Ratio:         {summary['risk']['sortino_ratio']:>10.2f}
Calmar Ratio:          {summary['risk']['calmar_ratio']:>10.2f}

TRADES
{'-' * 60}
Total Trades:          {summary['trades']['total_trades']:>10d}
Win Rate:              {summary['trades']['win_rate']:>10.2%}
Profit Factor:         {summary['trades']['profit_factor']:>10.2f}
Avg P&L per Trade:     {summary['trades']['avg_pnl']:>10.2f}

{'=' * 60}
"""
        return report
