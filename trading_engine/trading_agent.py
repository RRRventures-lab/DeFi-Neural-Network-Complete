"""
Automated Trading Agent

Autonomous trading decision maker:
- Signal evaluation and decision making
- Risk-aware position sizing
- Adaptive strategy adjustment
- Performance-based learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Trading agent state."""
    IDLE = "idle"
    EVALUATING = "evaluating"
    DECIDING = "deciding"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    ERROR = "error"


@dataclass
class AgentDecision:
    """Trading decision from agent."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold', 'close'
    confidence: float  # 0-1
    size: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_score: float = 0.5


class TradingAgent:
    """
    Autonomous trading agent that makes intelligent trading decisions.
    """

    def __init__(self, agent_id: str = "default", aggressiveness: float = 0.5):
        """
        Initialize trading agent.

        Args:
            agent_id: Unique agent identifier
            aggressiveness: Agent aggressiveness (0-1, 0=conservative, 1=aggressive)
        """
        self.agent_id = agent_id
        self.aggressiveness = aggressiveness
        self.state = AgentState.IDLE
        self.decisions: List[AgentDecision] = []
        self.performance_history: List[float] = []
        self.strategy_weights: Dict[str, float] = {
            "momentum": 0.25,
            "mean_reversion": 0.25,
            "volatility": 0.25,
            "sentiment": 0.25,
        }

        logger.info(f"TradingAgent {agent_id} initialized (aggressiveness: {aggressiveness})")

    def evaluate_signal(
        self,
        signal: Dict,
        current_price: float,
        current_positions: Dict,
        portfolio_value: float,
    ) -> Optional[AgentDecision]:
        """
        Evaluate signal and make trading decision.

        Args:
            signal: Trading signal with type, strength, metadata
            current_price: Current asset price
            current_positions: Current open positions
            portfolio_value: Current portfolio value

        Returns:
            AgentDecision or None
        """
        self.state = AgentState.EVALUATING

        symbol = signal.get("symbol")
        signal_type = signal.get("type")
        strength = signal.get("strength", 0.5)

        if not symbol:
            logger.warning("Signal missing symbol")
            return None

        # Evaluate signal components
        confidence = self._calculate_confidence(signal)
        risk_score = self._calculate_risk_score(signal, current_price, current_positions)

        # Make decision
        if signal_type == "buy":
            decision = self._make_buy_decision(
                symbol, strength, confidence, risk_score, current_positions, portfolio_value
            )
        elif signal_type == "sell":
            decision = self._make_sell_decision(symbol, strength, confidence, risk_score, current_positions)
        else:
            decision = AgentDecision(
                symbol=symbol,
                action="hold",
                confidence=confidence,
                size=0,
                reason="No clear signal direction",
                risk_score=risk_score,
            )

        if decision:
            self.decisions.append(decision)
            logger.info(f"Agent decision: {decision.action} {symbol} (confidence: {decision.confidence:.2%})")

        self.state = AgentState.IDLE

        return decision

    def _calculate_confidence(self, signal: Dict) -> float:
        """Calculate decision confidence from signal."""
        strength = signal.get("strength", 0.5)
        metadata = signal.get("metadata", {})

        # Base confidence from strength
        confidence = strength

        # Adjust for agreement across strategies
        if "momentum" in metadata and "mean_reversion" in metadata:
            if metadata["momentum"] == metadata["mean_reversion"]:
                confidence += 0.1  # Agreement boost
            else:
                confidence -= 0.1  # Disagreement penalty

        # Cap confidence at 0-1
        return max(0, min(1, confidence))

    def _calculate_risk_score(
        self, signal: Dict, current_price: float, current_positions: Dict
    ) -> float:
        """Calculate risk score for decision."""
        # Base risk from volatility
        volatility = signal.get("metadata", {}).get("volatility", 0.2)
        risk = volatility / 0.5  # Normalize to 0.2 volatility = 0.5 risk

        # Adjust for position concentration
        position_count = len(current_positions)
        if position_count > 10:
            risk += 0.2

        # Adjust for aggressiveness
        risk *= (1 - self.aggressiveness)

        return max(0, min(1, risk))

    def _make_buy_decision(
        self,
        symbol: str,
        strength: float,
        confidence: float,
        risk_score: float,
        current_positions: Dict,
        portfolio_value: float,
    ) -> AgentDecision:
        """Make buy decision."""
        # Check if already holding
        if symbol in current_positions:
            return AgentDecision(
                symbol=symbol,
                action="hold",
                confidence=confidence,
                size=0,
                reason=f"Already holding {symbol}",
                risk_score=risk_score,
            )

        # Check confidence threshold
        min_confidence = 0.5 - (self.aggressiveness * 0.2)  # More aggressive = lower threshold
        if confidence < min_confidence:
            return AgentDecision(
                symbol=symbol,
                action="hold",
                confidence=confidence,
                size=0,
                reason=f"Confidence {confidence:.2%} below threshold",
                risk_score=risk_score,
            )

        # Size position based on confidence and risk
        size_multiplier = confidence * (1 - risk_score) * (1 + self.aggressiveness * 0.5)
        position_size = portfolio_value * 0.02 * size_multiplier  # Base 2% position

        return AgentDecision(
            symbol=symbol,
            action="buy",
            confidence=confidence,
            size=position_size,
            reason=f"Buy signal (strength: {strength:.2%}, confidence: {confidence:.2%})",
            risk_score=risk_score,
        )

    def _make_sell_decision(
        self,
        symbol: str,
        strength: float,
        confidence: float,
        risk_score: float,
        current_positions: Dict,
    ) -> AgentDecision:
        """Make sell decision."""
        # Check if holding
        if symbol not in current_positions:
            return AgentDecision(
                symbol=symbol,
                action="hold",
                confidence=confidence,
                size=0,
                reason=f"Not holding {symbol}",
                risk_score=risk_score,
            )

        position = current_positions[symbol]

        return AgentDecision(
            symbol=symbol,
            action="sell",
            confidence=confidence,
            size=position.get("quantity", 0),
            reason=f"Sell signal (strength: {strength:.2%})",
            risk_score=risk_score,
        )

    def adaptive_learning(self, recent_trades: List[Dict]) -> None:
        """
        Adapt strategy weights based on recent performance.

        Args:
            recent_trades: List of recent executed trades
        """
        if not recent_trades:
            return

        # Calculate win rate by strategy
        wins_by_strategy = {
            "momentum": [],
            "mean_reversion": [],
            "volatility": [],
            "sentiment": [],
        }

        for trade in recent_trades:
            strategy = trade.get("strategy", "momentum")
            pnl = trade.get("pnl", 0)

            if strategy in wins_by_strategy:
                wins_by_strategy[strategy].append(pnl > 0)

        # Update weights based on success rate
        for strategy, results in wins_by_strategy.items():
            if results:
                win_rate = sum(results) / len(results)
                # Update weight (reward successful strategies)
                self.strategy_weights[strategy] *= (1 + (win_rate - 0.5) * 0.1)

        # Normalize weights
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}

        logger.info(f"Updated strategy weights: {self.strategy_weights}")

    def get_agent_state(self) -> Dict:
        """Get current agent state."""
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "aggressiveness": self.aggressiveness,
            "total_decisions": len(self.decisions),
            "recent_decisions": [
                {
                    "symbol": d.symbol,
                    "action": d.action,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                }
                for d in self.decisions[-5:]
            ],
            "strategy_weights": self.strategy_weights,
        }

    def reset_decisions(self) -> None:
        """Reset decision history."""
        self.decisions.clear()
        logger.info("Decision history cleared")

    def set_aggressiveness(self, level: float) -> None:
        """
        Adjust agent aggressiveness.

        Args:
            level: Aggressiveness level (0-1)
        """
        self.aggressiveness = max(0, min(1, level))
        logger.info(f"Aggressiveness adjusted to {self.aggressiveness}")
