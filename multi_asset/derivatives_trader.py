"""
Derivatives Trader Module

Implements derivatives trading (futures, options):
- Futures contracts
- Options chains
- Greeks for positions
- Settlement handling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FuturesContract:
    """Futures contract specification."""
    symbol: str
    expiration_date: str
    contract_size: float
    tick_size: float
    tick_value: float
    margin_requirement: float
    bid_price: float
    ask_price: float

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2


@dataclass
class FuturesPosition:
    """Futures position tracking."""
    symbol: str
    quantity: float  # Positive = long, negative = short
    entry_price: float
    current_price: float
    entry_date: str

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        return (self.current_price - self.entry_price) * self.quantity * 100

    @property
    def contract_value(self) -> float:
        """Current contract value."""
        return self.current_price * self.quantity * 100


class DerivativesTrader:
    """
    Derivatives trader (futures and options).
    """

    def __init__(self, account_balance: float = 50000):
        """
        Initialize derivatives trader.

        Args:
            account_balance: Starting account balance
        """
        self.account_balance = account_balance
        self.margin_available = account_balance
        self.futures_positions: Dict[str, FuturesPosition] = {}
        self.option_positions: List[Dict] = []
        self.contracts: Dict[str, FuturesContract] = {}

        logger.info(f"DerivativesTrader initialized: ${account_balance}")

    def add_contract(self, contract: FuturesContract) -> None:
        """Add futures contract specification."""
        self.contracts[contract.symbol] = contract
        logger.debug(f"Added contract: {contract.symbol}")

    def open_futures_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        leverage: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Open futures position.

        Args:
            symbol: Futures symbol
            quantity: Number of contracts
            entry_price: Entry price
            leverage: Leverage to use

        Returns:
            (success, position_id)
        """
        if symbol not in self.contracts:
            return False, "contract_not_found"

        contract = self.contracts[symbol]
        margin_required = (contract.margin_requirement * abs(quantity) * 100) / leverage

        if margin_required > self.margin_available:
            return False, "insufficient_margin"

        position = FuturesPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_date=datetime.now().isoformat()
        )

        self.futures_positions[symbol] = position
        self.margin_available -= margin_required

        logger.info(f"Opened {symbol} futures: {quantity} contracts @ {entry_price}")

        return True, f"pos_{symbol}"

    def close_futures_position(self, symbol: str, exit_price: float) -> Tuple[bool, float]:
        """
        Close futures position.

        Args:
            symbol: Futures symbol
            exit_price: Exit price

        Returns:
            (success, pnl)
        """
        if symbol not in self.futures_positions:
            return False, 0

        position = self.futures_positions[symbol]
        contract = self.contracts.get(symbol)

        if not contract:
            return False, 0

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity * 100

        # Release margin
        margin_used = (contract.margin_requirement * abs(position.quantity) * 100)
        self.margin_available += margin_used
        self.account_balance += pnl

        del self.futures_positions[symbol]

        logger.info(f"Closed {symbol} futures: P&L = ${pnl:.2f}")

        return True, pnl

    def update_futures_prices(self, symbol: str, bid: float, ask: float) -> None:
        """Update futures prices."""
        if symbol in self.contracts:
            self.contracts[symbol].bid_price = bid
            self.contracts[symbol].ask_price = ask

        if symbol in self.futures_positions:
            self.futures_positions[symbol].current_price = (bid + ask) / 2

    def add_option_position(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration: str,
        premium: float,
        contracts: int,
        side: str
    ) -> None:
        """Add option position."""
        position = {
            'symbol': symbol,
            'type': option_type,  # 'call' or 'put'
            'strike': strike,
            'expiration': expiration,
            'premium': premium,
            'contracts': contracts,
            'side': side,  # 'buy' or 'sell'
            'entry_date': datetime.now().isoformat(),
            'current_price': premium
        }

        self.option_positions.append(position)
        logger.info(f"Added {side} {option_type} option: {symbol} {strike} strike")

    def calculate_option_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """Calculate Greeks for option."""
        from scipy.stats import norm

        # Black-Scholes Greeks
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
            volatility * np.sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
        theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                 risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def get_portfolio_greeks(self) -> Dict:
        """Calculate portfolio Greeks."""
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0

        for position in self.option_positions:
            greeks = self.calculate_option_greeks(
                spot=100,  # Simplified
                strike=position['strike'],
                time_to_expiry=0.25,
                volatility=0.2
            )

            multiplier = position['contracts'] * 100
            if position['side'] == 'sell':
                multiplier *= -1

            total_delta += greeks['delta'] * multiplier
            total_gamma += greeks['gamma'] * multiplier
            total_theta += greeks['theta'] * multiplier
            total_vega += greeks['vega'] * multiplier

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }

    def get_margin_usage(self) -> Dict:
        """Get margin usage details."""
        futures_margin = sum(
            self.contracts[symbol].margin_requirement * abs(pos.quantity) * 100
            for symbol, pos in self.futures_positions.items()
            if symbol in self.contracts
        )

        return {
            'total_margin_available': self.margin_available,
            'futures_margin_used': futures_margin,
            'account_balance': self.account_balance,
            'margin_level': (self.account_balance / futures_margin) * 100 if futures_margin > 0 else float('inf')
        }

    def get_position_summary(self) -> Dict:
        """Get summary of all derivative positions."""
        total_pnl = 0
        for pos in self.futures_positions.values():
            total_pnl += pos.unrealized_pnl

        return {
            'futures_positions': len(self.futures_positions),
            'option_positions': len(self.option_positions),
            'unrealized_pnl': total_pnl,
            'account_balance': self.account_balance
        }
