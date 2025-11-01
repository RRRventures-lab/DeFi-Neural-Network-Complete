"""
Options Pricer Module

Implements option pricing and Greeks calculation:
- Black-Scholes pricing
- Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility
- Option strategies
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptionPrice:
    """Option pricing result."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_type: str  # 'call' or 'put'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'price': self.price,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'option_type': self.option_type
        }


@dataclass
class OptionStrategy:
    """Option strategy specification."""
    name: str
    components: list  # List of {option, position_type}
    net_cost: float
    max_profit: float
    max_loss: float
    break_even: list


class BlackScholesCalculator:
    """Black-Scholes option pricing calculator."""

    @staticmethod
    def price_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> OptionPrice:
        """
        Price a European call option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            OptionPrice with Greeks
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

        return OptionPrice(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            option_type='call'
        )

    @staticmethod
    def price_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> OptionPrice:
        """
        Price a European put option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            OptionPrice with Greeks
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return OptionPrice(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            option_type='put'
        )

    @staticmethod
    def implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        tolerance: float = 1e-4,
        max_iterations: int = 100
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson.

        Args:
            price: Current option price
            S: Spot price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: 'call' or 'put'
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            Implied volatility or None if not converged
        """
        sigma = 0.3  # Initial guess

        for iteration in range(max_iterations):
            if option_type == 'call':
                option = BlackScholesCalculator.price_call(S, K, T, r, sigma)
            else:
                option = BlackScholesCalculator.price_put(S, K, T, r, sigma)

            diff = option.price - price
            if abs(diff) < tolerance:
                return sigma

            # Newton-Raphson update: vega is in cents (divide by 100)
            vega_annual = option.vega / 100
            if vega_annual > 1e-6:
                sigma = sigma - diff / vega_annual
            else:
                sigma = sigma + 0.001

            if sigma <= 0.001:
                sigma = 0.001
            elif sigma >= 5.0:
                sigma = 5.0

        # Return last estimate even if not fully converged
        return sigma


class OptionsPricer:
    """
    Options pricing system.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize options pricer.

        Args:
            risk_free_rate: Risk-free rate (annual)
        """
        self.risk_free_rate = risk_free_rate
        self.calculator = BlackScholesCalculator()
        logger.info(f"Options pricer initialized (r={risk_free_rate:.1%})")

    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> OptionPrice:
        """
        Price an option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            OptionPrice
        """
        if option_type == 'call':
            return self.calculator.price_call(S, K, T, self.risk_free_rate, sigma)
        else:
            return self.calculator.price_put(S, K, T, self.risk_free_rate, sigma)

    def price_call(self, S: float, K: float, T: float, sigma: float) -> OptionPrice:
        """Price a call option."""
        return self.calculator.price_call(S, K, T, self.risk_free_rate, sigma)

    def price_put(self, S: float, K: float, T: float, sigma: float) -> OptionPrice:
        """Price a put option."""
        return self.calculator.price_put(S, K, T, self.risk_free_rate, sigma)

    def calculate_implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        option_type: str = 'call'
    ) -> Optional[float]:
        """
        Calculate implied volatility.

        Args:
            price: Market price of option
            S: Spot price
            K: Strike price
            T: Time to expiration
            option_type: 'call' or 'put'

        Returns:
            Implied volatility
        """
        return self.calculator.implied_volatility(
            price, S, K, T, self.risk_free_rate, option_type
        )

    def protective_put(
        self,
        S: float,
        K_put: float,
        T: float,
        sigma: float
    ) -> Dict:
        """
        Analyze protective put strategy.

        Args:
            S: Spot price
            K_put: Put strike
            T: Time to expiration
            sigma: Volatility

        Returns:
            Strategy analysis
        """
        put = self.price_put(S, K_put, T, sigma)

        return {
            'strategy': 'protective_put',
            'spot_price': S,
            'put_strike': K_put,
            'put_price': put.price,
            'cost': put.price,
            'protection_level': K_put,
            'break_even': S + put.price,
            'max_loss': put.price,
            'max_gain': float('inf'),
            'delta': 1 + put.delta
        }

    def covered_call(
        self,
        S: float,
        K_call: float,
        T: float,
        sigma: float
    ) -> Dict:
        """
        Analyze covered call strategy.

        Args:
            S: Spot price
            K_call: Call strike
            T: Time to expiration
            sigma: Volatility

        Returns:
            Strategy analysis
        """
        call = self.price_call(S, K_call, T, sigma)

        return {
            'strategy': 'covered_call',
            'spot_price': S,
            'call_strike': K_call,
            'call_premium': call.price,
            'income': call.price,
            'break_even': S - call.price,
            'max_loss': S - call.price,
            'max_gain': K_call - S + call.price,
            'delta': 1 - call.delta
        }

    def collar(
        self,
        S: float,
        K_put: float,
        K_call: float,
        T: float,
        sigma: float
    ) -> Dict:
        """
        Analyze collar strategy (long put + short call).

        Args:
            S: Spot price
            K_put: Put strike
            K_call: Call strike
            T: Time to expiration
            sigma: Volatility

        Returns:
            Strategy analysis
        """
        put = self.price_put(S, K_put, T, sigma)
        call = self.price_call(S, K_call, T, sigma)

        net_cost = put.price - call.price

        return {
            'strategy': 'collar',
            'spot_price': S,
            'put_strike': K_put,
            'call_strike': K_call,
            'put_price': put.price,
            'call_premium_received': call.price,
            'net_cost': net_cost,
            'break_even': S + net_cost,
            'max_loss': S - K_put + net_cost,
            'max_gain': K_call - S - net_cost,
            'delta': 1 + put.delta - call.delta
        }

    def straddle(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> Dict:
        """
        Analyze straddle strategy (long call + long put).

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility

        Returns:
            Strategy analysis
        """
        call = self.price_call(S, K, T, sigma)
        put = self.price_put(S, K, T, sigma)

        total_cost = call.price + put.price

        return {
            'strategy': 'straddle',
            'spot_price': S,
            'strike': K,
            'call_price': call.price,
            'put_price': put.price,
            'total_cost': total_cost,
            'break_even_up': K + total_cost,
            'break_even_down': K - total_cost,
            'max_loss': total_cost,
            'max_gain': float('inf'),
            'delta': call.delta + put.delta
        }

    def get_greeks_surface(
        self,
        S: float,
        K_range: Tuple[float, float],
        T_range: Tuple[float, float],
        sigma: float,
        option_type: str = 'call',
        num_points: int = 10
    ) -> Dict:
        """
        Calculate Greeks surface across strikes and times.

        Args:
            S: Spot price
            K_range: (min_strike, max_strike)
            T_range: (min_time, max_time)
            sigma: Volatility
            option_type: 'call' or 'put'
            num_points: Number of points per dimension

        Returns:
            Dictionary with Greeks surface
        """
        strikes = np.linspace(K_range[0], K_range[1], num_points)
        times = np.linspace(T_range[0], T_range[1], num_points)

        deltas = np.zeros((num_points, num_points))
        gammas = np.zeros((num_points, num_points))
        thetas = np.zeros((num_points, num_points))
        vegas = np.zeros((num_points, num_points))

        for i, K in enumerate(strikes):
            for j, T in enumerate(times):
                option = self.price_option(S, K, T, sigma, option_type)
                deltas[i, j] = option.delta
                gammas[i, j] = option.gamma
                thetas[i, j] = option.theta
                vegas[i, j] = option.vega

        return {
            'strikes': strikes,
            'times': times,
            'delta_surface': deltas,
            'gamma_surface': gammas,
            'theta_surface': thetas,
            'vega_surface': vegas
        }
