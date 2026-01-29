"""
Executor Agent Module
=====================

This agent converts approved signals into executable orders using TWAP
(Time-Weighted Average Price) execution.

Economic Role:
    The Executor minimizes market impact. Large orders move prices adversely:
    
        - Market Impact ∝ √(Order Size / Daily Volume)
    
    TWAP execution slices large orders over time to reduce this impact.
    
    For sector ETFs (highly liquid), market impact is minimal. But we
    implement TWAP for:
    1. Consistency with real execution
    2. Educational value (understanding execution algos)
    3. Future extension to less liquid assets

LangGraph Integration:
    Final node in the workflow:
    
        [Risk Manager] → [Executor] → [Fill Reports]
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .risk_manager import RiskAdjustedSignals


class Order(BaseModel):
    """Represents a single order to be executed."""
    
    ticker: str
    direction: Literal["buy", "sell"]
    target_shares: float
    slice_shares: float = Field(description="Shares per TWAP slice")
    n_slices: int = Field(description="Number of TWAP slices")
    urgency: Literal["low", "medium", "high"] = "medium"
    limit_price: float | None = None


class ExecutionReport(BaseModel):
    """Report of executed orders."""
    
    timestamp: str
    orders: list[Order]
    estimated_cost_bps: float = Field(description="Estimated execution cost in bps")
    estimated_time_minutes: int = Field(description="Estimated execution time")
    total_notional: float = Field(description="Total order value in USD")


@dataclass
class ExecutorState:
    """Internal state for Executor agent."""
    
    pending_orders: list[Order] = None
    executed_today: float = 0.0
    
    def __post_init__(self):
        if self.pending_orders is None:
            self.pending_orders = []


class ExecutorAgent:
    """Executor agent for order generation and TWAP execution.
    
    Converts risk-adjusted signals into executable orders:
    
    1. Compute target position changes
    2. Estimate market impact
    3. Determine TWAP slices based on volume
    4. Generate order schedule
    
    Example:
        >>> executor = ExecutorAgent(
        ...     tickers=['XLK', 'XLF'],
        ...     portfolio_value=1_000_000,
        ... )
        >>> report = executor.generate_orders(
        ...     signals=risk_adjusted_signals,
        ...     current_prices=prices,
        ...     current_positions=positions,
        ... )
    """
    
    def __init__(
        self,
        tickers: list[str],
        portfolio_value: float = 1_000_000.0,
        twap_slices: int = 5,
        max_participation_rate: float = 0.10,  # Max 10% of ADV
        urgency_threshold: float = 0.5,  # Signal strength for high urgency
    ) -> None:
        """Initialize Executor.
        
        Args:
            tickers: Asset universe.
            portfolio_value: Current portfolio value.
            twap_slices: Default number of TWAP slices.
            max_participation_rate: Maximum fraction of daily volume.
            urgency_threshold: Signal threshold for high urgency.
        """
        self.tickers = tickers
        self.portfolio_value = portfolio_value
        self.default_slices = twap_slices
        self.max_participation = max_participation_rate
        self.urgency_threshold = urgency_threshold
        
        self.state = ExecutorState()
    
    def generate_orders(
        self,
        signals: RiskAdjustedSignals,
        current_prices: dict[str, float],
        current_positions: dict[str, float],  # shares held
        avg_daily_volume: dict[str, float] | None = None,
    ) -> ExecutionReport:
        """Generate orders from approved signals.
        
        Args:
            signals: Risk-adjusted signals from Risk Manager.
            current_prices: Current prices per ticker.
            current_positions: Current shares held per ticker.
            avg_daily_volume: Optional ADV for market impact estimation.
            
        Returns:
            ExecutionReport with order details.
        """
        orders: list[Order] = []
        total_notional = 0.0
        
        for ticker in self.tickers:
            target_weight = signals.approved_signals.get(ticker, 0.0)
            price = current_prices.get(ticker, 0.0)
            
            if price <= 0:
                continue
            
            # Compute target shares
            target_value = target_weight * self.portfolio_value
            target_shares = target_value / price
            
            # Current shares
            current_shares = current_positions.get(ticker, 0.0)
            
            # Shares to trade
            trade_shares = target_shares - current_shares
            
            if abs(trade_shares) < 1:  # Minimum 1 share
                continue
            
            # Determine direction and urgency
            direction: Literal["buy", "sell"] = "buy" if trade_shares > 0 else "sell"
            urgency = self._compute_urgency(abs(target_weight))
            
            # Compute TWAP slices
            adv = avg_daily_volume.get(ticker, 100_000) if avg_daily_volume else 100_000
            n_slices = self._compute_slices(abs(trade_shares), adv)
            slice_shares = abs(trade_shares) / n_slices
            
            orders.append(Order(
                ticker=ticker,
                direction=direction,
                target_shares=abs(trade_shares),
                slice_shares=slice_shares,
                n_slices=n_slices,
                urgency=urgency,
            ))
            
            total_notional += abs(trade_shares * price)
        
        # Estimate execution cost
        estimated_cost = self._estimate_execution_cost(orders, avg_daily_volume)
        
        # Estimate execution time (5 min per slice on average)
        max_slices = max((o.n_slices for o in orders), default=1)
        estimated_time = max_slices * 5
        
        return ExecutionReport(
            timestamp=signals.timestamp,
            orders=orders,
            estimated_cost_bps=estimated_cost,
            estimated_time_minutes=estimated_time,
            total_notional=total_notional,
        )
    
    def _compute_urgency(
        self,
        signal_strength: float,
    ) -> Literal["low", "medium", "high"]:
        """Determine order urgency based on signal strength.
        
        Economic Logic:
            Strong signals = more conviction = faster execution
            Weak signals = less urgency = more patience
        """
        if signal_strength > self.urgency_threshold:
            return "high"
        elif signal_strength > self.urgency_threshold / 2:
            return "medium"
        else:
            return "low"
    
    def _compute_slices(
        self,
        shares: float,
        adv: float,
    ) -> int:
        """Compute optimal number of TWAP slices.
        
        Economic Logic:
            Larger orders relative to ADV need more slices to
            minimize market impact.
            
            Participation rate = shares / (ADV * slices)
            Target: participation < max_participation
        """
        if adv <= 0:
            return self.default_slices
        
        # Minimum slices to stay below max participation
        min_slices = int(np.ceil(shares / (adv * self.max_participation)))
        
        return max(min_slices, self.default_slices)
    
    def _estimate_execution_cost(
        self,
        orders: list[Order],
        adv: dict[str, float] | None,
    ) -> float:
        """Estimate total execution cost in basis points.
        
        Market Impact Model (simplified square-root model):
            Impact = σ * √(OrderSize / ADV)
        
        For liquid ETFs, this is typically 1-5 bps.
        """
        if not orders:
            return 0.0
        
        total_impact = 0.0
        total_value = 0.0
        
        sigma = 0.02  # Assume 2% daily vol for ETFs
        
        for order in orders:
            ticker_adv = adv.get(order.ticker, 100_000) if adv else 100_000
            
            # Square-root market impact
            participation = order.target_shares / ticker_adv
            impact = sigma * np.sqrt(participation) * 10000  # Convert to bps
            
            # Weight by order size (estimate value assuming $100 price)
            order_value = order.target_shares * 100
            total_impact += impact * order_value
            total_value += order_value
        
        return total_impact / total_value if total_value > 0 else 0.0
    
    def simulate_execution(
        self,
        order: Order,
        price: float,
        volatility: float = 0.02,
    ) -> list[tuple[float, float]]:
        """Simulate TWAP execution with slippage.
        
        Returns list of (shares, execution_price) tuples for each slice.
        
        This is useful for backtesting with realistic execution.
        """
        executions = []
        
        for i in range(order.n_slices):
            # Add some random noise to price (slippage)
            noise = np.random.normal(0, volatility / np.sqrt(252))
            
            if order.direction == "buy":
                exec_price = price * (1 + abs(noise))  # Adverse for buyer
            else:
                exec_price = price * (1 - abs(noise))  # Adverse for seller
            
            executions.append((order.slice_shares, exec_price))
        
        return executions
