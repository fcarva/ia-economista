"""
Risk Manager Agent Module
=========================

This agent filters and adjusts trading signals based on risk constraints.

Economic Role:
    The Risk Manager is the "gatekeeper". They ensure that:
    1. No single position exceeds concentration limits
    2. Portfolio VaR stays within acceptable bounds
    3. Drawdown limits are respected
    4. Sector exposures are balanced
    
    The Risk Manager can:
    - Scale down all signals (reduce gross exposure)
    - Reject specific signals (too risky)
    - Override in crisis mode (de-risk everything)

LangGraph Integration:
    This agent sits between Analyst and Executor:
    
        [Analyst] → [Risk Manager] → [Executor]
                          ↓
                   (VaR, Drawdown, Limits)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..strategy.strategy import RiskMetrics
from .analyst import SignalOutput


class RiskAdjustedSignals(BaseModel):
    """Output from Risk Manager agent."""
    
    timestamp: str = Field(description="Risk check timestamp")
    approved_signals: dict[str, float] = Field(description="Risk-adjusted signals")
    rejected_signals: dict[str, str] = Field(description="Rejected signals with reasons")
    risk_metrics: dict[str, float] = Field(description="Current risk metrics")
    scaling_factor: float = Field(description="Global scaling applied to signals")
    risk_status: Literal["normal", "elevated", "critical"] = Field(
        description="Overall risk status"
    )
    actions_taken: list[str] = Field(description="Risk actions taken")


@dataclass
class RiskState:
    """Internal state maintained by Risk Manager."""
    
    current_var: float = 0.0
    current_drawdown: float = 0.0
    high_water_mark: float = 0.0
    consecutive_losses: int = 0
    last_equity: float = 0.0


class RiskManagerAgent:
    """Risk Manager agent for signal filtering and adjustment.
    
    Implements a multi-layer risk framework:
    
    Layer 1: Position Limits
        - Max single position: 20% of portfolio
        - Max sector concentration: 30%
        
    Layer 2: Portfolio Risk
        - VaR(95%) < 2% daily
        - Max drawdown < 15%
        
    Layer 3: Regime-Based Scaling
        - High volatility → reduce exposure
        - Drawdown mode → reduce aggressively
        
    Example:
        >>> risk_mgr = RiskManagerAgent(tickers=['XLK', 'XLF'])
        >>> adjusted = risk_mgr.filter_signals(
        ...     analyst_output=signals,
        ...     portfolio_returns=recent_returns,
        ...     current_equity=1_000_000,
        ... )
    """
    
    def __init__(
        self,
        tickers: list[str],
        max_position_weight: float = 0.20,
        max_sector_weight: float = 0.30,
        max_var_daily: float = 0.02,
        max_drawdown: float = 0.15,
        var_scaling_threshold: float = 0.015,
        drawdown_scaling_threshold: float = 0.10,
        min_confidence: float = 0.15,
    ) -> None:
        """Initialize Risk Manager.
        
        Args:
            tickers: Asset universe.
            max_position_weight: Maximum weight for single position.
            max_sector_weight: Maximum weight for sector (for sector ETFs, same as position).
            max_var_daily: Maximum acceptable daily VaR at 95%.
            max_drawdown: Maximum acceptable drawdown.
            var_scaling_threshold: VaR level at which to start scaling.
            drawdown_scaling_threshold: Drawdown at which to start scaling.
        """
        self.tickers = tickers
        self.max_position = max_position_weight
        self.max_sector = max_sector_weight
        self.max_var = max_var_daily
        self.max_drawdown = max_drawdown
        self.var_threshold = var_scaling_threshold
        self.dd_threshold = drawdown_scaling_threshold
        self.min_confidence = min_confidence
        
        self.state = RiskState()
    
    def filter_signals(
        self,
        analyst_output: SignalOutput,
        portfolio_returns: pd.Series,
        current_equity: float,
    ) -> RiskAdjustedSignals:
        """Filter and adjust signals based on risk constraints.
        
        Args:
            analyst_output: Raw signals from Analyst agent.
            portfolio_returns: Recent portfolio return series.
            current_equity: Current portfolio value.
            
        Returns:
            RiskAdjustedSignals with approved signals and risk metadata.
        """
        actions_taken: list[str] = []
        rejected: dict[str, str] = {}
        
        # Update internal state
        self._update_state(portfolio_returns, current_equity)
        
        # Get current risk metrics
        risk_metrics = self._compute_risk_metrics(portfolio_returns, current_equity)
        
        # Determine risk status
        risk_status = self._assess_risk_status(risk_metrics)
        
        # 1. Compute global scaling factor based on risk
        scaling_factor = self._compute_scaling_factor(risk_metrics)
        regime_scale = self._regime_scaling(analyst_output.regime)
        if regime_scale < 1.0:
            actions_taken.append(
                f"Scaled signals by {regime_scale:.2f} due to regime={analyst_output.regime}"
            )
        scaling_factor *= regime_scale
        if scaling_factor < 1.0:
            actions_taken.append(f"Scaled all signals by {scaling_factor:.2f} due to elevated risk")
        
        # 2. Apply position limits and filter signals
        approved_signals: dict[str, float] = {}
        
        for ticker, signal in analyst_output.signals.items():
            # Scale by global factor
            scaled_signal = signal * scaling_factor
            
            # Check position limits
            if abs(scaled_signal) > self.max_position:
                original = scaled_signal
                scaled_signal = np.sign(scaled_signal) * self.max_position
                actions_taken.append(
                    f"{ticker}: Capped from {original:.2f} to {scaled_signal:.2f}"
                )
            
            # Check confidence threshold
            confidence = analyst_output.confidence.get(ticker, 0.0)
            if confidence < self.min_confidence and abs(scaled_signal) > 0.05:
                rejected[ticker] = f"Low confidence ({confidence:.2f})"
                scaled_signal = 0.0
            
            # Check for crisis mode override
            if risk_status == "critical" and abs(scaled_signal) > 0.05:
                rejected[ticker] = "Critical risk mode - reducing exposure"
                scaled_signal = scaled_signal * 0.25  # Reduce to 25%
            
            approved_signals[ticker] = scaled_signal
        
        # 3. Check gross exposure
        gross_exposure = sum(abs(s) for s in approved_signals.values())
        if gross_exposure > 2.0:  # Max 200% gross
            scale = 2.0 / gross_exposure
            approved_signals = {t: s * scale for t, s in approved_signals.items()}
            actions_taken.append(f"Reduced gross exposure from {gross_exposure:.0%} to 200%")
        
        return RiskAdjustedSignals(
            timestamp=analyst_output.timestamp,
            approved_signals=approved_signals,
            rejected_signals=rejected,
            risk_metrics=risk_metrics,
            scaling_factor=scaling_factor,
            risk_status=risk_status,
            actions_taken=actions_taken,
        )
    
    def _update_state(
        self,
        returns: pd.Series,
        current_equity: float,
    ) -> None:
        """Update internal risk state."""
        # Update high water mark
        if current_equity > self.state.high_water_mark:
            self.state.high_water_mark = current_equity
        
        # Calculate current drawdown
        if self.state.high_water_mark > 0:
            self.state.current_drawdown = (
                self.state.high_water_mark - current_equity
            ) / self.state.high_water_mark
        
        # Track consecutive losses
        if len(returns) > 0:
            if returns.iloc[-1] < 0:
                self.state.consecutive_losses += 1
            else:
                self.state.consecutive_losses = 0
        
        self.state.last_equity = current_equity
    
    def _compute_risk_metrics(
        self,
        returns: pd.Series,
        current_equity: float,
    ) -> dict[str, float]:
        """Compute current risk metrics."""
        if len(returns) < 20:
            return {
                "var_95": 0.0,
                "cvar_95": 0.0,
                "current_drawdown": self.state.current_drawdown,
                "volatility_20d": 0.0,
                "consecutive_losses": float(self.state.consecutive_losses),
            }
        
        var_95 = RiskMetrics.compute_var(returns.tail(60), confidence=0.95)
        cvar_95 = RiskMetrics.compute_cvar(returns.tail(60), confidence=0.95)
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        
        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "current_drawdown": self.state.current_drawdown,
            "volatility_20d": vol_20d,
            "consecutive_losses": float(self.state.consecutive_losses),
        }
    
    def _assess_risk_status(
        self,
        metrics: dict[str, float],
    ) -> Literal["normal", "elevated", "critical"]:
        """Assess overall risk status."""
        var = metrics.get("var_95", 0.0)
        dd = metrics.get("current_drawdown", 0.0)
        consec = metrics.get("consecutive_losses", 0)
        
        # Critical conditions
        if dd > self.max_drawdown or var > self.max_var or consec > 5:
            return "critical"
        
        # Elevated conditions
        if dd > self.dd_threshold or var > self.var_threshold or consec > 3:
            return "elevated"
        
        return "normal"
    
    def _compute_scaling_factor(
        self,
        metrics: dict[str, float],
    ) -> float:
        """Compute global scaling factor based on risk.
        
        Economic Logic:
            Scale down exposure as VaR or drawdown approach limits.
            This implements "VolTargeting" - reduce risk as vol rises.
        """
        var = metrics.get("var_95", 0.0)
        dd = metrics.get("current_drawdown", 0.0)
        
        # VaR-based scaling
        if var > self.var_threshold:
            var_scale = self.var_threshold / var
        else:
            var_scale = 1.0
        
        # Drawdown-based scaling
        if dd > self.dd_threshold:
            # Progressive scaling: 50% at threshold, 0% at max
            dd_scale = max(0.0, 1.0 - (dd - self.dd_threshold) / (self.max_drawdown - self.dd_threshold))
        else:
            dd_scale = 1.0
        
        # Take the minimum (most conservative)
        return min(var_scale, dd_scale, 1.0)

    def _regime_scaling(self, regime: str) -> float:
        """Apply a light regime-based scaling to exposure."""
        if regime == "high_correlation":
            return 0.8
        if regime == "dispersion":
            return 1.0
        return 1.0
