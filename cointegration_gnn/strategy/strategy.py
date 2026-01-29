"""
Trading Strategy Module
=======================

This module translates GNN signals into executable trading positions
with Kelly-based position sizing and risk controls.

Economic Logic:
    Raw GNN output is a signal in [-1, 1]. We need to convert this to:
    1. Actual dollar positions
    2. With Kelly-optimal sizing (maximize log utility)
    3. Subject to risk constraints (VaR, sector limits)
    
    The Kelly Criterion states that optimal bet size is:
    
        f* = (p * b - q) / b = edge / odds
        
    where p = win probability, q = 1-p, b = win/loss ratio.
    
    For continuous returns, this becomes:
    
        f* = μ / σ² = Sharpe / σ
        
    We use a "fractional Kelly" (typically 0.25-0.5 of full Kelly) to
    reduce variance and avoid ruin.

References:
    - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
    - Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting,
      and the Stock Market"
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Position:
    """Represents a position in a single asset."""
    
    ticker: str
    shares: float
    direction: Literal["long", "short", "flat"]
    dollar_value: float
    weight: float  # As fraction of portfolio
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None


@dataclass
class Portfolio:
    """Current portfolio state."""
    
    positions: dict[str, Position]
    cash: float
    total_value: float
    timestamp: pd.Timestamp
    
    @property
    def weights(self) -> dict[str, float]:
        """Get current portfolio weights."""
        return {ticker: pos.weight for ticker, pos in self.positions.items()}
    
    @property
    def net_exposure(self) -> float:
        """Net market exposure (long - short)."""
        return sum(pos.weight for pos in self.positions.values())
    
    @property
    def gross_exposure(self) -> float:
        """Gross market exposure (|long| + |short|)."""
        return sum(abs(pos.weight) for pos in self.positions.values())


class KellyCalculator:
    """Computes Kelly-optimal position sizes.
    
    Economic Rationale:
        Kelly criterion maximizes long-run growth rate. However, full Kelly
        can have extreme variance (50% drawdowns are common). We use
        fractional Kelly (0.25-0.5x) for practical risk management.
    
    Example:
        >>> kelly = KellyCalculator(lookback=252, fraction=0.25)
        >>> optimal_weight = kelly.compute_kelly_weight(
        ...     signal=0.8,  # Strong buy signal
        ...     returns=historical_returns,
        ...     volatility=0.20,
        ... )
    """
    
    def __init__(
        self,
        lookback: int = 252,
        fraction: float = 0.25,  # Quarter-Kelly
        max_weight: float = 0.20,
        min_weight: float = 0.01,
    ) -> None:
        """Initialize Kelly calculator.
        
        Args:
            lookback: Days for estimating mean/variance.
            fraction: Kelly fraction (0.25 = quarter-Kelly).
            max_weight: Maximum position weight.
            min_weight: Minimum position weight (positions below go to 0).
        """
        self.lookback = lookback
        self.fraction = fraction
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def compute_kelly_weight(
        self,
        signal: float,
        returns: pd.Series,
        volatility: float | None = None,
    ) -> float:
        """Compute Kelly-optimal weight for a position.
        
        Kelly formula for continuous returns:
            f* = μ / σ²
        
        Modified by signal strength and Kelly fraction:
            f = fraction * signal * (μ / σ²)
        
        Args:
            signal: GNN signal in [-1, 1].
            returns: Historical return series for this asset.
            volatility: Optional pre-computed volatility.
            
        Returns:
            Optimal position weight (positive = long, negative = short).
        """
        if len(returns) < self.lookback:
            # Not enough history, use conservative sizing
            return signal * self.min_weight
        
        recent_returns = returns.iloc[-self.lookback:]
        
        mu = recent_returns.mean() * 252  # Annualized mean
        sigma = volatility if volatility else recent_returns.std() * np.sqrt(252)
        
        if sigma < 1e-8:
            return 0.0
        
        # Kelly fraction: f* = μ / σ²
        full_kelly = mu / (sigma ** 2)
        
        # Apply Kelly fraction and signal
        kelly_weight = self.fraction * signal * full_kelly
        
        # Clip to bounds
        kelly_weight = np.clip(kelly_weight, -self.max_weight, self.max_weight)
        
        # Zero out tiny positions
        if abs(kelly_weight) < self.min_weight:
            kelly_weight = 0.0
        
        return kelly_weight


class Strategy:
    """Converts GNN signals to tradeable positions.
    
    Pipeline:
        1. Receive raw signals from GNN [-1, 1]
        2. Apply Kelly sizing based on historical returns
        3. Apply risk constraints (VaR, concentration)
        4. Generate target portfolio
        5. Compute trades to reach target
    
    Economic Logic:
        We treat each sector ETF as an independent bet. The GNN provides
        the "edge" (expected return direction), and Kelly provides the
        "sizing" (how much to bet given the edge and variance).
    """
    
    def __init__(
        self,
        tickers: list[str],
        kelly_fraction: float = 0.25,
        max_position: float = 0.20,
        max_gross_exposure: float = 2.0,
        max_sector_concentration: float = 0.30,
        min_holding_period: int = 1,
    ) -> None:
        """Initialize strategy.
        
        Args:
            tickers: Asset universe.
            kelly_fraction: Fraction of Kelly to use.
            max_position: Maximum single position weight.
            max_gross_exposure: Maximum gross exposure (sum of |weights|).
            max_sector_concentration: Maximum weight in any single sector.
            min_holding_period: Minimum days to hold a position.
        """
        self.tickers = tickers
        self.kelly = KellyCalculator(fraction=kelly_fraction, max_weight=max_position)
        self.max_gross_exposure = max_gross_exposure
        self.max_sector_concentration = max_sector_concentration
        self.min_holding_period = min_holding_period
        
        # Track position entry dates for holding period
        self.entry_dates: dict[str, pd.Timestamp] = {}
    
    def generate_target_weights(
        self,
        signals: dict[str, float],
        returns: pd.DataFrame,
        volatilities: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Convert signals to target portfolio weights.
        
        Args:
            signals: GNN signals for each ticker.
            returns: Historical return DataFrame.
            volatilities: Optional pre-computed volatilities.
            
        Returns:
            Target weights for each ticker.
        """
        target_weights: dict[str, float] = {}
        
        for ticker in self.tickers:
            signal = signals.get(ticker, 0.0)
            
            if ticker not in returns.columns:
                target_weights[ticker] = 0.0
                continue
            
            vol = volatilities.get(ticker) if volatilities else None
            
            # Compute Kelly-optimal weight
            # Noise Filter: Ignore weak signals
            if abs(signal) < 0.05:
                signal = 0.0
                
            weight = self.kelly.compute_kelly_weight(
                signal=signal,
                returns=returns[ticker],
                volatility=vol,
            )
            
            target_weights[ticker] = weight
        
        # Apply risk constraints
        target_weights = self._apply_risk_constraints(target_weights)
        
        return target_weights
    
    def _apply_risk_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply risk limits to target weights.
        
        Constraints:
        1. Max position size
        2. Max gross exposure
        3. Sector concentration (for sector ETFs, this is single-asset)
        """
        # 1. Cap individual positions
        capped_weights = {
            ticker: np.clip(w, -self.max_sector_concentration, self.max_sector_concentration)
            for ticker, w in weights.items()
        }
        
        # 2. Scale down if gross exposure too high
        gross = sum(abs(w) for w in capped_weights.values())
        
        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            capped_weights = {t: w * scale for t, w in capped_weights.items()}
        
        return capped_weights
    
    def compute_trades(
        self,
        current_portfolio: Portfolio,
        target_weights: dict[str, float],
        prices: dict[str, float],
        timestamp: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute trades needed to reach target portfolio.
        
        Args:
            current_portfolio: Current portfolio state.
            target_weights: Target weights from generate_target_weights.
            prices: Current prices for each ticker.
            timestamp: Current timestamp (for holding period check).
            
        Returns:
            Dictionary of trades: {ticker: dollar_amount} (negative = sell).
        """
        trades: dict[str, float] = {}
        total_value = current_portfolio.total_value
        
        for ticker in self.tickers:
            target_weight = target_weights.get(ticker, 0.0)
            current_weight = current_portfolio.weights.get(ticker, 0.0)
            
            # Check minimum holding period
            if ticker in self.entry_dates:
                days_held = (timestamp - self.entry_dates[ticker]).days
                if days_held < self.min_holding_period:
                    # Keep current position
                    continue
            
            # Compute trade
            weight_diff = target_weight - current_weight
            dollar_trade = weight_diff * total_value
            
            if abs(dollar_trade) > 100:  # Minimum trade size
                trades[ticker] = dollar_trade
                
                # Update entry date if new position
                if current_weight == 0 and target_weight != 0:
                    self.entry_dates[ticker] = timestamp
                elif target_weight == 0:
                    self.entry_dates.pop(ticker, None)
        
        return trades
    
    def compute_turnover(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> float:
        """Compute portfolio turnover.
        
        Turnover = Σ |w_target - w_current| / 2
        
        High turnover typically indicates poor signal quality or overfitting.
        Target: < 20% monthly turnover for this strategy.
        """
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        
        turnover = sum(
            abs(target_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        )
        
        return turnover / 2


class RiskMetrics:
    """Calculate risk metrics for portfolio monitoring.
    
    Economic Logic:
        We need to monitor several risk dimensions:
        1. VaR (Value at Risk): Maximum expected loss at 95%
        2. CVaR (Expected Shortfall): Average loss in worst 5%
        3. Max Drawdown: Worst peak-to-trough decline
        4. Correlation: Exposure to market risk
    """
    
    @staticmethod
    def compute_var(
        returns: pd.Series,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Compute Value at Risk (Parametric, assuming normal distribution).
        
        VaR_{95%} = μ - 1.645 * σ * √(horizon)
        
        Args:
            returns: Portfolio return series.
            confidence: Confidence level (0.95 = 95%).
            horizon_days: Horizon in trading days.
            
        Returns:
            VaR as a positive number (potential loss).
        """
        from scipy import stats
        
        mu = returns.mean()
        sigma = returns.std()
        
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mu - z_score * sigma * np.sqrt(horizon_days))
        
        return max(var, 0.0)
    
    @staticmethod
    def compute_cvar(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall).
        
        CVaR = E[Loss | Loss > VaR]
        
        This is the average of losses worse than VaR.
        """
        var = RiskMetrics.compute_var(returns, confidence)
        worst_returns = returns[returns < -var]
        
        if len(worst_returns) == 0:
            return var
        
        return -worst_returns.mean()
    
    @staticmethod
    def compute_max_drawdown(equity_curve: pd.Series) -> float:
        """Compute maximum drawdown.
        
        MaxDD = max(peak - trough) / peak
        """
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        return abs(drawdown.min())
    
    @staticmethod
    def compute_sharpe(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252,
    ) -> float:
        """Compute annualized Sharpe ratio.
        
        Sharpe = (μ - r_f) / σ * √252
        """
        excess_return = returns.mean() - risk_free_rate / annualization_factor
        volatility = returns.std()
        
        if volatility < 1e-8:
            return 0.0
        
        return excess_return / volatility * np.sqrt(annualization_factor)
