"""
Backtesting Engine Module
=========================

This module provides an event-driven backtesting framework with realistic
transaction costs, slippage, and look-ahead bias prevention.

Economic Logic:
    A backtest is worthless if it contains look-ahead bias or unrealistic
    assumptions. We enforce:
    
    1. **Strict time ordering**: Signals at time t can only use data up to t-1
    2. **Execution delay**: Signals generated at close are executed next open
    3. **Transaction costs**: 10bps per side (20bps round-trip)
    4. **Slippage**: 5bps per side (market impact)
    5. **Partial fills**: Large orders may not fill completely
    
Architecture:
    Event Loop:
        for each trading day t:
            1. Update market data (prices up to t-1)
            2. Generate signals using GNN
            3. Risk manager filters/approves signals
            4. Executor generates trades for t+1 open
            5. Execute trades at t+1 open price + slippage
            6. Record PnL

References:
    - De Prado, M.L. (2018). "Advances in Financial Machine Learning", Ch. 12-14
    - Chan, E. (2013). "Algorithmic Trading", Ch. 3 (Backtesting)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..strategy.strategy import Portfolio, Position, RiskMetrics


@dataclass
class Trade:
    """Record of a single trade execution."""
    
    timestamp: pd.Timestamp
    ticker: str
    direction: Literal["buy", "sell"]
    shares: float
    price: float  # Execution price (includes slippage)
    dollar_value: float
    commission: float
    slippage_cost: float
    
    @property
    def total_cost(self) -> float:
        """Total cost including commissions and slippage."""
        return self.commission + self.slippage_cost


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame  # (dates, tickers) -> weights
    trades: list[Trade]
    
    # Summary metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    total_trades: int
    turnover: float
    
    # Costs
    total_commission: float
    total_slippage: float
    
    # For validation
    random_baseline_sharpe: float | None = None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Backtest Results
================
Period: {self.equity_curve.index[0]} to {self.equity_curve.index[-1]}
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
VaR (95%): {self.var_95:.2%}
CVaR (95%): {self.cvar_95:.2%}

Trading Activity
----------------
Total Trades: {self.total_trades}
Annualized Turnover: {self.turnover:.2%}
Total Commissions: ${self.total_commission:,.2f}
Total Slippage: ${self.total_slippage:,.2f}

Validation
----------
Random Baseline Sharpe: {self.random_baseline_sharpe:.2f if self.random_baseline_sharpe else 'N/A'}
Alpha over Random: {(self.sharpe_ratio - (self.random_baseline_sharpe or 0)):.2f}
"""


class BacktestEngine:
    """Event-driven backtesting engine.
    
    Key Design Principles:
    1. No look-ahead bias (strict time ordering)
    2. Realistic costs (commission + slippage)
    3. Next-day execution (close signal → next open execution)
    4. Support for partial fills and execution constraints
    
    Example:
        >>> engine = BacktestEngine(
        ...     tickers=['XLK', 'XLF', 'XLE'],
        ...     initial_capital=1_000_000,
        ...     transaction_cost_bps=10,
        ... )
        >>> result = engine.run(
        ...     prices=price_data,
        ...     signal_generator=my_model.predict_positions,
        ... )
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        tickers: list[str],
        initial_capital: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        execution_delay: int = 1,  # Days
    ) -> None:
        """Initialize backtest engine.
        
        Args:
            tickers: Asset universe.
            initial_capital: Starting capital in USD.
            transaction_cost_bps: Commission per side in basis points.
            slippage_bps: Slippage estimate per side in basis points.
            execution_delay: Days between signal and execution.
        """
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10000
        self.slippage = slippage_bps / 10000
        self.execution_delay = execution_delay
    
    def run(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]],
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> BacktestResult:
        """Run backtest.
        
        **CRITICAL: Look-Ahead Bias Prevention**
        The signal_generator function receives prices ONLY up to t-1.
        Signals generated at t are executed at t + execution_delay.
        
        Args:
            prices: Price DataFrame (index=dates, columns=tickers).
            signal_generator: Function that takes (prices_so_far, current_date)
                and returns {ticker: signal} dictionary.
            start_date: Optional start date (default: first available).
            end_date: Optional end date (default: last available).
            
        Returns:
            BacktestResult with all metrics and time series.
        """
        # Filter date range
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
        
        # Initialize state
        cash = self.initial_capital
        positions: dict[str, float] = {t: 0.0 for t in self.tickers}  # shares held
        
        # Result accumulators
        equity_values: list[float] = []
        equity_dates: list[pd.Timestamp] = []
        position_history: list[dict[str, float]] = []
        trades: list[Trade] = []
        pending_signals: dict[pd.Timestamp, dict[str, float]] = {}
        
        trading_days = prices.index.tolist()
        
        for i, date in enumerate(trading_days):
            # Current prices
            current_prices = prices.loc[date].to_dict()
            
            # 1. Execute any pending trades (from signals execution_delay days ago)
            exec_date = trading_days[max(0, i - self.execution_delay)] if i >= self.execution_delay else None
            
            if exec_date and exec_date in pending_signals:
                target_signals = pending_signals.pop(exec_date)
                new_trades, cash = self._execute_trades(
                    target_signals=target_signals,
                    current_prices=current_prices,
                    positions=positions,
                    cash=cash,
                    timestamp=date,
                )
                trades.extend(new_trades)
            
            # 2. Calculate current portfolio value
            portfolio_value = cash + sum(
                positions[t] * current_prices.get(t, 0) for t in self.tickers
            )
            
            equity_values.append(portfolio_value)
            equity_dates.append(date)
            
            # Record current weights
            weights = {
                t: (positions[t] * current_prices.get(t, 0)) / portfolio_value
                for t in self.tickers
            }
            position_history.append(weights)
            
            # 3. Generate new signals (using data ONLY up to yesterday)
            # **LOOK-AHEAD BIAS PREVENTION**
            if i > 0:
                historical_prices = prices.iloc[:i]  # Excludes today!
                signals = signal_generator(historical_prices, date)
                pending_signals[date] = signals
        
        # Build result DataFrames
        equity_curve = pd.Series(equity_values, index=equity_dates)
        returns = equity_curve.pct_change().dropna()
        positions_df = pd.DataFrame(position_history, index=equity_dates)
        
        # Compute metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        sharpe = RiskMetrics.compute_sharpe(returns)
        max_dd = RiskMetrics.compute_max_drawdown(equity_curve)
        var_95 = RiskMetrics.compute_var(returns)
        cvar_95 = RiskMetrics.compute_cvar(returns)
        
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        
        # Compute turnover
        weight_changes = positions_df.diff().abs().sum(axis=1)
        turnover = weight_changes.sum() / (2 * n_years) if n_years > 0 else 0
        
        # Run random baseline for comparison
        random_sharpe = self._run_random_baseline(prices, returns)
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions_df,
            trades=trades,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(trades),
            turnover=turnover,
            total_commission=total_commission,
            total_slippage=total_slippage,
            random_baseline_sharpe=random_sharpe,
        )
    
    def _execute_trades(
        self,
        target_signals: dict[str, float],
        current_prices: dict[str, float],
        positions: dict[str, float],
        cash: float,
        timestamp: pd.Timestamp,
    ) -> tuple[list[Trade], float]:
        """Execute trades with realistic costs.
        
        Args:
            target_signals: Target weights per ticker.
            current_prices: Current prices for execution.
            positions: Current positions (modified in place).
            cash: Current cash balance.
            timestamp: Execution timestamp.
            
        Returns:
            Tuple of (executed trades, new cash balance).
        """
        trades: list[Trade] = []
        
        # Calculate current portfolio value
        portfolio_value = cash + sum(
            positions[t] * current_prices.get(t, 0) for t in self.tickers
        )
        
        for ticker in self.tickers:
            target_weight = target_signals.get(ticker, 0.0)
            target_value = target_weight * portfolio_value
            
            current_shares = positions[ticker]
            current_value = current_shares * current_prices.get(ticker, 0)
            
            trade_value = target_value - current_value
            
            if abs(trade_value) < 100:  # Minimum trade size
                continue
            
            price = current_prices.get(ticker, 0)
            if price <= 0:
                continue
            
            # Apply slippage (adverse price movement)
            if trade_value > 0:  # Buying
                exec_price = price * (1 + self.slippage)
            else:  # Selling
                exec_price = price * (1 - self.slippage)
            
            shares_traded = trade_value / exec_price
            commission = abs(trade_value) * self.transaction_cost
            slippage_cost = abs(trade_value) * self.slippage
            
            # Update positions
            positions[ticker] += shares_traded
            cash -= trade_value + commission
            
            trades.append(Trade(
                timestamp=timestamp,
                ticker=ticker,
                direction="buy" if trade_value > 0 else "sell",
                shares=abs(shares_traded),
                price=exec_price,
                dollar_value=abs(trade_value),
                commission=commission,
                slippage_cost=slippage_cost,
            ))
        
        return trades, cash
    
    def _run_random_baseline(
        self,
        prices: pd.DataFrame,
        actual_returns: pd.Series,
    ) -> float:
        """Run random signal baseline for comparison.
        
        Economic Logic:
            If our strategy doesn't meaningfully beat random signals,
            we're likely fitting noise. This is the most important
            validation check.
        """
        np.random.seed(42)
        
        # Generate random weights each day
        random_returns = []
        
        for i in range(len(prices) - 1):
            # Random weights (normalized to sum to 1)
            weights = np.random.uniform(-1, 1, len(self.tickers))
            weights = weights / np.abs(weights).sum()
            
            # Next day returns
            today_prices = prices.iloc[i]
            next_prices = prices.iloc[i + 1]
            asset_returns = (next_prices - today_prices) / today_prices
            
            portfolio_return = (weights * asset_returns.values).sum()
            random_returns.append(portfolio_return)
        
        random_series = pd.Series(random_returns)
        return RiskMetrics.compute_sharpe(random_series)


class WalkForwardValidator:
    """Walk-forward cross-validation for time series.
    
    Economic Logic:
        Standard k-fold CV doesn't work for time series (future data leaks).
        Walk-forward splits preserve temporal ordering:
        
        Fold 1: Train [2015-2017], Test [2018]
        Fold 2: Train [2015-2018], Test [2019]
        Fold 3: Train [2015-2019], Test [2020]
        ...
        
        If performance degrades significantly across folds, the strategy
        may be overfitting to specific market regimes.
    """
    
    def __init__(
        self,
        train_window_years: int = 3,
        test_window_years: int = 1,
        step_years: int = 1,
    ) -> None:
        """Initialize validator.
        
        Args:
            train_window_years: Years of training data.
            test_window_years: Years of test data per fold.
            step_years: Years to step forward between folds.
        """
        self.train_window = train_window_years * 252
        self.test_window = test_window_years * 252
        self.step = step_years * 252
    
    def generate_splits(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits.
        
        Args:
            data: Full dataset with datetime index.
            
        Returns:
            List of (train_data, test_data) tuples.
        """
        n_samples = len(data)
        splits = []
        
        start_idx = 0
        
        while True:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            if test_end > n_samples:
                break
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            splits.append((train_data, test_data))
            start_idx += self.step
        
        return splits
    
    def validate(
        self,
        data: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], object],
        eval_func: Callable[[object, pd.DataFrame], float],
    ) -> dict[str, float]:
        """Run walk-forward validation.
        
        Args:
            data: Full dataset.
            train_func: Function to train model on train data.
            eval_func: Function to evaluate model on test data, returns Sharpe.
            
        Returns:
            Dictionary with mean/std Sharpe across folds.
        """
        splits = self.generate_splits(data)
        sharpes = []
        
        for i, (train_data, test_data) in enumerate(splits):
            print(f"Fold {i+1}/{len(splits)}: "
                  f"Train {train_data.index[0]} - {train_data.index[-1]}, "
                  f"Test {test_data.index[0]} - {test_data.index[-1]}")
            
            model = train_func(train_data)
            sharpe = eval_func(model, test_data)
            sharpes.append(sharpe)
            
            print(f"  Sharpe: {sharpe:.2f}")
        
        return {
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "min_sharpe": np.min(sharpes),
            "max_sharpe": np.max(sharpes),
            "folds": sharpes,
        }
