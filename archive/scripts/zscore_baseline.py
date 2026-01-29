"""
Z-Score Threshold Baseline Strategy
====================================

Simple baseline: buy when z-score < -2, sell when z-score > 2.
Tests if there is ANY alpha in the data before blaming the GNN.

Usage:
    python scripts/zscore_baseline.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.config import default_config


def zscore_strategy(prices: pd.DataFrame, window: int = 20, entry: float = 2.0) -> pd.DataFrame:
    """Generate positions based on z-score mean reversion.
    
    Args:
        prices: DataFrame of adjusted close prices.
        window: Rolling window for mean/std calculation.
        entry: Z-score threshold for entry (symmetric).
        
    Returns:
        DataFrame of positions in range [-1, 1].
    """
    # Compute rolling z-score for each asset
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    zscore = (prices - rolling_mean) / rolling_std
    
    # Generate signals:
    # z < -entry: oversold, buy (+1)
    # z > +entry: overbought, sell (-1)
    # otherwise: hold (0)
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    positions[zscore < -entry] = 1.0   # Buy oversold
    positions[zscore > entry] = -1.0   # Sell overbought
    
    return positions


def backtest_zscore(prices: pd.DataFrame, positions: pd.DataFrame, tc_bps: float = 10.0) -> dict:
    """Run simple backtest on z-score strategy.
    
    Returns:
        Dictionary with performance metrics.
    """
    # Compute log returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Align positions (shift to avoid look-ahead bias)
    positions = positions.shift(1).dropna()
    
    # Common index
    common_idx = returns.index.intersection(positions.index)
    returns = returns.loc[common_idx]
    positions = positions.loc[common_idx]
    
    # Portfolio return = sum(position * return) / num_assets (equal weight)
    portfolio_returns = (positions * returns).sum(axis=1) / len(prices.columns)
    
    # Transaction costs
    turnover = positions.diff().abs().sum(axis=1)
    tc = (tc_bps / 10000) * turnover
    net_returns = portfolio_returns - tc
    
    # Metrics
    total_return = (1 + net_returns).prod() - 1
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = (1 + net_returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Avg turnover
    avg_turnover = turnover.mean()
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_daily_turnover": avg_turnover,
        "n_trades": (positions.diff().abs() > 0).sum().sum(),
    }


def main():
    print("="*60)
    print("Z-SCORE BASELINE STRATEGY TEST")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading data...")
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start,
        end=default_config.data.val_end
    )
    print(f"      Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"      Days: {len(prices)}")
    
    # Generate positions
    print("\n[2/3] Running z-score strategy...")
    positions = zscore_strategy(prices, window=20, entry=2.0)
    
    # Count signals
    long_signals = (positions > 0).sum().sum()
    short_signals = (positions < 0).sum().sum()
    neutral = (positions == 0).sum().sum()
    print(f"      Long signals:  {long_signals:,}")
    print(f"      Short signals: {short_signals:,}")
    print(f"      Neutral:       {neutral:,}")
    
    # Backtest
    print("\n[3/3] Running backtest...")
    results = backtest_zscore(prices, positions, tc_bps=10.0)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS (Z-Score Mean Reversion)")
    print("="*60)
    print(f"""
    Strategy: Buy z < -2, Sell z > +2, Window = 20d
    ─────────────────────────────────────────────────
    Total Return:      {results['total_return']:>10.2%}
    Sharpe Ratio:      {results['sharpe']:>10.2f}
    Max Drawdown:      {results['max_drawdown']:>10.2%}
    Avg Daily Turnover:{results['avg_daily_turnover']:>10.2%}
    Total Trades:      {results['n_trades']:>10,}
    """)
    
    # Interpretation
    print("\n[Interpretation]")
    if results['sharpe'] > 0.5:
        print("✅ Alpha exists! Z-score strategy has positive Sharpe.")
        print("   The GNN should be able to learn this signal.")
    elif results['sharpe'] > 0:
        print("⚠️  Weak alpha. Z-score has marginal positive Sharpe.")
        print("   Consider adding more features or different assets.")
    else:
        print("❌ No alpha detected. Z-score strategy is unprofitable.")
        print("   The problem is in the data, not the model.")
    
    print("="*60)
    return results


if __name__ == "__main__":
    main()
