"""
Backtest Runner with New Turnover-Penalized Model
==================================================

Runs a complete backtest using the latest trained model and
compares performance metrics.

Usage:
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from datetime import datetime, date

from cointegration_gnn.config import default_config
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher
from cointegration_gnn.validation.bootstrap import BootstrapValidator
from cointegration_gnn.data.macro_data import MacroDataFetcher, add_macro_to_node_features
from cointegration_gnn.features.cointegration_graph import CointegrationGraph
from cointegration_gnn.models.gnn_model import CointegrationGNN
from cointegration_gnn.strategy.strategy import Strategy


def find_latest_model():
    """Find the most recently modified checkpoint."""
    model_dir = Path("models")
    ckpts = list(model_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint files found in models/")
    
    # Sort by modification time
    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    return latest


def detect_encoder_type(model_path: str) -> str:
    """Detect encoder type from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Check for GATv2 signature (has 'att' key for attention)
    for key in state_dict.keys():
        if 'encoder.convs.0.att' in key:
            return "gatv2"
    
    # Otherwise it's SAGE
    return "sage"


def run_backtest(model, prices, returns, features, coint_graph, tickers, transaction_cost_bps=10.0):
    """Run backtest with GNN model signals."""
    
    strategy = Strategy(
        tickers=tickers,
        kelly_fraction=0.50,  # Aumentado para buscar mais retorno vs Benchmark
        max_position=0.30,    # Aumentado limite individual
    )
    
    # Compute rolling adjacencies
    adj_cache = coint_graph.compute_rolling_adjacencies(prices, recompute_freq=5)
    
    # Start after warm-up period
    start_idx = coint_graph.rolling_window + 20
    dates = prices.index[start_idx:]
    
    nav_history = [1.0]
    positions_history = []
    current_weights = {t: 0.0 for t in tickers}
    
    for i, date in enumerate(dates):
        # Get adjacency
        closest_date = max([d for d in adj_cache.keys() if d <= date], default=None)
        if closest_date is None:
            nav_history.append(nav_history[-1])
            continue
        
        adjacency = adj_cache[closest_date]
        
        # Get features for this date
        try:
            x_rows = []
            for ticker in tickers:
                x_rows.append(features.loc[date][ticker].values)
            x = np.stack(x_rows)
        except (KeyError, ValueError):
            nav_history.append(nav_history[-1])
            continue
        
        # Create PyG data
        data = coint_graph.to_pyg_data(adjacency, x)
        
        # Get model predictions
        with torch.no_grad():
            signals = model(data).numpy().flatten()
        
        # Map to dict
        signal_dict = dict(zip(tickers, signals))
        
        # Generate target weights using Kelly
        lookback_returns = returns.loc[:date].tail(60)
        target_weights = strategy.generate_target_weights(signal_dict, lookback_returns)
        
        # Compute turnover
        turnover = sum(abs(target_weights.get(t, 0) - current_weights.get(t, 0)) for t in tickers)
        
        # Apply transaction costs
        tx_cost = (transaction_cost_bps / 10000) * turnover
        
        # Get returns for this day
        day_returns = returns.loc[date, tickers]
        
        # Portfolio return
        port_return = sum(current_weights.get(t, 0) * day_returns[t] for t in tickers) - tx_cost
        
        # Update NAV
        new_nav = nav_history[-1] * (1 + port_return)
        nav_history.append(new_nav)
        
        # Update weights
        current_weights = target_weights.copy()
        positions_history.append({**current_weights, 'date': date, 'turnover': turnover})
    
    nav = pd.Series(nav_history[1:], index=dates)
    positions = pd.DataFrame(positions_history)
    
    return nav, positions


def main():
    print("=" * 70)
    print("BACKTEST RUNNER - Turnover Penalty Model")
    print("=" * 70)
    
    tickers = default_config.data.tickers
    
    # 1. Find latest model
    model_path = find_latest_model()
    print(f"\n[1/5] Loading model: {model_path.name}")
    
    encoder_type = detect_encoder_type(str(model_path))
    print(f"      Encoder type: {encoder_type}")
    
    # Load model with correct encoder
    model = CointegrationGNN.load_from_checkpoint(
        str(model_path),
        encoder_type=encoder_type
    )
    model.eval()
    turnover_penalty = model.hparams.get('turnover_penalty', 'N/A')
    print(f"      Turnover penalty: {turnover_penalty}")
    
    # 2. Fetch test data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2023-01-01")
    parser.add_argument("--end_date", type=str, default="2023-12-31")
    args = parser.parse_args()

    print(f"[2/5] Fetching test data ({args.start_date} to {args.end_date})...")
    
    loader = DataLoader(
        tickers=default_config.data.tickers
    )
    try:
        prices, returns = loader.fetch_data(
            start=pd.Timestamp(args.start_date).date(), 
            end=pd.Timestamp(args.end_date).date(),
            cache_path="data/prices.csv"
        )
    except Exception as e:
        print(f"CRITICAL ERROR in fetch_data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # Explicitly slice to ensure OOS
    prices = prices.loc[args.start_date:args.end_date]
    returns = returns.loc[args.start_date:args.end_date]
    
    features = loader.compute_features(prices, returns)
    
    # Add macro features (same as training)
    # Add macro features (Brazil: BCB/IBGE)
    print("      Fetching macro features (Brazil: BCB/IBGE)...")
    macro_fetcher = BrazilMacroFetcher(cache_path="data/brazil_macro.csv")
    macro_df = macro_fetcher.fetch_all(start_date="2023-01-01")  # Backtest year
    features = add_macro_to_node_features(features, macro_df)
    
    # 2b. NORMALIZE FEATURES (using scaler fit on historical data)
    # Fetch historical data to create scaler with same distribution as training
    print("      Fetching historical data for scaler...")
    hist_prices, hist_returns = loader.fetch_data(
        start=date(2016, 1, 1),
        end=date(2020, 12, 31),
        use_cache=True,
        cache_path="data/prices_hist.csv"
    )
    hist_features = loader.compute_features(hist_prices, hist_returns)
    hist_macro = macro_fetcher.fetch_all(start_date="2016-01-01")
    # Filter to historical period
    hist_macro = hist_macro.loc["2016-01-01":"2020-12-31"]
    hist_features = add_macro_to_node_features(hist_features, hist_macro)
    
    # Fit scaler on historical (train) data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(hist_features)
    
    # Apply to test features
    features = pd.DataFrame(
        scaler.transform(features),
        index=features.index,
        columns=features.columns
    )
    print(f"      ✅ Features normalizadas com scaler histórico")
    
    print(f"      Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"      Days: {len(prices)}")
    print(f"      Features per node: {features.shape[1] // len(tickers)}")
    
    # 3. Build graph
    print("\n[3/5] Building cointegration graph...")
    coint_graph = CointegrationGraph(
        tickers=tickers,
        rolling_window=60
    )
    
    # 4. Run backtest
    print("\n[4/5] Running backtest with GNN strategy...")
    
    nav, positions = run_backtest(
        model=model,
        prices=prices,
        returns=returns,
        features=features,
        coint_graph=coint_graph,
        tickers=tickers,
        transaction_cost_bps=10.0,
    )
    
    # 5. Calculate metrics
    print("\n[5/5] Calculating performance metrics...")
    
    # Returns
    strat_returns = nav.pct_change().dropna()
    total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
    
    # Sharpe
    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252) if strat_returns.std() > 0 else 0
    
    # Max Drawdown
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()
    
    # Turnover
    if 'turnover' in positions.columns:
        avg_turnover = positions['turnover'].mean()
        total_turnover = positions['turnover'].sum()
    else:
        avg_turnover = 0
        total_turnover = 0
    
    # Benchmark (Buy & Hold IBOV proxy = equal weight)
    bh_nav = (1 + returns.mean(axis=1)).cumprod()
    bh_return = (bh_nav.iloc[-1] / bh_nav.iloc[0]) - 1
    bh_std = returns.mean(axis=1).std()
    bh_sharpe = returns.mean(axis=1).mean() / bh_std * np.sqrt(252) if bh_std > 0 else 0
    
    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"""
    📈 GNN Strategy (with Turnover Penalty = {turnover_penalty}):
       ────────────────────────────────────
       Total Return:      {total_return:>10.2%}
       Sharpe Ratio:      {sharpe:>10.2f}
       Max Drawdown:      {max_dd:>10.2%}
       Avg Daily Turnover:{avg_turnover:>10.2%}
       Total Turnover:    {total_turnover:>10.1f}x
    
    📊 Benchmark (Equal Weight):
       ────────────────────────────────────
       Total Return:      {bh_return:>10.2%}
       Sharpe Ratio:      {bh_sharpe:>10.2f}
    
    📌 Alpha (vs Benchmark): {total_return - bh_return:>+.2%}
    """)
    
    # Validation
    print("\n[Validation] Running Stationary Bootstrap (1000 samples)...")
    validator = BootstrapValidator(n_samples=1000, seed=42)
    boot_res = validator.validate(strat_returns)
    print("=" * 70)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save NAV
    nav.to_pickle(output_dir / f"backtest_nav_{timestamp}.pkl")
    
    # Save summary
    summary = {
        "model": model_path.name,
        "encoder_type": encoder_type,
        "turnover_penalty": turnover_penalty if isinstance(turnover_penalty, (int, float)) else 0,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_daily_turnover": avg_turnover,
        "benchmark_return": bh_return,
        "alpha": total_return - bh_return,
    }
    
    pd.Series(summary).to_json(output_dir / f"backtest_summary_{timestamp}.json")
    
    print(f"    Results saved to: results/backtest_*_{timestamp}.*")
    print("=" * 70)
    
    return nav, positions


if __name__ == "__main__":
    main()
