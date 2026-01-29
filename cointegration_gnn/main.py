"""
Main Entry Point
================

This module provides the main entry point for the Cointegration GNN
trading system. It demonstrates the full pipeline from data loading
to backtesting.

Usage:
    python -m cointegration_gnn.main
    
    # Or with custom config:
    python -m cointegration_gnn.main --train-start 2016-01-01 --test-end 2024-01-01
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import torch

from .config import Config, default_config
from .data.data_loader import DataLoader, create_data_splits
from .features.cointegration_graph import CointegrationGraph
from .features.causal_discovery import NOTEARS
from .models.gnn_model import CointegrationGNN
from .backtest.backtest_engine import BacktestEngine, WalkForwardValidator
from .workflow import TradingWorkflow


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cointegration GNN Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--train-start",
        type=str,
        default="2015-01-01",
        help="Training period start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2019-12-31",
        help="Training period end date",
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default="2020-01-01",
        help="Validation period start date",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2020-12-31",
        help="Validation period end date",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2021-01-01",
        help="Test period start date",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default="2024-12-31",
        help="Test period end date",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for training",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point.
    
    Pipeline:
    1. Load and preprocess data (OpenBB/YFinance)
    2. Compute rolling cointegration graph
    3. Run causal discovery (NOTEARS)
    4. Train GNN on training period
    5. Validate on COVID period
    6. Backtest on out-of-sample period
    7. Run sanity checks (randomization test)
    """
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("COINTEGRATION GNN TRADING SYSTEM")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Training: {args.train_start} to {args.train_end}")
    print(f"  Validation: {args.val_start} to {args.val_end}")
    print(f"  Testing: {args.test_start} to {args.test_end}")
    print(f"  Device: {args.device}")
    print()
    
    # 1. Load Data
    print("[1/7] Loading data...")
    config = default_config
    
    loader = DataLoader(tickers=config.data.tickers)
    
    prices, returns = loader.fetch_data(
        start=date.fromisoformat(args.train_start),
        end=date.fromisoformat(args.test_end),
    )
    
    print(f"  Loaded {len(prices)} trading days for {len(config.data.tickers)} assets")
    
    # 2. Stationarity Tests
    print("\n[2/7] Running stationarity tests...")
    stationarity_results = loader.test_stationarity(prices)
    
    for ticker, result in stationarity_results.items():
        status = "I(0) Stationary" if result.is_stationary else f"I({result.integration_order})"
        print(f"  {ticker}: {status} (ADF p={result.adf_pvalue:.4f})")
    
    # 3. Compute Features
    print("\n[3/7] Computing features...")
    features = loader.compute_features(prices, returns, lookback=20)
    print(f"  Feature shape: {features.shape}")
    
    # 4. Split Data
    print("\n[4/7] Splitting data (regime-aware)...")
    splits = create_data_splits(
        prices=prices,
        returns=returns,
        features=features,
        train_end=date.fromisoformat(args.train_end),
        val_end=date.fromisoformat(args.val_end),
    )
    
    print(f"  Train: {len(splits['train']['prices'])} days")
    print(f"  Validation: {len(splits['val']['prices'])} days")
    print(f"  Test: {len(splits['test']['prices'])} days")
    
    # 5. Build Cointegration Graph (on training data)
    print("\n[5/7] Building cointegration graph...")
    coint_graph = CointegrationGraph(
        tickers=config.data.tickers,
        rolling_window=config.cointegration.rolling_window,
    )
    
    # Compute adjacency at end of training
    train_end_date = pd.Timestamp(args.train_end)
    adjacency = coint_graph.compute_adjacency(
        splits["train"]["prices"],
        as_of_date=train_end_date,
    )
    
    n_edges = (adjacency > 0).sum() // 2
    print(f"  Edges detected: {n_edges}")
    print(f"  Graph density: {n_edges / (len(config.data.tickers) * (len(config.data.tickers) - 1) / 2):.2%}")
    
    # 6. Run Causal Discovery
    print("\n[6/7] Running NOTEARS causal discovery...")
    notears = NOTEARS(
        lambda_l1=config.causal.lambda_l1,
        max_iter=config.causal.max_iter,
    )
    
    train_returns = splits["train"]["returns"].dropna()
    causal_result = notears.fit(train_returns.values)
    
    print(f"  Causal edges: {causal_result.n_edges}")
    print(f"  Acyclicity violation: {causal_result.acyclicity_violation:.2e}")
    print(f"  Converged: {causal_result.converged}")
    
    # 7. Initialize Model
    print("\n[7/7] Initializing GNN model...")
    model = CointegrationGNN(
        node_feature_dim=config.gnn.node_feature_dim,
        hidden_dim=config.gnn.hidden_dim,
        num_layers=config.gnn.num_layers,
        dropout=config.gnn.dropout,
        learning_rate=config.gnn.learning_rate,
        weight_decay=config.gnn.weight_decay,
        transaction_cost_bps=config.backtest.transaction_cost_bps,
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("""
Next Steps:
1. Train the model using PyTorch Lightning
2. Run walk-forward validation
3. Backtest on out-of-sample data
4. Execute randomization tests

Example:
    from pytorch_lightning import Trainer
    from torch_geometric.loader import DataLoader
    
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, train_loader)
    
    # Run backtest
    engine = BacktestEngine(tickers=config.data.tickers)
    result = engine.run(prices, signal_generator=model.predict_positions)
    print(result.summary())
""")
    
    return


if __name__ == "__main__":
    main()
