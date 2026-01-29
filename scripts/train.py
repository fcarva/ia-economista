
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler

from cointegration_gnn.config import default_config
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher
from cointegration_gnn.data.macro_data import add_macro_to_node_features
from cointegration_gnn.features.cointegration_graph import CointegrationGraph
from cointegration_gnn.features.causal_discovery import GrangerLassoDiscovery
from cointegration_gnn.models.gnn_model import CointegrationGNN

def prepare_dataset(prices, returns, features, coint_graph, tickers):
    """Prepare PyTorch Geometric dataset from time series."""
    print("Computing rolling adjacency matrices...")
    # [OPTIMIZATION] Disk Cache for Adjacency
    import pickle
    import os
    adj_cache_path = "data/adj_cache.pkl"
    
    if os.path.exists(adj_cache_path):
        print(f"      Loading adjacency cache from {adj_cache_path}...")
        with open(adj_cache_path, "rb") as f:
            adj_cache = pickle.load(f)
    else:
        # Recompute graph every 20 days (approx monthly) for efficiency
        adj_cache = coint_graph.compute_rolling_adjacencies(prices, recompute_freq=20)
        print(f"      Saving adjacency cache to {adj_cache_path}...")
        with open(adj_cache_path, "wb") as f:
            pickle.dump(adj_cache, f)
    
    dataset = []
    
    # Align dates
    common_dates = features.index.intersection(returns.index).intersection(prices.index[coint_graph.rolling_window:])
    
    print(f"Constructing {len(common_dates)} graph samples...")
    
    for date in common_dates:
        # 1. Get Adjacency
        # Find latest available adjacency (forward fill logic if daily not computed)
        # Using searchsorted or simple loop. Since we have a cache dict:
        # We need the adjacency valid for this date.
        # coint_graph.compute_rolling_adjacencies returns dict[date] -> adj
        # where date is the 'as_of' date.
        # If we are at 'date', we want graph constructed from info strictly BEFORE 'date'.
        
        # However, compute_rolling_adjacencies keys are the dates we computed FOR.
        # If we recompute every 5 days, we might not have 'date' key.
        # We need the most recent key <= date.
        
        closest_date = max([d for d in adj_cache.keys() if d <= date], default=None)
        if closest_date is None:
            continue
            
        adjacency = adj_cache[closest_date]
        
        # 2. Get Node Features
        # features.loc[date] contains features known at start of 'date' (shifted)
        x_np = features.loc[date].values
        # Reshape if MultiIndex (n_tickers, n_features)
        # features columns are (ticker, feature).
        # We need to ensure correct ordering of tickers matching adjacency.
        # Pivot or extraction:
        x_rows = []
        for ticker in tickers:
            x_rows.append(features.loc[date][ticker].values)
        x = np.stack(x_rows) # (n_assets, n_features)
        
        # 3. Get Target (Returns for 'date')
        # We want to predict return of 'date'.
        y_np = returns.loc[date, tickers].values
        
        # 4. Create Data Object
        data = coint_graph.to_pyg_data(adjacency, x)
        data.y = torch.tensor(y_np, dtype=torch.float)
        data.tickers = tickers
        data.date = date
        
        dataset.append(data)
        
    return dataset

def main():
    # [NEW] Argument Parsing
    parser = argparse.ArgumentParser(description="Train Cointegration GNN")
    parser.add_argument("--turnover_penalty", type=float, default=0.5, help="Penalty for portfolio turnover")
    parser.add_argument("--learning_rate", type=float, default=default_config.gnn.learning_rate, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="gnn_ibov_experiment", help="Name for the log folder")
    parser.add_argument("--max_epochs", type=int, default=80, help="Maximum training epochs")
    parser.add_argument("--min_epochs", type=int, default=10, help="Minimum training epochs before early stop")
    parser.add_argument("--early_stop_patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    # Date Args
    parser.add_argument("--train_start", type=str, default=str(default_config.data.train_start), help="Train start date (YYYY-MM-DD)")
    parser.add_argument("--train_end", type=str, default=str(default_config.data.train_end), help="Train end date (YYYY-MM-DD)")
    parser.add_argument("--val_start", type=str, default=str(default_config.data.val_start), help="Val start date (YYYY-MM-DD)")
    parser.add_argument("--val_end", type=str, default=str(default_config.data.val_end), help="Val end date (YYYY-MM-DD)")
    
    args = parser.parse_args()

    print(f"🚀 Starting Training Pipeline with Params: TP={args.turnover_penalty}, LR={args.learning_rate}...")
    print(f"      Dates: Train[{args.train_start} to {args.train_end}], Val[{args.val_start} to {args.val_end}]")
    
    # 1. Fetch Data
    print("Fetching data...")
    loader = DataLoader(tickers=default_config.data.tickers)
    
    # Parse dates
    train_start = pd.Timestamp(args.train_start).date()
    train_end = pd.Timestamp(args.train_end).date()
    val_start = pd.Timestamp(args.val_start).date()
    val_end = pd.Timestamp(args.val_end).date()
    
    fetch_start = train_start - pd.Timedelta(days=700)
    prices, returns = loader.fetch_data(start=fetch_start, end=val_end)
    features = loader.compute_features(prices, returns)
    
    # 1. RANK TRANSFORM
    print("Transformando features técnicas via Ranking Transversal...")
    features = loader.rank_transform_features(features, output_range=(-0.5, 0.5))
    
    # 1b. Fetch and Merge Macro Features
    print("Fetching macro features (Brazil: BCB/IBGE)...")
    macro_fetcher = BrazilMacroFetcher(cache_path="data/brazil_macro.csv")
    macro_df = macro_fetcher.fetch_all(start_date=fetch_start.strftime("%Y-%m-%d"))
    
    if not macro_df.empty:
        print("Normalizing Macro Features (StandardScaler)...")
        macro_scaler = StandardScaler()
        cols_to_norm = macro_df.select_dtypes(include=[np.number]).columns
        if len(cols_to_norm) > 0:
            macro_df[cols_to_norm] = macro_scaler.fit_transform(macro_df[cols_to_norm])
        features = add_macro_to_node_features(features, macro_df)
    
    # 2. Split Data
    train_mask = (features.index >= pd.Timestamp(train_start)) & \
                 (features.index <= pd.Timestamp(train_end))
    val_mask = (features.index >= pd.Timestamp(val_start)) & \
               (features.index <= pd.Timestamp(val_end))
    
    # 3. Build Graphs & Dataset
    coint_graph = CointegrationGraph(
        tickers=default_config.data.tickers,
        rolling_window=default_config.cointegration.rolling_window
    )
    
    print("Preparing Datasets...")
    train_dataset = prepare_dataset(prices, returns, features[train_mask], coint_graph, default_config.data.tickers)
    val_dataset = prepare_dataset(prices, returns, features[val_mask], coint_graph, default_config.data.tickers)
    
    train_loader = PyGLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = PyGLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 4. Initialize Model
    actual_node_feature_dim = train_dataset[0].x.shape[1]
    print(f"      Detected node_feature_dim: {actual_node_feature_dim}")
    
    model = CointegrationGNN(
        node_feature_dim=actual_node_feature_dim,
        hidden_dim=default_config.gnn.hidden_dim,
        num_layers=default_config.gnn.num_layers,
        dropout=default_config.gnn.dropout,
        learning_rate=args.learning_rate,       # [NEW] Use args
        weight_decay=default_config.gnn.weight_decay,
        transaction_cost_bps=10.0,
        turnover_penalty=args.turnover_penalty, # [NEW] Use args
        use_ranking_loss=True,
    )
    
    # 5. Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val/IC',
        mode='max',
        dirpath='models/',
        filename=f'gnn-ibov-tp{args.turnover_penalty}-lr{args.learning_rate}-{{epoch:02d}}-{{val_IC:.4f}}', # [NEW] Informative filename
        save_top_k=1
    )
    
    early_stop = EarlyStopping(monitor='val/return', patience=args.early_stop_patience, mode='max')
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=CSVLogger("logs", name=args.experiment_name), # [NEW] Use args
        log_every_n_steps=10,
        gradient_clip_val=args.gradient_clip,
    )
    
    # 6. Fit
    trainer.fit(model, train_loader, val_loader)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
