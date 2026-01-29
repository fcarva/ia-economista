
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph
from cointegration_gnn.models.gnn_model import CointegrationGNN
from cointegration_gnn.config import default_config
from torch_geometric.loader import DataLoader as PyGLoader

def prepare_dataset(prices, returns, features, coint_graph, tickers):
    # Recompute adjacency using the cache to save time if already computed
    adj_cache = coint_graph.compute_rolling_adjacencies(prices, recompute_freq=5)
    
    dataset = []
    # Just take the last 100 dates for debugging
    common_dates = features.index.intersection(returns.index).intersection(prices.index[coint_graph.rolling_window:])[-100:]
    
    print(f"Constructing {len(common_dates)} graph samples for debugging...")
    
    for date in common_dates:
        closest_date = max([d for d in adj_cache.keys() if d <= date], default=None)
        if closest_date is None:
            continue
            
        adjacency = adj_cache[closest_date]
        x_rows = []
        for ticker in tickers:
            x_rows.append(features.loc[date][ticker].values)
        x = np.stack(x_rows)
        y_np = returns.loc[date, tickers].values
        
        data = coint_graph.to_pyg_data(adjacency, x)
        data.y = torch.tensor(y_np, dtype=torch.float)
        dataset.append(data)
        
    return dataset

def main():
    print("="*60)
    print("DEBUG: Checking Scale of Loss Components")
    print("="*60)
    
    # 1. Load Data
    loader = DataLoader(tickers=default_config.data.tickers)
    # Using cached data if available
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start,
        end=default_config.data.val_end
    )
    features = loader.compute_features(prices, returns)
    
    # 2. Prepare small batch
    coint_graph = CointegrationGraph(tickers=default_config.data.tickers, rolling_window=60)
    dataset = prepare_dataset(prices, returns, features, coint_graph, default_config.data.tickers)
    loader = PyGLoader(dataset, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    
    # 3. Load Model
    # Try to find latest checkpoint
    model_dir = Path("models")
    ckpts = list(model_dir.glob("*.ckpt"))
    if ckpts:
        latest = max(ckpts, key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest.name}")
        model = CointegrationGNN.load_from_checkpoint(str(latest))
    else:
        print("No checkpoint found, initializing new model")
        model = CointegrationGNN()
        
    model.eval()
    
    # 4. Run Forward Pass & Inspect Values
    print("\n--- Forward Pass Inspection ---")
    
    with torch.no_grad():
        positions = model(batch)
        
        # Reshape like in training_step
        num_graphs = batch.num_graphs
        num_nodes = positions.shape[0] // num_graphs
        pos_matrix = positions.view(num_graphs, num_nodes)
        
        returns = batch.y
        ret_matrix = returns.view(num_graphs, num_nodes)
        
        # Portfolio returns
        portfolio_returns = (pos_matrix * ret_matrix).sum(dim=1)
        
        # Turnover
        turnover = torch.abs(pos_matrix[1:] - pos_matrix[:-1]).sum(dim=1)
        cost = model.transaction_cost * turnover
        net_returns = portfolio_returns[1:] - cost
        
        # Loss components
        sharpe_loss = model.differentiable_sharpe(net_returns)
        mag_penalty = 0.01 * (positions ** 2).mean()
        
        turnover_loss = torch.abs(pos_matrix[1:] - pos_matrix[:-1]).mean()
        penalty_val = model.turnover_penalty * turnover_loss
        
    # 5. Print Statistics
    print(f"\nModel Turnover Penalty Setting: {model.turnover_penalty}")
    print(f"Transaction Cost Setting:     {model.transaction_cost} (decimal)")

    print("\n[Magnitudes]")
    print(f"Asset Returns (Mean Abs):     {ret_matrix.abs().mean():.6f}")
    print(f"Positions (Mean Abs):         {pos_matrix.abs().mean():.6f}")
    print(f"Positions (Max):              {pos_matrix.max():.6f}")
    print(f"Positions (Min):              {pos_matrix.min():.6f}")
    
    print("\n[Return Components]")
    print(f"Raw Portfolio Ret (Mean):     {portfolio_returns.mean():.6f}")
    print(f"Raw Portfolio Ret (Std):      {portfolio_returns.std():.6f}")
    print(f"Turnover (Sum per step):      {turnover.mean():.6f}")
    print(f"Tx Cost (Mean):               {cost.mean():.6f}")
    print(f"Net Return (Mean):            {net_returns.mean():.6f}")
    
    print("\n[Loss Term Comparison]")
    print(f"Sharpe Term (Magnitude):      {abs(sharpe_loss):.6f}")
    print(f"Magnitude Reg Term:           {mag_penalty:.6f}")
    print(f"Turnover Term (Raw Loss):     {turnover_loss:.6f}")
    print(f"Turnover Term (Weighted):     {penalty_val:.6f}")
    
    print("\n[Ratio]")
    if penalty_val > 0:
        ratio = abs(sharpe_loss) / penalty_val
        print(f"Sharpe / Penalty Ratio:       {ratio:.2f}x (Sharpe is {ratio:.1f} times larger)")
    else:
        print("Penalty is zero.")
        
    print("\n[Conclusion]")
    if abs(portfolio_returns.mean()) < 1e-4:
        print("⚠️  WARNING: Portfolio returns are extremely small. Signal might be drowned out.")
    if cost.mean() > abs(portfolio_returns.mean()):
        print("⚠️  WARNING: Transaction costs exceed raw returns!")
    
if __name__ == "__main__":
    main()
