
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cointegration_gnn.config import default_config
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph

def main():
    print("1. Testing Data Fetching...")
    loader = DataLoader(tickers=default_config.data.tickers)
    # Fetch small chunk
    start = default_config.data.train_start
    end = default_config.data.train_end
    print(f"Fetching {start} to {end}...")
    prices, returns = loader.fetch_data(start=start, end=end)
    print(f"Prices shape: {prices.shape}")
    print("Data Fetch OK.")
    
    print("\n2. Testing Feature Computation...")
    features = loader.compute_features(prices, returns)
    print(f"Features shape: {features.shape}")
    print("Features OK.")
    
    print("\n3. Testing Cointegration Graph (One Day)...")
    coint = CointegrationGraph(tickers=default_config.data.tickers)
    adj = coint.compute_adjacency(prices, prices.index[-1])
    print(f"Adjacency shape: {adj.shape}")
    print(f"Edges: {np.sum(adj > 0) / 2}")
    
    print("\n4. Testing Rolling Graph (First 5 days)...")
    # Just compute for first few valid days
    valid_dates = prices.index[coint.rolling_window:][:5]
    for date in valid_dates:
        print(f"Computing for {date}...")
        coint.compute_adjacency(prices, date)
    print("Rolling Graph OK.")

import numpy as np
if __name__ == "__main__":
    main()
