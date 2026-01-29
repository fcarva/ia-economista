
import sys
from pathlib import Path
import glob
import torch
import numpy as np
from torch_geometric.data import Data

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cointegration_gnn.models.gnn_model import CointegrationGNN

def load_latest_model():
    model_dir = Path(__file__).parent.parent / "models"
    model_files = sorted(glob.glob(str(model_dir / "*.ckpt")), reverse=True)
    if not model_files:
        raise FileNotFoundError("No trained models found")
    print(f"Loading: {model_files[0]}")
    return CointegrationGNN.load_from_checkpoint(model_files[0])

def test_logic():
    model = load_latest_model()
    model.eval()
    
    # Feature indices from data_loader.py:
    # 0: return_1d
    # 1: return_5d
    # 2: volatility_20d
    # 3: zscore_20d
    # 4: rsi_14d
    # 5: spread_from_mean
    # ...
    
    print("\n--- Logic Test: High Z-Score (Overbought) ---")
    # Scenario: Asset is 3-sigma expensive (High Z-Score). 
    # Mean Reversion Logic: Should SELL (Output < 0).
    # Momentum Logic: Should BUY (Output > 0).
    
    x = torch.zeros(1, 8) # 1 node, 8 features
    x[0, 3] = 3.0 # Z-Score = +3.0
    
    # Empty graph (self-loop only implicitly handled by SAGE if configured)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1))
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    with torch.no_grad():
        embeddings = model.encoder(data.x, data.edge_index, data.edge_attr)
        position = model.position_head(embeddings)
        
    print(f"Input Z-Score: +3.0")
    print(f"Model Output: {position.item():.4f}")
    
    if position.item() > 0:
        print("Diagnosis: MOMENTUM (High Price -> Buy) 🔴")
        print("This loses money in mean-reverting markets.")
    else:
        print("Diagnosis: MEAN REVERSION (High Price -> Sell) 🟢")
        print("This is correct for pairs trading.")

    print("\n--- Logic Test: Low RSI (Oversold) ---")
    # Feature 4: RSI. Low RSI (<30) usually means Oversold -> Buy.
    x_rsi = torch.zeros(1, 8)
    x_rsi[0, 4] = 20.0 # RSI = 20
    # Note: If input features were scaled/normalized, raw 20 might be huge.
    # Assuming standard scaler or raw? DataLoader returns raw for RSI (0-100).
    # Actually, verify data_loader scaling!
    
    data_rsi = Data(x=x_rsi, edge_index=edge_index, edge_attr=edge_attr)
    with torch.no_grad():
        pos_rsi = model.position_head(model.encoder(data_rsi.x, data_rsi.edge_index, data_rsi.edge_attr))
        
    print(f"Input RSI: 20 (Oversold)")
    print(f"Model Output: {pos_rsi.item():.4f}")
    
if __name__ == "__main__":
    test_logic()
    
