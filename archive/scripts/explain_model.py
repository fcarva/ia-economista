
import sys
from pathlib import Path
import glob

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch

from cointegration_gnn.config import default_config
from cointegration_gnn.models.gnn_model import CointegrationGNN
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph

def load_latest_model():
    model_dir = Path(__file__).parent.parent / "models"
    model_files = sorted(glob.glob(str(model_dir / "*.ckpt")), reverse=True)
    if not model_files:
        raise FileNotFoundError("No trained models found in models/")
    print(f"Loading model: {model_files[0]}")
    return CointegrationGNN.load_from_checkpoint(model_files[0])

def get_feature_names():
    # From data_loader.py compute_features
    return [
        "Return (1d)", 
        "Momentum (5d)", 
        "Volatility (20d)", 
        "Z-Score (20d)", 
        "RSI (14d)", 
        "Spread vs Mean", 
        "Lag Return (t-1)", 
        "Lag Return (t-2)"
    ]

def analyze_feature_importance(model, data_sample, feature_names):
    """
    Compute Feature Importance using Gradient * Input.
    Sensitivity Analysis: How much does changing a feature change the output?
    """
    model.eval()
    
    # Enable gradients for input features
    x = data_sample.x.clone().detach().requires_grad_(True)
    edge_index = data_sample.edge_index
    edge_attr = data_sample.edge_attr
    
    # Forward pass
    # Verify shape of edge_attr
    if edge_attr is not None and edge_attr.dim() == 2:
        edge_attr = edge_attr.squeeze(-1)
        
    embeddings = model.encoder(x, edge_index, edge_attr)
    positions = model.position_head(embeddings)
    
    # We want to explain the MAGNITUDE of positions (conviction)
    # Target: mean absolute position
    target = positions.abs().mean()
    
    # Backward pass
    target.backward()
    
    # Feature Importance = |Gradient * Input|
    # Averaged across all nodes in the batch
    grads = x.grad
    importance = (grads * x).abs().mean(dim=0).detach().cpu().numpy()
    
    # Normalize to %
    importance_pct = importance / importance.sum() * 100
    
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": importance_pct
    }).sort_values("Importance (%)", ascending=False)

def analyze_sage_weights(model):
    """
    Analyze inner weights of GraphSAGE to estimate Network vs Self influence.
    SAGEConv: x_new = W_self * x + W_neigh * mean(neighbors)
    """
    # PyG SAGEConv usually has lin_l (neighbors) and lin_r (self/root)
    # Check first layer
    conv1 = model.encoder.convs[0]
    
    # Compute norms of weight matrices
    if hasattr(conv1, 'lin_l') and hasattr(conv1, 'lin_r'):
         # lin_l is for neighbors, lin_r is for root (self)
         # Note: In some PyG versions, it might be combined.
         # For SAGEConv(in, out), if project=False, we might have specific structure.
         # Let's inspect weights safely.
         
         w_neigh = conv1.lin_l.weight.detach().abs().mean().item()
         w_self = conv1.lin_r.weight.detach().abs().mean().item()
         
    else:
        # Some versions use a single linear layer
        print("Warning: Could not inspect SAGE weights directly (PyG version diff).")
        return 50.0, 50.0

    total = w_neigh + w_self
    return (w_self / total) * 100, (w_neigh / total) * 100

def main():
    print("🔍 AI Economist: Model Interpretability Module")
    print("==============================================")
    
    # 1. Load Model
    try:
        model = load_latest_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Please train a model first using scripts/train.py")
        return

    # 2. Get Data Sample (Validation Set)
    print("Fetching validation data sample...")
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(start=default_config.data.val_start, end=default_config.data.val_end)
    features = loader.compute_features(prices, returns)
    
    # Build one graph (latest validation date)
    coint = CointegrationGraph(tickers=default_config.data.tickers)
    date = features.index[-1]
    adj = coint.compute_adjacency(prices, date)
    
    x_rows = [features.loc[date][t].values for t in default_config.data.tickers]
    x = np.stack(x_rows)
    
    data = coint.to_pyg_data(adj, x)
    
    # 3. Feature Attribution (Gradient x Input)
    feat_names = get_feature_names()
    print("\nCalculating Feature Importance (Gradient * Input)...")
    imp_df = analyze_feature_importance(model, data, feat_names)
    
    print("\n📊 Feature Importance Ranking:")
    print(imp_df.to_string(index=False))
    
    # 4. Network Influence Analysis
    print("\nCalculating Network vs Self Influence...")
    try:
        self_pct, neigh_pct = analyze_sage_weights(model)
        print(f"Self-Signal Influence:    {self_pct:.1f}%")
        print(f"Network (Coint) Influence: {neigh_pct:.1f}%")
        
        if neigh_pct > 30:
            print("✅ Model is effectively using the Cointegration Graph!")
        else:
            print("⚠️ Model is dominating by self-features (momentum/reversion).")
            
    except Exception as e:
        print(f"Skipped weight analysis: {e}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance (%)", y="Feature", palette="viridis")
    plt.title("GNN Feature Importance (Gradient Analysis)")
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "outputs" / "feature_importance.png"
    plt.savefig(output_path)
    print(f"\nSaved plot to: {output_path}")

if __name__ == "__main__":
    main()
