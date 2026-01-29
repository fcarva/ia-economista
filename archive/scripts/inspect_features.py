"""
Feature Variance Inspection
===========================

Checks if features have adequate variance and are not degenerate.

Usage:
    python scripts/inspect_features.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.config import default_config


def main():
    print("="*60)
    print("FEATURE VARIANCE INSPECTION")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading data...")
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start,
        end=default_config.data.val_end
    )
    
    print(f"      Prices shape: {prices.shape}")
    print(f"      Returns shape: {returns.shape}")
    
    # Compute features
    print("\n[2/3] Computing features...")
    features = loader.compute_features(prices, returns)
    print(f"      Features shape: {features.shape}")
    
    # Analyze variance per feature
    print("\n[3/3] Analyzing feature statistics...")
    
    # Get feature names (second level of MultiIndex)
    feature_names = features.columns.get_level_values(1).unique()
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS (Aggregated Across All Tickers)")
    print("="*60)
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'% NaN':>8}")
    print("-"*76)
    
    issues = []
    for feat in feature_names:
        # Extract all values for this feature across tickers
        feat_data = features.xs(feat, level=1, axis=1).values.flatten()
        
        mean_val = np.nanmean(feat_data)
        std_val = np.nanstd(feat_data)
        min_val = np.nanmin(feat_data)
        max_val = np.nanmax(feat_data)
        nan_pct = np.isnan(feat_data).mean() * 100
        
        print(f"{feat:<20} {mean_val:>12.6f} {std_val:>12.6f} {min_val:>12.6f} {max_val:>12.6f} {nan_pct:>7.1f}%")
        
        # Check for issues
        if std_val < 0.001:
            issues.append(f"⚠️  '{feat}' has very low variance (std={std_val:.6f})")
        if nan_pct > 10:
            issues.append(f"⚠️  '{feat}' has high NaN rate ({nan_pct:.1f}%)")
        if abs(mean_val) > 10:
            issues.append(f"⚠️  '{feat}' has large mean ({mean_val:.2f}) - consider normalization")
    
    # Return correlation check
    print("\n" + "="*60)
    print("RETURN STATISTICS PER TICKER")
    print("="*60)
    print(f"{'Ticker':<12} {'Mean Ret':>12} {'Std Ret':>12} {'Sharpe':>10}")
    print("-"*48)
    
    for ticker in prices.columns:
        ret = returns[ticker]
        mean_ret = ret.mean()
        std_ret = ret.std()
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        print(f"{ticker:<12} {mean_ret:>12.6f} {std_ret:>12.6f} {sharpe:>10.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if issues:
        print("\n⚠️  Potential Issues Found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ All features have adequate variance and low NaN rates.")
    
    # Check if returns are too small
    all_returns = returns.values.flatten()
    ret_std = np.nanstd(all_returns)
    if ret_std < 0.01:
        print(f"\n⚠️  Returns have very low std ({ret_std:.6f}).")
        print("   This may cause vanishing gradients in the model.")
        print("   Consider: scaling returns by 100 or using percentage returns.")
    
    print("="*60)


if __name__ == "__main__":
    main()
