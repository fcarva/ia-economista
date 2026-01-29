
import sys
import pandas as pd
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher, IpeadataClient

def verify_ipea_client():
    print("\n--- Verifying IpeadataClient ---")
    client = IpeadataClient()
    # Test fetch SELIC (small, recent range)
    df = client.fetch_series('SELIC', '2023-01-01')
    if not df.empty:
        print(f"[OK] IPEA SELIC Fetched: {len(df)} rows")
        print(df.tail())
    else:
        print("[X] IPEA SELIC Failed (Empty)")

def verify_fetch_all():
    print("\n--- Verifying BrazilMacroFetcher.fetch_all() ---")
    fetcher = BrazilMacroFetcher()
    # Use a recent start date to be quick
    df = fetcher.fetch_all(start_date="2023-01-01")
    
    if df.empty:
        print("[X] fetch_all returned empty DataFrame!")
        return
        
    print(f"[OK] Data Fetched: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for NaNs
    nans = df.isnull().sum().sum()
    if nans == 0:
        print("[OK] No NaNs found (Clean Data)")
    else:
        print(f"[!] Found {nans} NaNs!")
        print(df.isnull().sum())
        
    # Check specific columns
    expected = ['selic', 'usd_brl', 'ipca_mom', 'pib_yoy']
    for col in expected:
        if col in df.columns:
            print(f"   - {col}: OK")
        else:
            print(f"   - {col}: MISSING [!]")

if __name__ == "__main__":
    verify_ipea_client()
    verify_fetch_all()
