
import pandas as pd
import warnings
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher, IpeadataClient
try:
    from bcb import sgs
    print("BCB library imported.")
except ImportError:
    print("BCB library NOT found.")

try:
    import sidrapy
    print("Sidrapy library imported.")
except ImportError:
    print("Sidrapy library NOT found.")

def test_bcb():
    print("\nTesting BCB (SGS)...")
    try:
        # Test just one code
        df = sgs.get({'selic': 432}, start='2024-01-01')
        print(f"BCB Result:\n{df.head()}")
    except Exception as e:
        print(f"BCB Failed: {e}")

def test_ipea():
    print("\nTesting IPEA...")
    client = IpeadataClient()
    # Test Selic code from fallback
    df = client.fetch_series('SELIC', '2024-01-01')
    print(f"IPEA Selic Result:\n{df.head()}")
    if not df.empty:
        print(f"Columns: {df.columns}")

def test_fetcher():
    print("\nTesting Full Fetcher...")
    fetcher = BrazilMacroFetcher()
    df = fetcher.fetch_all("2024-01-01")
    print(f"Full Fetcher Result Columns: {df.columns}")
    if 'selic' in df.columns:
        print(f"Selic Head:\n{df['selic'].head()}")
    else:
        print("Selic column convert failed.")

if __name__ == "__main__":
    test_bcb()
    test_ipea()
    test_fetcher()
