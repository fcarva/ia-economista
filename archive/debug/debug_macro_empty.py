
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher
import pandas as pd

def test_fetch():
    print("Testing Macro Fetch with training params...")
    # Matches train.py: fetch_start = default_config.data.train_start - pd.Timedelta(days=700)
    # Assuming train_start is around 2016-01-01 -> fetch_start ~ 2014-02-01
    start_date = "2014-02-01"
    
    fetcher = BrazilMacroFetcher()
    df = fetcher.fetch_all(start_date=start_date)
    
    print(f"Final DF Shape: {df.shape}")
    print(f"Empty? {df.empty}")
    if not df.empty:
        print(df.head())
        print(df.tail())
    else:
        print("DF is EMPTY!")

if __name__ == "__main__":
    test_fetch()
