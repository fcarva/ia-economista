
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher
import pandas as pd

def test_duplicates():
    print("Testing Macro Fetch for duplicates...")
    start_date = "2014-02-01"
    
    fetcher = BrazilMacroFetcher()
    df = fetcher.fetch_all(start_date=start_date)
    
    print(f"Final DF Shape: {df.shape}")
    
    if df.index.has_duplicates:
        print("!!! DF HAS DUPLICATES !!!")
        print(df.index[df.index.duplicated()].unique())
    else:
        print("Index is unique.")

if __name__ == "__main__":
    test_duplicates()
