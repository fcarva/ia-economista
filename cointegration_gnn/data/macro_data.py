"""
Macro Data Fetcher
==================

Fetches macroeconomic indicators for use as additional features:
- SELIC: Brazilian interest rate (via BCB API or proxy)
- USD/BRL: Currency exchange rate
- VIX: Volatility index (fear gauge)

These features provide macroeconomic context that can improve
cointegration-based trading signals.
"""

from datetime import date
from typing import Literal
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None


class MacroDataFetcher:
    """Fetches macroeconomic data for feature enrichment.
    
    Tickers used:
        - USD/BRL: "USDBRL=X" (Yahoo Finance)
        - VIX: "^VIX" (Yahoo Finance)
        - SELIC: Approximated via short-term Brazilian bond ETF or BCB API
        
    Example:
        >>> fetcher = MacroDataFetcher()
        >>> macro_df = fetcher.fetch_all(start=date(2015, 1, 1), end=date(2023, 12, 31))
        >>> print(macro_df.columns)
        # Index(['usdbrl', 'usdbrl_return', 'usdbrl_vol', 'vix', 'vix_change', ...])
    """
    
    # Yahoo Finance tickers for macro data
    TICKERS = {
        "usdbrl": "USDBRL=X",  # USD/BRL exchange rate
        "vix": "^VIX",         # CBOE Volatility Index
        "selic_proxy": "^IRX", # 13-week T-Bill rate as risk-free proxy
    }
    
    def __init__(self, cache_path: str = "data/macro.csv"):
        self.cache_path = cache_path
        self._cache: pd.DataFrame | None = None
    
    def fetch_all(
        self,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch all macro indicators and compute derived features.
        
        Args:
            start: Start date.
            end: End date.
            use_cache: Whether to use local cache.
            
        Returns:
            DataFrame with macro features, indexed by date.
        """
        from pathlib import Path
        
        cache_file = Path(self.cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        if use_cache and cache_file.exists():
            print(f"Loading macro data from cache: {self.cache_path}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            self._cache = df
            return df
        
        print("Fetching macro data from Yahoo Finance...")
        
        # Fetch raw data
        raw_data = {}
        for name, ticker in self.TICKERS.items():
            try:
                data = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    progress=False,
                    threads=False,
                )
                if not data.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        raw_data[name] = data["Close"][ticker]
                    else:
                        raw_data[name] = data["Close"]
                    print(f"  ✓ {name}: {len(raw_data[name])} rows")
                else:
                    print(f"  ✗ {name}: no data")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        # Combine into DataFrame
        df = pd.DataFrame(raw_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Forward fill missing values (holidays, weekends)
        df = df.ffill()
        
        # Compute derived features
        df = self._compute_features(df)
        
        # Cache to disk
        print(f"Saving macro data to cache: {self.cache_path}")
        df.to_csv(cache_file)
        self._cache = df
        
        return df
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived macro features.
        
        Features computed:
            - usdbrl: Raw USD/BRL exchange rate
            - usdbrl_return: Log return of USD/BRL
            - usdbrl_vol_20d: 20-day rolling volatility of USD/BRL
            - usdbrl_zscore: Z-score of USD/BRL vs 60-day mean
            - vix: Raw VIX value
            - vix_change: 1-day change in VIX
            - vix_zscore: Z-score of VIX vs 60-day mean
            - selic_proxy: Short-term rate proxy
        """
        result = pd.DataFrame(index=df.index)
        
        # USD/BRL features
        if "usdbrl" in df.columns:
            result["usdbrl"] = df["usdbrl"]
            result["usdbrl_return"] = np.log(df["usdbrl"] / df["usdbrl"].shift(1))
            result["usdbrl_vol_20d"] = result["usdbrl_return"].rolling(20).std()
            usdbrl_mean = df["usdbrl"].rolling(60).mean()
            usdbrl_std = df["usdbrl"].rolling(60).std()
            result["usdbrl_zscore"] = (df["usdbrl"] - usdbrl_mean) / usdbrl_std
        
        # VIX features
        if "vix" in df.columns:
            result["vix"] = df["vix"]
            result["vix_change"] = df["vix"].diff()
            result["vix_pct_change"] = df["vix"].pct_change()
            vix_mean = df["vix"].rolling(60).mean()
            vix_std = df["vix"].rolling(60).std()
            result["vix_zscore"] = (df["vix"] - vix_mean) / vix_std
            # VIX regime: high volatility when VIX > 20
            result["vix_high"] = (df["vix"] > 20).astype(float)
        
        # SELIC proxy (if available)
        if "selic_proxy" in df.columns:
            result["selic_proxy"] = df["selic_proxy"]
            result["selic_change"] = df["selic_proxy"].diff()
        
        # Drop rows with NaN from rolling calculations
        result = result.dropna()
        
        return result
    
    def get_aligned_features(
        self,
        dates: pd.DatetimeIndex,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get macro features aligned to specific dates.
        
        Args:
            dates: Target date index to align to.
            features: List of feature names to include. If None, include all.
            
        Returns:
            DataFrame with macro features aligned to input dates.
        """
        if self._cache is None:
            raise ValueError("No macro data loaded. Call fetch_all() first.")
        
        df = self._cache
        
        if features is not None:
            df = df[[f for f in features if f in df.columns]]
        
        # Reindex to target dates with forward fill
        aligned = df.reindex(dates, method="ffill")
        
        return aligned


def add_macro_to_node_features(
    node_features: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add macro features to existing node features.
    
    Each macro feature is broadcast to all assets (same value per date).
    
    Args:
        node_features: Existing node features DataFrame (MultiIndex columns).
        macro_df: Macro features DataFrame.
        
    Returns:
        Combined features DataFrame.
    """
    # Get common dates
    common_dates = node_features.index.intersection(macro_df.index)
    
    # Filter to common dates
    node_features = node_features.loc[common_dates]
    macro_df = macro_df.loc[common_dates]
    
    # Get list of tickers from node_features
    tickers = node_features.columns.get_level_values(0).unique()
    
    # Create macro features for each ticker (broadcast)
    macro_expanded = {}
    for ticker in tickers:
        for col in macro_df.columns:
            macro_expanded[(ticker, f"macro_{col}")] = macro_df[col]
    
    # Combine
    macro_features_df = pd.DataFrame(macro_expanded)
    macro_features_df.columns = pd.MultiIndex.from_tuples(
        macro_features_df.columns, names=["ticker", "feature"]
    )
    
    # Concatenate with existing features
    combined = pd.concat([node_features, macro_features_df], axis=1)
    
    return combined.dropna()


# Convenience function
def fetch_macro_data(start: date, end: date, cache_path: str = "data/macro.csv") -> pd.DataFrame:
    """Convenience function to fetch macro data.
    
    Example:
        >>> macro = fetch_macro_data(date(2015, 1, 1), date(2023, 12, 31))
    """
    fetcher = MacroDataFetcher(cache_path=cache_path)
    return fetcher.fetch_all(start, end)
