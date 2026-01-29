"""
Data Loader Module using OpenBB SDK
====================================

This module handles fetching and preprocessing of market data using OpenBB.
It implements a robust pipeline with stationarity checks and data validation.

Economic Logic:
    - We fetch adjusted close prices to account for splits/dividends
    - Log returns are used for stationarity in most analyses
    - Price levels are preserved for cointegration tests (require I(1) series)
    - Look-ahead bias is prevented by careful time indexing

References:
    - Hamilton, J.D. (1994) Time Series Analysis, Chapter 17 (Cointegration)
    - OpenBB Documentation: https://docs.openbb.co/
"""

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

# We'll use yfinance as fallback if OpenBB isn't available
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    import yfinance as yf


class StationarityResult(BaseModel):
    """Result of stationarity tests for a single series."""
    
    ticker: str
    adf_statistic: float
    adf_pvalue: float
    adf_critical_1pct: float
    adf_critical_5pct: float
    kpss_statistic: float
    kpss_pvalue: float
    is_stationary: bool = Field(
        description="True if ADF rejects AND KPSS fails to reject."
    )
    integration_order: Literal[0, 1, 2] = Field(
        description="Estimated integration order: I(0), I(1), or I(2)."
    )


class DataLoader:
    """Fetches and preprocesses market data for cointegration analysis.
    
    This class handles:
    1. Data fetching via OpenBB (or yfinance fallback)
    2. Data sanitization (missing values, outliers)
    3. Stationarity testing (ADF + KPSS)
    4. Feature engineering (returns, volatility, z-scores)
    
    Economic Rationale:
        Cointegration requires I(1) price series. We verify this using both
        ADF (null: unit root) and KPSS (null: stationary) tests. Agreement
        between both tests increases confidence in integration order.
    
    Example:
        >>> loader = DataLoader(tickers=['XLK', 'XLF', 'XLE'])
        >>> prices, returns = loader.fetch_data(
        ...     start=date(2015, 1, 1),
        ...     end=date(2023, 12, 31)
        ... )
        >>> stationarity = loader.test_stationarity(prices)
    """
    
    def __init__(
        self,
        tickers: list[str],
        adjust_for_splits: bool = True,
    ) -> None:
        """Initialize DataLoader.
        
        Args:
            tickers: List of ticker symbols to fetch.
            adjust_for_splits: Whether to use split-adjusted prices.
        """
        self.tickers = [t.upper() for t in tickers]
        self.adjust_for_splits = adjust_for_splits
        self._prices_cache: pd.DataFrame | None = None
        
    def fetch_data(
        self,
        start: date,
        end: date,
        use_cache: bool = True,
        cache_path: str = "data/prices.csv",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch OHLCV data and compute returns.
        
        Args:
            start: Start date for data fetch.
            end: End date for data fetch.
            use_cache: Whether to use cached data if available.
            cache_path: Path to local CSV cache file.
            
        Returns:
            Tuple of (prices, returns) DataFrames.
            - prices: Adjusted close prices, indexed by date
            - returns: Log returns, indexed by date
            
        Raises:
            ValueError: If insufficient data is fetched.
            
        Note:
            **Look-Ahead Bias Warning**: Returns are computed as:
            r_t = log(P_t / P_{t-1})
            This ensures r_t only uses information available at time t.
        """
        import os
        from pathlib import Path
        
        # Ensure data directory exists
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        if use_cache and Path(cache_path).exists():
            print(f"Loading data from local cache: {cache_path}")
            prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            self._prices_cache = prices
        elif use_cache and self._prices_cache is not None:
            prices = self._prices_cache
        else:
            print("Downloading data from source...")
            prices = self._fetch_prices(start, end)
            self._prices_cache = prices
            # Save to cache
            print(f"Saving data to cache: {cache_path}")
            prices.to_csv(cache_path)
            
        # Validate data quality
        self._validate_data(prices)
        
        # Compute log returns (shift ensures no look-ahead)
        # r_t = log(P_t) - log(P_{t-1})
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        
        return prices, returns
    
    def _fetch_prices(self, start: date, end: date) -> pd.DataFrame:
        """Fetch adjusted close prices from data source."""
        if OPENBB_AVAILABLE:
            return self._fetch_via_openbb(start, end)
        else:
            return self._fetch_via_yfinance(start, end)
    
    def _fetch_via_openbb(self, start: date, end: date) -> pd.DataFrame:
        """Fetch data using OpenBB SDK.
        
        OpenBB provides institutional-grade data with better coverage
        for corporate actions and splits.
        """
        prices_dict: dict[str, pd.Series] = {}
        
        for ticker in self.tickers:
            try:
                # OpenBB equity price historical data
                result = obb.equity.price.historical(
                    symbol=ticker,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    provider="yfinance",  # Use yfinance as provider
                )
                df = result.to_df()
                
                # Use adjusted close for split/dividend adjustment
                col = "adj_close" if "adj_close" in df.columns else "close"
                prices_dict[ticker] = df[col]
                
            except Exception as e:
                print(f"Warning: Failed to fetch {ticker} via OpenBB: {e}")
                # Fallback to yfinance for this ticker
                prices_dict[ticker] = self._fetch_single_yfinance(
                    ticker, start, end
                )
        
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        
        return prices
    
    def _fetch_via_yfinance(self, start: date, end: date) -> pd.DataFrame:
        """Fallback: Fetch data using yfinance with retry logic."""
        import yfinance as yf
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=self.tickers,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=self.adjust_for_splits,
                    progress=True,
                    threads=False,  # Avoid database lock issues
                )
                
                # Handle single vs multiple tickers
                if len(self.tickers) == 1:
                    prices = data[["Close"]].rename(columns={"Close": self.tickers[0]})
                else:
                    # yfinance returns MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
                    else:
                        prices = data[["Close"]] if "Close" in data.columns else data[["Adj Close"]]
                
                return prices
                
            except Exception as e:
                if "database is locked" in str(e) or "OperationalError" in str(e):
                    print(f"Database lock error (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        # Final fallback: fetch one by one
        print("Falling back to individual ticker fetch...")
        prices_dict = {}
        for ticker in self.tickers:
            try:
                prices_dict[ticker] = self._fetch_single_yfinance(ticker, start, end)
                time.sleep(0.5)  # Rate limit
            except Exception as e:
                print(f"Warning: Could not fetch {ticker}: {e}")
        
        return pd.DataFrame(prices_dict)
    
    def _fetch_single_yfinance(
        self, ticker: str, start: date, end: date
    ) -> pd.Series:
        """Fetch single ticker via yfinance (helper for fallback)."""
        import yfinance as yf
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=self.adjust_for_splits,
                    progress=False,
                    threads=False,
                )
                
                if data.empty:
                    raise ValueError(f"No data for {ticker}")
                
                # Handle MultiIndex for single ticker (yfinance quirk)
                if isinstance(data.columns, pd.MultiIndex):
                    return data["Close"][ticker]
                return data["Close"]
                
            except Exception as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise ValueError(f"Failed to fetch {ticker} after {max_retries} attempts")
    
    def _validate_data(self, prices: pd.DataFrame) -> None:
        """Validate data quality and handle issues.
        
        Checks:
        1. No more than 5% missing values per column
        2. No duplicate indices
        3. Monotonic datetime index
        
        Raises:
            ValueError: If data quality issues are detected.
        """
        # Check for missing data
        missing_pct = prices.isna().mean()
        high_missing = missing_pct[missing_pct > 0.05]
        if len(high_missing) > 0:
            raise ValueError(
                f"Tickers with >5% missing data: {high_missing.to_dict()}"
            )
        
        # Forward-fill small gaps (weekends, holidays)
        prices.ffill(inplace=True)
        
        # Check for duplicate indices
        if prices.index.duplicated().any():
            raise ValueError("Duplicate dates detected in price data")
            
        # Ensure datetime index is monotonic
        if not prices.index.is_monotonic_increasing:
            raise ValueError("Price index is not monotonically increasing")
    
    def test_stationarity(
        self,
        data: pd.DataFrame,
        significance: float = 0.05,
    ) -> dict[str, StationarityResult]:
        """Test stationarity of each series using ADF and KPSS tests.
        
        Economic Rationale:
            For cointegration analysis, we need I(1) series (unit root in levels,
            stationary in first differences). We use two complementary tests:
            
            - ADF Test: H0 = unit root exists
              (We want to FAIL to reject at 5% for levels to confirm I(1))
              
            - KPSS Test: H0 = series is stationary  
              (We want to REJECT at 5% for levels to confirm I(1))
              
            Agreement between both tests provides stronger evidence.
        
        Args:
            data: DataFrame of price or return series.
            significance: Significance level for hypothesis tests.
            
        Returns:
            Dictionary mapping ticker to StationarityResult.
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        results: dict[str, StationarityResult] = {}
        
        for col in data.columns:
            series = data[col].dropna()
            
            # ADF Test (null: unit root)
            adf_result = adfuller(series, autolag="AIC")
            adf_stat, adf_pval = adf_result[0], adf_result[1]
            adf_crit = adf_result[4]
            
            # KPSS Test (null: stationary)
            # regression='c' tests level stationarity
            kpss_result = kpss(series, regression="c", nlags="auto")
            kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]
            
            # Determine stationarity
            # Stationary if: ADF rejects H0 (p < 0.05) AND KPSS fails to reject (p > 0.05)
            adf_rejects = adf_pval < significance
            kpss_fails_to_reject = kpss_pval > significance
            is_stationary = adf_rejects and kpss_fails_to_reject
            
            # Determine integration order
            if is_stationary:
                integration_order = 0
            else:
                # Test first difference
                diff_series = series.diff().dropna()
                adf_diff = adfuller(diff_series, autolag="AIC")
                if adf_diff[1] < significance:
                    integration_order = 1
                else:
                    integration_order = 2
            
            results[col] = StationarityResult(
                ticker=col,
                adf_statistic=adf_stat,
                adf_pvalue=adf_pval,
                adf_critical_1pct=adf_crit["1%"],
                adf_critical_5pct=adf_crit["5%"],
                kpss_statistic=kpss_stat,
                kpss_pvalue=kpss_pval,
                is_stationary=is_stationary,
                integration_order=integration_order,
            )
        
        return results
    
    def compute_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """Compute node features for GNN.
        
        Features (all computed with shift(1) to avoid look-ahead):
        1. Return (1-day)
        2. Return (5-day momentum)
        3. Volatility (rolling 20-day)
        4. Z-score of price vs 20-day MA
        5. RSI (14-day)
        6. Volume ratio (vs 20-day avg) - if available
        7. Spread from sector mean
        8. Lagged return (t-1)
        
        Args:
            prices: DataFrame of adjusted close prices.
            returns: DataFrame of log returns.
            lookback: Lookback period for rolling calculations.
            
        Returns:
            DataFrame of shape (n_dates, n_tickers, n_features).
            Indexed by date, columns are MultiIndex (ticker, feature).
        """
        features: dict[str, pd.DataFrame] = {}
        
        for ticker in prices.columns:
            price = prices[ticker]
            ret = returns[ticker]
            
            # All features use shift to prevent look-ahead
            ticker_features = pd.DataFrame({
                "return_1d": ret.shift(1),
                "return_5d": ret.rolling(5).sum().shift(1),
                "volatility_20d": ret.rolling(lookback).std().shift(1),
                "zscore_20d": (
                    (price - price.rolling(lookback).mean()) / 
                    price.rolling(lookback).std()
                ).shift(1),
                "rsi_14d": self._compute_rsi(price, 14).shift(1),
                "spread_from_mean": (
                    price / prices.mean(axis=1) - 1
                ).shift(1),
                "return_lag1": ret.shift(1),
                "return_lag2": ret.shift(2),
            })
            
            features[ticker] = ticker_features
        
        # Combine into MultiIndex DataFrame
        result = pd.concat(features, axis=1)
        result.columns = pd.MultiIndex.from_tuples(
            [(ticker, feat) for ticker, feat in result.columns],
            names=["ticker", "feature"]
        )
        
        return result.dropna()
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = avg_gain / avg_loss over `period` days.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for initial periods
    
    def normalize_features(
        self,
        features: pd.DataFrame,
        train_end_date: pd.Timestamp | str | None = None,
        scaler: StandardScaler | None = None,
    ) -> tuple[pd.DataFrame, StandardScaler]:
        """Normalize features using Z-Score (StandardScaler).
        
        CRITICAL: To avoid look-ahead bias, the scaler is fit ONLY on
        training data and then applied to the entire dataset.
        
        Args:
            features: DataFrame with MultiIndex columns (ticker, feature).
            train_end_date: End date of training period. If None, fits on all data (⚠️).
            scaler: Pre-fitted scaler (for inference). If None, create new one.
            
        Returns:
            Tuple of (normalized_features, fitted_scaler).
            Save the scaler for inference on new data.
        """
        # Flatten MultiIndex for sklearn (it expects 2D array)
        # Original shape: (n_dates, n_tickers * n_features)
        features_flat = features.copy()
        
        if scaler is None:
            scaler = StandardScaler()
            
            if train_end_date is not None:
                # FIT ONLY ON TRAINING DATA
                train_end_ts = pd.Timestamp(train_end_date)
                train_mask = features_flat.index <= train_end_ts
                train_data = features_flat.loc[train_mask]
                
                if len(train_data) == 0:
                    raise ValueError(f"No training data before {train_end_date}")
                
                scaler.fit(train_data)
                print(f"✅ Features normalizadas (Scaler fit em {len(train_data)} dias até {train_end_date})")
            else:
                # FALLBACK: Fit on all data (⚠️ Look-ahead risk)
                scaler.fit(features_flat)
                print("⚠️ AVISO: Normalizando com dataset inteiro (Look-ahead potencial)")
        
        # Transform entire dataset
        features_scaled = pd.DataFrame(
            scaler.transform(features_flat),
            index=features_flat.index,
            columns=features_flat.columns,
        )
        
        # Log scale info for debugging
        n_features = len(features.columns.get_level_values('feature').unique())
        print(f"   📊 Features por ticker: {n_features}")
        print(f"   📊 Média após normalização: {features_scaled.values.mean():.4f}")
        print(f"   📊 Std após normalização: {features_scaled.values.std():.4f}")
        
        return features_scaled, scaler
    
    def rank_transform_features(
        self,
        features: pd.DataFrame,
        output_range: tuple[float, float] = (-0.5, 0.5),
    ) -> pd.DataFrame:
        """Transform features to cross-sectional percentile ranks.
        
        For each date t and each feature f, this method:
        1. Ranks the N assets from 1 to N
        2. Normalizes to output_range (default [-0.5, 0.5])
        
        This is the KEY transformation for Learning to Rank:
        - Robust to outliers: +50% return has same rank as +5% if both are the best
        - Stabilizes neural network gradients during training
        - Aligns input representation with ranking objective
        - Non-parametric: no Gaussian assumption
        
        Args:
            features: DataFrame with MultiIndex columns (ticker, feature).
            output_range: Tuple (min, max) for normalized ranks.
            
        Returns:
            DataFrame with rank-transformed features.
        """
        ranked_features = features.copy()
        
        # Get unique feature names
        feature_names = features.columns.get_level_values('feature').unique()
        tickers = features.columns.get_level_values('ticker').unique()
        n_tickers = len(tickers)
        
        for feat in feature_names:
            # Extract this feature for all tickers (N columns)
            feat_cols = [(t, feat) for t in tickers]
            feat_data = features[feat_cols]
            
            # Rank across tickers for each date (axis=1)
            # rank() returns 1 to N, we normalize to output_range
            ranked = feat_data.rank(axis=1, method='average', na_option='keep')
            
            # Normalize to output_range: (rank - 1) / (N - 1) * (max - min) + min
            min_val, max_val = output_range
            if n_tickers > 1:
                normalized = (ranked - 1) / (n_tickers - 1) * (max_val - min_val) + min_val
            else:
                normalized = ranked * 0  # Single ticker: just zero
            
            # Put back into result
            ranked_features[feat_cols] = normalized
        
        print(f"✅ Rank Transform aplicado:")
        print(f"   📊 Features: {len(feature_names)}")
        print(f"   📊 Range: {output_range}")
        print(f"   📊 Média: {ranked_features.values[~np.isnan(ranked_features.values)].mean():.4f}")
        print(f"   📊 Std: {ranked_features.values[~np.isnan(ranked_features.values)].std():.4f}")
        
        return ranked_features


def create_data_splits(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    features: pd.DataFrame,
    train_end: date,
    val_end: date,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Split data into train/validation/test sets.
    
    This implements a regime-aware split:
    - Train: 2015-2019 (Low volatility bull market)
    - Validation: 2020 (COVID crisis - stress test)
    - Test: 2021+ (Inflation/rate hike regime)
    
    Args:
        prices: Full price DataFrame.
        returns: Full returns DataFrame.  
        features: Full features DataFrame.
        train_end: End date for training period.
        val_end: End date for validation period.
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        'prices', 'returns', 'features' DataFrames.
    """
    train_mask = prices.index <= pd.Timestamp(train_end)
    val_mask = (prices.index > pd.Timestamp(train_end)) & (
        prices.index <= pd.Timestamp(val_end)
    )
    test_mask = prices.index > pd.Timestamp(val_end)
    
    return {
        "train": {
            "prices": prices[train_mask],
            "returns": returns[train_mask],
            "features": features[features.index.isin(prices[train_mask].index)],
        },
        "val": {
            "prices": prices[val_mask],
            "returns": returns[val_mask],
            "features": features[features.index.isin(prices[val_mask].index)],
        },
        "test": {
            "prices": prices[test_mask],
            "returns": returns[test_mask],
            "features": features[features.index.isin(prices[test_mask].index)],
        },
    }
