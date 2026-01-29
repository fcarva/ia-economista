"""
Cointegration Graph Module
==========================

This module constructs a weighted adjacency matrix from rolling Johansen
cointegration tests. The graph represents the dynamic cointegration structure
of the asset universe.

Economic Logic:
    Traditional pairs trading uses static Engle-Granger cointegration, which:
    1. Only handles pairwise relationships
    2. Ignores regime changes
    3. Assumes a fixed cointegrating vector
    
    Our approach uses the Johansen test in a rolling window to:
    1. Handle multivariate cointegration (find all cointegrating vectors)
    2. Capture time-varying relationships (regime detection)
    3. Construct a graph where edge weights = cointegration strength

Theory:
    The Johansen test estimates the rank r of the cointegration matrix Π in:
    
        ΔX_t = Π X_{t-1} + Γ_1 ΔX_{t-1} + ... + ε_t
        
    where Π = αβ', α = adjustment speeds, β = cointegrating vectors.
    
    The trace statistic tests H0: rank(Π) ≤ r vs H1: rank(Π) > r:
    
        λ_trace(r) = -T Σ_{i=r+1}^{n} log(1 - λ̂_i)

References:
    - Johansen, S. (1991). "Estimation and Hypothesis Testing of 
      Cointegration Vectors in Gaussian Vector Autoregressive Models"
    - Hamilton (1994), Chapter 19
"""

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class JohansenResult:
    """Container for Johansen test results between two assets."""
    
    def __init__(
        self,
        asset_i: str,
        asset_j: str,
        trace_stat: float,
        critical_value_95: float,
        eigenvalue: float,
        cointegrating_vector: np.ndarray,
        is_cointegrated: bool,
    ) -> None:
        self.asset_i = asset_i
        self.asset_j = asset_j
        self.trace_stat = trace_stat
        self.critical_value_95 = critical_value_95
        self.eigenvalue = eigenvalue
        self.cointegrating_vector = cointegrating_vector
        self.is_cointegrated = is_cointegrated
        
        # Normalized strength: ratio of trace stat to critical value
        # Values > 1 indicate cointegration at 95% confidence
        self.strength = trace_stat / critical_value_95 if critical_value_95 > 0 else 0.0


class CointegrationGraph:
    """Constructs and maintains a rolling cointegration graph.
    
    The graph G = (V, E, W) where:
    - V = set of assets (tickers)
    - E = pairs with trace_stat/critical_value > threshold
    - W = edge weights = normalized cointegration strength
    
    Economic Interpretation:
        High edge weight between XLF and XLK suggests these sectors move
        together in the long run, implying shared macroeconomic drivers
        (e.g., interest rates affect both financials and tech valuations).
    
    Example:
        >>> graph = CointegrationGraph(
        ...     tickers=['XLK', 'XLF', 'XLE'],
        ...     rolling_window=252,
        ... )
        >>> adjacency = graph.compute_adjacency(prices, as_of_date=date(2023, 6, 30))
        >>> print(adjacency)
        # Returns 3x3 weighted adjacency matrix
    """
    
    def __init__(
        self,
        tickers: list[str],
        rolling_window: int = 252,
        det_order: Literal[-1, 0, 1] = 0,
        min_edge_weight: float = 0.3,
    ) -> None:
        """Initialize CointegrationGraph.
        
        Args:
            tickers: List of asset tickers.
            rolling_window: Number of trading days for rolling window.
            det_order: Deterministic term in Johansen test:
                -1 = no deterministic term
                 0 = constant in cointegrating relation only
                 1 = constant and trend
            min_edge_weight: Minimum normalized trace stat for edge inclusion.
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rolling_window = rolling_window
        self.det_order = det_order
        self.min_edge_weight = min_edge_weight
        
        # Cache for rolling adjacencies
        self._adjacency_cache: dict[pd.Timestamp, np.ndarray] = {}
        self._results_cache: dict[pd.Timestamp, dict[tuple[str, str], JohansenResult]] = {}
    
    def compute_pairwise_johansen(
        self,
        data: pd.DataFrame,
        asset_i: str,
        asset_j: str,
    ) -> JohansenResult:
        """Compute Johansen test for a pair of assets.
        
        Args:
            data: DataFrame with columns for at least asset_i and asset_j.
            asset_i: First asset ticker.
            asset_j: Second asset ticker.
            
        Returns:
            JohansenResult containing trace statistic and cointegrating vector.
            
        Note:
            We test for rank r=0 (no cointegration) vs r≥1 (cointegration).
            If trace_stat > critical_value at 95%, we reject H0 and conclude
            the pair is cointegrated.
        """
        # Extract the two series
        pair_data = data[[asset_i, asset_j]].dropna()
        
        if len(pair_data) < 30:
            # Not enough data for meaningful test
            return JohansenResult(
                asset_i=asset_i,
                asset_j=asset_j,
                trace_stat=0.0,
                critical_value_95=1.0,
                eigenvalue=0.0,
                cointegrating_vector=np.array([1.0, 0.0]),
                is_cointegrated=False,
            )
        
        try:
            # Run Johansen test
            # k_ar_diff=1 uses 1 lag in the VECM
            result = coint_johansen(
                pair_data.values,
                det_order=self.det_order,
                k_ar_diff=1,
            )
            
            # Trace statistic for r=0 (testing if at least 1 cointegrating vector)
            trace_stat = result.lr1[0]  # lr1 = trace statistic
            
            # Critical value at 95% (index 1 in cvt)
            critical_value = result.cvt[0, 1]
            
            # First eigenvalue
            eigenvalue = result.eig[0]
            
            # First cointegrating vector (normalized)
            coint_vector = result.evec[:, 0]
            coint_vector = coint_vector / np.linalg.norm(coint_vector)
            
            is_cointegrated = trace_stat > critical_value
            
            return JohansenResult(
                asset_i=asset_i,
                asset_j=asset_j,
                trace_stat=trace_stat,
                critical_value_95=critical_value,
                eigenvalue=eigenvalue,
                cointegrating_vector=coint_vector,
                is_cointegrated=is_cointegrated,
            )
            
        except Exception as e:
            # Numerical issues in Johansen can occur
            print(f"Warning: Johansen test failed for ({asset_i}, {asset_j}): {e}")
            return JohansenResult(
                asset_i=asset_i,
                asset_j=asset_j,
                trace_stat=0.0,
                critical_value_95=1.0,
                eigenvalue=0.0,
                cointegrating_vector=np.array([1.0, 0.0]),
                is_cointegrated=False,
            )
    
    def compute_adjacency(
        self,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Compute weighted adjacency matrix as of a specific date.
        
        **CRITICAL: Look-ahead bias prevention**
        This function only uses data STRICTLY BEFORE as_of_date.
        The rolling window is [as_of_date - window, as_of_date - 1].
        
        Args:
            prices: Full price DataFrame (all available data).
            as_of_date: Date to compute adjacency for (exclusive upper bound).
            use_cache: Whether to use cached results.
            
        Returns:
            Weighted adjacency matrix of shape (n_assets, n_assets).
            A[i,j] = cointegration strength between asset i and j.
            Diagonal is 0 (no self-loops).
        """
        as_of_ts = pd.Timestamp(as_of_date)
        
        if use_cache and as_of_ts in self._adjacency_cache:
            return self._adjacency_cache[as_of_ts]
        
        # **LOOK-AHEAD BIAS PREVENTION**
        # Only use data STRICTLY BEFORE as_of_date
        window_end = as_of_ts - pd.Timedelta(days=1)
        window_start = window_end - pd.Timedelta(days=self.rolling_window * 2)
        
        window_data = prices[
            (prices.index >= window_start) & (prices.index <= window_end)
        ].tail(self.rolling_window)
        
        if len(window_data) < self.rolling_window // 2:
            # Not enough data, return zero adjacency
            return np.zeros((self.n_assets, self.n_assets))
        
        # Compute pairwise Johansen tests
        adjacency = np.zeros((self.n_assets, self.n_assets))
        results: dict[tuple[str, str], JohansenResult] = {}
        
        for i, asset_i in enumerate(self.tickers):
            for j, asset_j in enumerate(self.tickers):
                if i >= j:
                    continue  # Skip diagonal and lower triangle
                    
                result = self.compute_pairwise_johansen(
                    window_data, asset_i, asset_j
                )
                results[(asset_i, asset_j)] = result
                
                # Symmetric weighted adjacency
                weight = max(result.strength, 0.0)
                if weight >= self.min_edge_weight:
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
        
        # Cache results
        self._adjacency_cache[as_of_ts] = adjacency
        self._results_cache[as_of_ts] = results
        
        return adjacency
    
    def compute_rolling_adjacencies(
        self,
        prices: pd.DataFrame,
        recompute_freq: int = 5,
    ) -> dict[pd.Timestamp, np.ndarray]:
        """Compute adjacency matrices for all dates in the data.
        
        Args:
            prices: Price DataFrame.
            recompute_freq: Recompute every N trading days (for efficiency).
            
        Returns:
            Dictionary mapping dates to adjacency matrices.
            
        Economic Rationale:
            We don't need to recompute daily - cointegration relationships
            are slow-moving. Recomputing weekly (5 days) captures regime
            changes while saving computation.
        """
        # Start from after the initial window
        valid_dates = prices.index[self.rolling_window:]
        
        adjacencies: dict[pd.Timestamp, np.ndarray] = {}
        last_adjacency: np.ndarray | None = None
        
        for i, date in enumerate(valid_dates):
            if i % recompute_freq == 0:
                last_adjacency = self.compute_adjacency(prices, date)
            
            if last_adjacency is not None:
                adjacencies[date] = last_adjacency
        
        return adjacencies
    
    def get_spread(
        self,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
        asset_i: str,
        asset_j: str,
    ) -> pd.Series:
        """Compute the cointegrating spread for a pair.
        
        The spread z_t = Y_t - β X_t where β is from the cointegrating vector.
        By the Granger Representation Theorem, z_t is stationary and
        mean-reverting.
        
        Args:
            prices: Price DataFrame.
            as_of_date: Date to get cointegrating vector from.
            asset_i: First asset (Y in regression).
            asset_j: Second asset (X in regression).
            
        Returns:
            Spread time series.
        """
        as_of_ts = pd.Timestamp(as_of_date)
        
        # Get cached result or compute
        if as_of_ts in self._results_cache:
            key = (asset_i, asset_j) if (asset_i, asset_j) in self._results_cache[as_of_ts] else (asset_j, asset_i)
            result = self._results_cache[as_of_ts].get(key)
        else:
            self.compute_adjacency(prices, as_of_date)
            key = (asset_i, asset_j) if (asset_i, asset_j) in self._results_cache[as_of_ts] else (asset_j, asset_i)
            result = self._results_cache[as_of_ts].get(key)
        
        if result is None:
            # No cointegration result, use simple log spread
            return np.log(prices[asset_i]) - np.log(prices[asset_j])
        
        # Extract hedge ratio from cointegrating vector
        # If vector is [β1, β2], spread = β1*log(Y) + β2*log(X)
        coint_vec = result.cointegrating_vector
        
        spread = (
            coint_vec[0] * np.log(prices[asset_i]) +
            coint_vec[1] * np.log(prices[asset_j])
        )
        
        return spread
    
    def get_edge_list(
        self,
        adjacency: np.ndarray,
        include_weights: bool = True,
    ) -> list[tuple[int, int]] | list[tuple[int, int, float]]:
        """Convert adjacency matrix to edge list for PyTorch Geometric.
        
        Args:
            adjacency: Weighted adjacency matrix.
            include_weights: Whether to include edge weights.
            
        Returns:
            List of (source, target) or (source, target, weight) tuples.
        """
        edges = []
        
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if adjacency[i, j] > 0:
                    if include_weights:
                        edges.append((i, j, adjacency[i, j]))
                    else:
                        edges.append((i, j))
        
        return edges
    
    def to_pyg_data(
        self,
        adjacency: np.ndarray,
        node_features: np.ndarray,
    ):
        """Convert to PyTorch Geometric Data object.
        
        Args:
            adjacency: Weighted adjacency matrix (n_assets, n_assets).
            node_features: Node feature matrix (n_assets, n_features).
            
        Returns:
            PyTorch Geometric Data object.
        """
        import torch
        from torch_geometric.data import Data
        
        # Get edge indices and weights
        edge_index_list = []
        edge_weights = []
        
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if adjacency[i, j] > 0:
                    edge_index_list.append([i, j])
                    edge_weights.append(adjacency[i, j])
        
        if len(edge_index_list) == 0:
            # No edges - create self-loops to avoid empty graph
            edge_index = torch.tensor([[i, i] for i in range(self.n_assets)]).T
            edge_weights = torch.ones(self.n_assets)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(-1),
        )
