"""
Analyst Agent Module
====================

This agent is responsible for generating trading signals using the GNN
and updating the cointegration graph.

Economic Role:
    In a trading desk, the Analyst is the "idea generator". They:
    1. Monitor market data and statistical relationships
    2. Identify potential trading opportunities
    3. Generate initial trade recommendations
    
    The Analyst does NOT consider position limits or risk - that's the
    Risk Manager's job. This separation enforces the "generate, then filter"
    pattern that prevents premature optimization.

LangGraph Integration:
    This agent is a node in the LangGraph workflow:
    
        [Market Data] → [Analyst] → [Risk Manager] → [Executor]
                           ↑
                     (GNN + Coint Graph)
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from pydantic import BaseModel, Field

from ..features.cointegration_graph import CointegrationGraph
from ..features.causal_discovery import NOTEARS, combine_cointegration_and_causal_graphs
from ..models.gnn_model import CointegrationGNN


class SignalOutput(BaseModel):
    """Output from Analyst agent."""
    
    timestamp: str = Field(description="Signal generation timestamp")
    signals: dict[str, float] = Field(description="Ticker to signal mapping [-1, 1]")
    confidence: dict[str, float] = Field(description="Confidence per signal [0, 1]")
    regime: str = Field(description="Detected market regime")
    graph_density: float = Field(description="Cointegration graph edge density")
    causal_edges: list[tuple[str, str]] = Field(description="Detected causal relationships")


@dataclass
class AnalystState:
    """Internal state maintained by Analyst agent."""
    
    last_graph_update: pd.Timestamp | None = None
    current_regime: str = "normal"
    graph_update_frequency: int = 5  # Days
    model: CointegrationGNN | None = None
    coint_graph: CointegrationGraph | None = None


class AnalystAgent:
    """Analyst agent for signal generation.
    
    This agent:
    1. Updates the cointegration graph (rolling Johansen)
    2. Runs causal discovery (NOTEARS)
    3. Generates signals via GNN inference
    4. Detects regime changes via graph density
    
    Example:
        >>> analyst = AnalystAgent(
        ...     tickers=['XLK', 'XLF', 'XLE'],
        ...     model=trained_gnn,
        ... )
        >>> signals = analyst.generate_signals(
        ...     prices=price_data,
        ...     features=feature_data,
        ...     current_date=pd.Timestamp('2023-06-30'),
        ... )
    """
    
    def __init__(
        self,
        tickers: list[str],
        model: CointegrationGNN,
        rolling_window: int = 252,
        causal_lambda: float = 0.01,
        graph_update_freq: int = 5,
    ) -> None:
        """Initialize Analyst agent.
        
        Args:
            tickers: Asset universe.
            model: Trained GNN model.
            rolling_window: Window for cointegration calculation.
            causal_lambda: L1 penalty for NOTEARS.
            graph_update_freq: Days between graph updates.
        """
        self.tickers = tickers
        self.model = model
        self.model.eval()
        
        self.coint_graph = CointegrationGraph(
            tickers=tickers,
            rolling_window=rolling_window,
        )
        
        self.notears = NOTEARS(lambda_l1=causal_lambda)
        
        self.state = AnalystState(graph_update_frequency=graph_update_freq)
        self.state.model = model
        self.state.coint_graph = self.coint_graph
        
        # Cache for graph
        self._current_adjacency: Any = None
        self._current_causal: Any = None
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> SignalOutput:
        """Generate trading signals.
        
        **LOOK-AHEAD BIAS PREVENTION**
        This function only uses data STRICTLY BEFORE current_date.
        Prices and features are filtered to exclude current_date.
        
        Args:
            prices: Price DataFrame (must extend to before current_date).
            features: Feature DataFrame from data_loader.
            current_date: Date to generate signals for.
            
        Returns:
            SignalOutput with signals, confidence, and metadata.
        """
        # 1. Check if we need to update the graph
        should_update_graph = (
            self.state.last_graph_update is None or
            (current_date - self.state.last_graph_update).days >= self.state.graph_update_frequency
        )
        
        if should_update_graph:
            self._update_graphs(prices, current_date)
            self.state.last_graph_update = current_date
        
        # 2. Prepare node features (up to yesterday)
        # **CRITICAL: Exclude current_date from features**
        feature_date = current_date - pd.Timedelta(days=1)
        
        # Find the closest available date in features
        available_dates = features.index[features.index < current_date]
        if len(available_dates) == 0:
            return self._empty_signal(current_date)
        
        feature_date = available_dates[-1]
        
        # Extract features for each ticker
        node_features = []
        for ticker in self.tickers:
            if (ticker, self.tickers[0]) in features.columns.names or ticker in features.columns.get_level_values(0):
                ticker_features = features.loc[feature_date, ticker].values
            else:
                ticker_features = [0.0] * 8  # Default features
            node_features.append(ticker_features)
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        
        # 3. Construct PyG data object
        if self._current_adjacency is None:
            combined_adjacency = None
        else:
            combined_adjacency = combine_cointegration_and_causal_graphs(
                self._current_adjacency,
                self._current_causal,
                alpha=0.5,
            )
        
        if combined_adjacency is None:
            combined_adjacency = self._current_adjacency
        
        data = self.coint_graph.to_pyg_data(
            adjacency=combined_adjacency,
            node_features=node_features_tensor.numpy(),
        )
        data.tickers = self.tickers
        
        # 4. Run GNN inference
        with torch.no_grad():
            positions = self.model(data).detach().cpu()
        
        signals = {
            ticker: float(max(-1.0, min(1.0, pos)))
            for ticker, pos in zip(self.tickers, positions.numpy())
        }
        
        # 5. Compute confidence (based on graph connectivity)
        confidences = self._compute_confidence(combined_adjacency)
        
        # 6. Detect regime
        regime = self._detect_regime(combined_adjacency)
        
        # 7. Extract causal edges
        causal_edges = self._extract_causal_edges()
        
        return SignalOutput(
            timestamp=current_date.isoformat(),
            signals=signals,
            confidence=confidences,
            regime=regime,
            graph_density=self._compute_graph_density(combined_adjacency),
            causal_edges=causal_edges,
        )
    
    def _update_graphs(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> None:
        """Update cointegration and causal graphs.
        
        Uses data STRICTLY BEFORE current_date.
        """
        import numpy as np
        
        # Update cointegration graph
        self._current_adjacency = self.coint_graph.compute_adjacency(
            prices, current_date
        )
        
        # Update causal graph using returns
        # Get returns up to yesterday
        returns_data = prices.pct_change().dropna()
        returns_data = returns_data[returns_data.index < current_date]
        
        if len(returns_data) >= 30:
            # Use last 252 days for causal discovery
            recent_returns = returns_data.tail(252).values
            causal_result = self.notears.fit(recent_returns)
            self._current_causal = causal_result.thresholded_adjacency
        else:
            self._current_causal = np.zeros_like(self._current_adjacency)
    
    def _compute_confidence(
        self,
        adjacency,
    ) -> dict[str, float]:
        """Compute confidence per signal based on graph connectivity.
        
        Economic Logic:
            Higher connectivity = more confirmation from related assets
            = higher confidence in the signal.
        """
        import numpy as np
        
        # Degree centrality as confidence proxy
        degrees = adjacency.sum(axis=1)
        max_degree = degrees.max() if degrees.max() > 0 else 1
        normalized = degrees / max_degree
        
        return {
            ticker: float(conf)
            for ticker, conf in zip(self.tickers, normalized)
        }
    
    def _detect_regime(
        self,
        adjacency,
    ) -> str:
        """Detect market regime from graph structure.
        
        Economic Logic:
            - High density = high correlation regime (crisis or momentum)
            - Low density = idiosyncratic regime (stock-picking)
            - Sudden changes = regime transitions
        """
        density = self._compute_graph_density(adjacency)
        
        if density > 0.7:
            return "high_correlation"
        elif density > 0.4:
            return "normal"
        else:
            return "dispersion"
    
    def _compute_graph_density(
        self,
        adjacency,
    ) -> float:
        """Compute graph density (edges / possible edges)."""
        import numpy as np
        
        n = adjacency.shape[0]
        possible_edges = n * (n - 1) / 2
        actual_edges = (adjacency > 0).sum() / 2  # Symmetric, divide by 2
        
        return float(actual_edges / possible_edges) if possible_edges > 0 else 0.0
    
    def _extract_causal_edges(self) -> list[tuple[str, str]]:
        """Extract directed causal edges from causal graph."""
        import numpy as np
        
        if self._current_causal is None:
            return []
        
        edges = []
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                if i != j and self._current_causal[i, j] > 0:
                    edges.append((self.tickers[i], self.tickers[j]))
        
        return edges
    
    def _empty_signal(self, current_date: pd.Timestamp) -> SignalOutput:
        """Return empty signals when no data available."""
        return SignalOutput(
            timestamp=current_date.isoformat(),
            signals={t: 0.0 for t in self.tickers},
            confidence={t: 0.0 for t in self.tickers},
            regime="unknown",
            graph_density=0.0,
            causal_edges=[],
        )
