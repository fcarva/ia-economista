"""
Causal Discovery Module
=======================

This module implements causal discovery algorithms for learning directed
graphs representing information flow between assets.

Algorithms:
    1. NOTEARS - DAG learning via continuous optimization (deprecated for daily data)
    2. GrangerLassoDiscovery - Lagged Granger causality with Lasso sparsity (RECOMMENDED)

Economic Logic:
    In market microstructure, information flows between assets with latency.
    For example:
    - Interest rate expectations affect Financials (XLF) first, then Tech (XLK)
    - Oil prices affect Energy (XLE) which affects Transportation costs (XLI)
    
    A directed graph captures this causal structure:
    
        XLF → XLK  (financials lead tech)
        XLE → XLI  (energy costs affect industrials)
    
    This information is valuable for:
    1. Lead-lag trading (trade the lagging asset when leading asset moves)
    2. Risk management (understand contagion paths)
    3. Regime detection (causal structure changes indicate regime shifts)

Why Granger-Lasso over NOTEARS for Daily Data:
    NOTEARS assumes contemporaneous causation (X_t causes Y_t). In efficient
    markets, news is absorbed quasi-instantaneously, so daily data shows
    synchronous correlation rather than causal structure.
    
    Granger causality looks at WHETHER X_{t-1} predicts Y_t, which captures
    the empirically observable lead-lag patterns at end-of-day frequency.
    
    Lasso regularization induces sparsity, selecting only the most predictive
    lead-lag relationships.

References:
    - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric
      Models and Cross-spectral Methods"
    - Billio et al. (2012). "Econometric measures of connectedness and systemic
      risk in the finance and insurance sectors"
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class GrangerLassoResult:
    """Container for Granger-Lasso causal discovery results.
    
    Attributes:
        adjacency: Raw coefficient matrix (n_assets, n_assets).
                   adjacency[i, j] = coefficient of asset_i in predicting asset_j.
                   Non-zero means i Granger-causes j.
        thresholded_adjacency: Binary adjacency after threshold.
        edge_weights: Absolute values of significant coefficients.
        n_edges: Number of detected causal edges.
        r2_scores: R² for each target regression.
        selected_alphas: Lasso alpha selected for each target.
        tickers: Asset names for labeling.
    """
    adjacency: NDArray[np.float64]
    thresholded_adjacency: NDArray[np.float64]
    edge_weights: NDArray[np.float64]
    n_edges: int
    r2_scores: NDArray[np.float64]
    selected_alphas: NDArray[np.float64]
    tickers: list[str] | None = None


class GrangerLassoDiscovery:
    """Granger Causality Discovery using Lasso-VAR(1).
    
    This class implements sparse Granger causality testing using Lasso
    regression. For each target asset, we regress its return at time t
    on ALL assets' returns at time t-1. Non-zero coefficients indicate
    Granger-causal relationships.
    
    Model for each target asset j:
        r_{j,t} = Σ_i β_{i,j} * r_{i,t-1} + ε_{j,t}
    
    If β_{i,j} ≠ 0, we say asset i Granger-causes asset j.
    
    Example:
        >>> discovery = GrangerLassoDiscovery(n_lags=1)
        >>> result = discovery.fit(returns_df)
        >>> print(result.n_edges)  # Number of causal edges
        >>> discovery.plot_graph(result, save_path="causal_network.png")
    
    Economic Interpretation:
        - If XLF → XLK with weight 0.15, then a 1% move in Financials
          yesterday predicts a 0.15% move in Tech today (controlling for
          all other sectors).
        - This is actionable: when XLF moves, position in XLK.
    """
    
    def __init__(
        self,
        n_lags: int = 1,
        cv_folds: int = 5,
        alpha_min_ratio: float = 0.001,
        n_alphas: int = 100,
        threshold: float = 0.0,  # 0 means use Lasso's built-in sparsity
        standardize: bool = True,
    ) -> None:
        """Initialize Granger-Lasso discovery.
        
        Args:
            n_lags: Number of lags to include (default 1 for VAR(1)).
            cv_folds: Cross-validation folds for LassoCV.
            alpha_min_ratio: Minimum alpha as ratio of max alpha.
            n_alphas: Number of alphas to try in CV.
            threshold: Additional threshold for edge inclusion.
            standardize: Whether to standardize returns before fitting.
        """
        self.n_lags = n_lags
        self.cv_folds = cv_folds
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas
        self.threshold = threshold
        self.standardize = standardize
    
    def fit(
        self,
        returns: pd.DataFrame | NDArray[np.float64],
        tickers: list[str] | None = None,
    ) -> GrangerLassoResult:
        """Fit Granger-Lasso model to returns data.
        
        Args:
            returns: Returns DataFrame (n_samples, n_assets) or numpy array.
                    Should be log-returns or simple returns.
            tickers: Asset names. Inferred from DataFrame columns if not provided.
        
        Returns:
            GrangerLassoResult containing the causal adjacency matrix.
        
        Note:
            The first `n_lags` observations are lost due to lagging.
        """
        # Convert to numpy if DataFrame
        if isinstance(returns, pd.DataFrame):
            tickers = tickers or list(returns.columns)
            X_full = returns.values
        else:
            X_full = returns
            tickers = tickers or [f"Asset_{i}" for i in range(X_full.shape[1])]
        
        n_samples, n_assets = X_full.shape
        
        # Create lagged features (X at t-1) and targets (X at t)
        X_lagged = X_full[:-self.n_lags]  # t-1 (predictors)
        y_targets = X_full[self.n_lags:]   # t (targets)
        
        # Standardize if requested
        if self.standardize:
            scaler = StandardScaler()
            X_lagged = scaler.fit_transform(X_lagged)
            # Don't standardize targets - we want interpretable coefficients
        
        # Initialize output matrices
        adjacency = np.zeros((n_assets, n_assets))
        r2_scores = np.zeros(n_assets)
        selected_alphas = np.zeros(n_assets)
        
        # Fit Lasso for each target asset
        for j in range(n_assets):
            y_j = y_targets[:, j]
            
            # LassoCV with cross-validation for automatic alpha selection
            lasso_cv = LassoCV(
                cv=self.cv_folds,
                n_alphas=self.n_alphas,
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4,
            )
            
            lasso_cv.fit(X_lagged, y_j)
            
            # Store coefficients: adjacency[i, j] = effect of i on j
            adjacency[:, j] = lasso_cv.coef_
            r2_scores[j] = lasso_cv.score(X_lagged, y_j)
            selected_alphas[j] = lasso_cv.alpha_
        
        # Apply additional threshold if specified
        thresholded = np.where(
            np.abs(adjacency) > self.threshold, 
            adjacency, 
            0
        )
        
        # Compute edge weights (absolute values)
        edge_weights = np.abs(thresholded)
        
        # Count edges
        n_edges = int(np.sum(thresholded != 0))
        
        return GrangerLassoResult(
            adjacency=adjacency,
            thresholded_adjacency=thresholded,
            edge_weights=edge_weights,
            n_edges=n_edges,
            r2_scores=r2_scores,
            selected_alphas=selected_alphas,
            tickers=tickers,
        )
    
    def plot_graph(
        self,
        result: GrangerLassoResult,
        save_path: str | None = None,
        min_edge_weight: float = 0.01,
        figsize: tuple[int, int] = (12, 10),
        node_size: int = 2000,
        font_size: int = 10,
        edge_scale: float = 3.0,
    ) -> None:
        """Visualize the causal network using NetworkX.
        
        Args:
            result: GrangerLassoResult from fit().
            save_path: Path to save figure. If None, displays plot.
            min_edge_weight: Minimum edge weight to display.
            figsize: Figure size.
            node_size: Size of nodes.
            font_size: Font size for labels.
            edge_scale: Scale factor for edge widths.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        tickers = result.tickers or [f"Asset_{i}" for i in range(len(result.adjacency))]
        
        # Add nodes
        G.add_nodes_from(tickers)
        
        # Add edges with weights
        n_assets = len(tickers)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:  # No self-loops
                    weight = result.edge_weights[i, j]
                    if weight > min_edge_weight:
                        # Edge from i to j means i Granger-causes j
                        G.add_edge(
                            tickers[i], 
                            tickers[j], 
                            weight=weight,
                            sign=np.sign(result.thresholded_adjacency[i, j])
                        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.circular_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color='lightblue',
            node_size=node_size,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=font_size,
            font_weight='bold',
            ax=ax
        )
        
        # Draw edges with width proportional to weight
        edges = G.edges(data=True)
        
        # Separate positive and negative edges
        pos_edges = [(u, v) for u, v, d in edges if d.get('sign', 1) > 0]
        neg_edges = [(u, v) for u, v, d in edges if d.get('sign', 1) < 0]
        
        pos_weights = [G[u][v]['weight'] * edge_scale for u, v in pos_edges]
        neg_weights = [G[u][v]['weight'] * edge_scale for u, v in neg_edges]
        
        # Draw positive edges (blue)
        if pos_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=pos_edges,
                width=pos_weights,
                edge_color='blue',
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
        
        # Draw negative edges (red)
        if neg_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=neg_edges,
                width=neg_weights,
                edge_color='red',
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
        
        # Add edge labels for significant edges
        edge_labels = {
            (u, v): f"{d['weight']:.3f}"
            for u, v, d in edges
            if d['weight'] > 0.02  # Only label stronger edges
        }
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                ax=ax
            )
        
        ax.set_title(
            f"Granger Causal Network (Lasso-VAR)\n"
            f"Edges: {G.number_of_edges()} | "
            f"Blue=Positive, Red=Negative",
            fontsize=14
        )
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Graph saved to: {save_path}")
        
        plt.close()
    
    def plot_adjacency_heatmap(
        self,
        result: GrangerLassoResult,
        save_path: str | None = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> None:
        """Plot adjacency matrix as heatmap.
        
        Args:
            result: GrangerLassoResult from fit().
            save_path: Path to save figure.
            figsize: Figure size.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        tickers = result.tickers or [f"Asset_{i}" for i in range(len(result.adjacency))]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use diverging colormap centered at 0
        max_val = np.abs(result.thresholded_adjacency).max()
        
        sns.heatmap(
            result.thresholded_adjacency.T,  # Transpose so row i -> col j
            xticklabels=tickers,
            yticklabels=tickers,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-max_val,
            vmax=max_val,
            ax=ax,
        )
        
        ax.set_title(
            f"Granger-Lasso Causal Matrix\n"
            f"(Row i → Column j: i Granger-causes j)\n"
            f"Total Edges: {result.n_edges}",
            fontsize=12
        )
        ax.set_xlabel("Effect (Target at t)")
        ax.set_ylabel("Cause (Predictor at t-1)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Heatmap saved to: {save_path}")
        
        plt.close()
    
    def get_top_edges(
        self,
        result: GrangerLassoResult,
        n_top: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Get top N strongest causal edges.
        
        Args:
            result: GrangerLassoResult from fit().
            n_top: Number of top edges to return.
        
        Returns:
            List of (cause, effect, weight) tuples.
        """
        tickers = result.tickers or [f"Asset_{i}" for i in range(len(result.adjacency))]
        
        edges = []
        n_assets = len(tickers)
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j and result.edge_weights[i, j] > 0:
                    edges.append((
                        tickers[i],  # Cause
                        tickers[j],  # Effect
                        result.thresholded_adjacency[i, j],  # Signed weight
                    ))
        
        # Sort by absolute weight
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return edges[:n_top]


# =============================================================================
# LEGACY: NOTEARS Implementation (kept for reference/comparison)
# =============================================================================

class NOTEARSResult:
    """Container for NOTEARS optimization result."""
    
    def __init__(
        self,
        adjacency: NDArray[np.float64],
        thresholded_adjacency: NDArray[np.float64],
        loss: float,
        acyclicity_violation: float,
        n_edges: int,
        converged: bool,
    ) -> None:
        self.adjacency = adjacency
        self.thresholded_adjacency = thresholded_adjacency
        self.loss = loss
        self.acyclicity_violation = acyclicity_violation
        self.n_edges = n_edges
        self.converged = converged


class NOTEARS:
    """NOTEARS algorithm for learning DAGs from observational data.
    
    NOTE: This method is deprecated for daily financial data due to
    high contemporaneous correlation. Use GrangerLassoDiscovery instead.
    
    This implementation uses least-squares loss with L1 regularization
    and the trace exponential acyclicity constraint.
    """
    
    def __init__(
        self,
        lambda_l1: float = 0.01,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.1,
    ) -> None:
        self.lambda_l1 = lambda_l1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
    
    def fit(
        self,
        X: NDArray[np.float64],
        method: Literal["linear", "nonlinear"] = "linear",
        ema_span: int | None = None,
    ) -> NOTEARSResult:
        """Learn DAG structure from data."""
        import scipy.linalg as sla
        
        if ema_span is not None and ema_span > 1:
            X = self._apply_ema_smoothing(X, ema_span)
        
        if method == "linear":
            return self._fit_linear(X)
        else:
            raise NotImplementedError("Nonlinear NOTEARS not yet implemented")
    
    @staticmethod
    def _apply_ema_smoothing(
        X: NDArray[np.float64], 
        span: int
    ) -> NDArray[np.float64]:
        """Apply exponential moving average smoothing."""
        df = pd.DataFrame(X)
        smoothed = df.ewm(span=span, adjust=False).mean()
        return smoothed.dropna().values
    
    def _fit_linear(self, X: NDArray[np.float64]) -> NOTEARSResult:
        """Linear NOTEARS using least-squares loss."""
        import scipy.linalg as sla
        from scipy.optimize import minimize
        
        n, d = X.shape
        
        def _loss(W: NDArray) -> float:
            M = X @ W
            return 0.5 * np.sum((X - M) ** 2) / n
        
        def _loss_grad(W: NDArray) -> NDArray:
            M = X @ W
            return -X.T @ (X - M) / n
        
        def _h(W: NDArray) -> float:
            E = sla.expm(W * W)
            return np.trace(E) - d
        
        def _h_grad(W: NDArray) -> NDArray:
            E = sla.expm(W * W)
            return 2 * W * E
        
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        converged = False
        
        for iteration in range(self.max_iter):
            W_new = self._solve_inner(
                W, X, n, d, rho, alpha, _loss, _loss_grad, _h, _h_grad
            )
            h_new = _h(W_new)
            
            if h_new > 0.25 * _h(W):
                rho *= 10
            else:
                alpha += rho * h_new
            
            W = W_new
            
            if abs(h_new) < self.h_tol:
                converged = True
                break
            
            if rho > self.rho_max:
                break
        
        W_thresholded = np.where(np.abs(W) > self.w_threshold, W, 0)
        
        return NOTEARSResult(
            adjacency=W,
            thresholded_adjacency=W_thresholded,
            loss=_loss(W),
            acyclicity_violation=_h(W),
            n_edges=int(np.sum(W_thresholded != 0)),
            converged=converged,
        )
    
    def _solve_inner(
        self,
        W_init: NDArray,
        X: NDArray,
        n: int,
        d: int,
        rho: float,
        alpha: float,
        loss_fn,
        loss_grad_fn,
        h_fn,
        h_grad_fn,
    ) -> NDArray:
        """Solve inner optimization problem using L-BFGS-B."""
        from scipy.optimize import minimize
        
        def objective(w_flat: NDArray) -> float:
            W = w_flat.reshape((d, d))
            h = h_fn(W)
            return (
                loss_fn(W) +
                alpha * h +
                0.5 * rho * h * h +
                self.lambda_l1 * np.sum(np.abs(W))
            )
        
        def gradient(w_flat: NDArray) -> NDArray:
            W = w_flat.reshape((d, d))
            h = h_fn(W)
            grad = (
                loss_grad_fn(W) +
                (alpha + rho * h) * h_grad_fn(W) +
                self.lambda_l1 * np.sign(W)
            )
            return grad.flatten()
        
        result = minimize(
            objective,
            W_init.flatten(),
            method="L-BFGS-B",
            jac=gradient,
            options={"maxiter": 100, "disp": False},
        )
        
        return result.x.reshape((d, d))


def combine_cointegration_and_causal_graphs(
    coint_adjacency: NDArray[np.float64],
    causal_adjacency: NDArray[np.float64],
    alpha: float = 0.5,
) -> NDArray[np.float64]:
    """Combine undirected cointegration graph with directed causal graph.
    
    Economic Logic:
        - Cointegration tells us WHICH assets to trade together
        - Causality tells us WHICH direction to trade
        
        For a pair (XLF, XLK):
        - If cointegrated: they share a stable long-run relationship
        - If XLF → XLK: trade XLK when XLF moves (XLF leads)
    
    Args:
        coint_adjacency: Symmetric cointegration adjacency (n_assets, n_assets).
        causal_adjacency: Asymmetric causal DAG (n_assets, n_assets).
        alpha: Weight for cointegration (1-alpha for causal).
        
    Returns:
        Combined directed adjacency matrix.
    """
    # Normalize both matrices to [0, 1]
    coint_norm = coint_adjacency / (coint_adjacency.max() + 1e-8)
    causal_max = np.abs(causal_adjacency).max()
    causal_norm = causal_adjacency / (causal_max + 1e-8) if causal_max > 0 else causal_adjacency
    
    # Combine: keep cointegration strength but use causal direction
    combined = np.zeros_like(coint_norm)
    
    n = coint_adjacency.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            coint_strength = coint_norm[i, j]
            causal_ij = np.abs(causal_norm[i, j])
            causal_ji = np.abs(causal_norm[j, i])
            
            # Direction: positive if i→j is stronger than j→i
            if causal_ij >= causal_ji:
                combined[i, j] = alpha * coint_strength + (1 - alpha) * causal_ij
            else:
                combined[j, i] = alpha * coint_strength + (1 - alpha) * causal_ji
    
    return combined
