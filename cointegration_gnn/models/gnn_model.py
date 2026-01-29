"""
Graph Neural Network Model Module
==================================

This module implements a GraphSAGE-based neural network for learning
trading signals from the cointegration graph.

Economic Logic:
    Traditional pairs trading uses a simple z-score threshold:
    - Buy spread if z < -2
    - Sell spread if z > 2
    
    This is suboptimal because:
    1. Threshold is static (ignores volatility regimes)
    2. Ignores multi-asset information (other spreads in the graph)
    3. No learning of optimal entry/exit timing
    
    Our GNN approach:
    1. Each node (asset) has features: return, vol, z-score, momentum
    2. Edges encode cointegration+causal relationships
    3. Message passing aggregates information from related assets
    4. Output: optimal position size for each asset [-1, 1]

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │              GraphSAGE Encoder                          │
    ├─────────────────────────────────────────────────────────┤
    │  Input: (n_assets, 8) node features                     │
    │  Edges: from cointegration + causal graphs              │
    │                                                         │
    │  SAGEConv(8 → 32) → ReLU → Dropout                     │
    │  SAGEConv(32 → 32) → ReLU → Dropout                    │
    │  SAGEConv(32 → 16) → ReLU → Dropout                    │
    │                                                         │
    │  Node Readout: MLP(16 → 8 → 1) → Tanh                  │
    │  Output: (n_assets,) positions ∈ [-1, 1]               │
    └─────────────────────────────────────────────────────────┘

References:
    - Hamilton et al. (2017). "Inductive Representation Learning on
      Large Graphs" (GraphSAGE)
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool
from cointegration_gnn.models.losses import ListFoldLoss

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl


class GATv2Encoder(nn.Module):
    """GATv2 encoder for attention-weighted node representations.
    
    Unlike GraphSAGE (uniform aggregation), GATv2 learns *which* neighbors
    matter most via attention coefficients. This is critical for finance:
    a shock in Petrobras should influence IBOV more than a shock in WEG.
    
    Architecture (Track 2):
        - Multi-head attention (heads=4) to capture different relation types
        - Last layer uses concat=False (mean) to reduce dimensionality
        - Layer normalization for training stability
    
    References:
        - Brody et al. (2021) "How Attentive are Graph Attention Networks?"
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        """Initialize GATv2 encoder.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer dimension (per head).
            out_channels: Output embedding dimension.
            num_layers: Number of GAT layers.
            heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        
        from torch_geometric.nn import GATv2Conv
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.heads = heads
        
        # First layer: in_channels -> hidden_channels * heads
        self.convs.append(GATv2Conv(
            in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True,
            edge_dim=1  # Edge weights from cointegration graph
        ))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers: hidden_channels * heads -> hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(
                hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True,
                edge_dim=1
            ))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Output layer: hidden_channels * heads -> out_channels (concat=False for mean)
        self.convs.append(GATv2Conv(
            hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False,
            edge_dim=1
        ))
        self.norms.append(nn.LayerNorm(out_channels))
        
        # Store attention weights for explainability
        self._attention_weights: list[Tensor] = []
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Forward pass through GATv2 layers.
        
        Args:
            x: Node features (n_nodes, in_channels).
            edge_index: Edge connectivity (2, n_edges).
            edge_weight: Optional edge weights (n_edges,). Used as edge_attr.
            return_attention_weights: If True, return attention coefficients.
            
        Returns:
            Node embeddings (n_nodes, out_channels).
            If return_attention_weights: also returns list of (edge_index, alpha) tuples.
        """
        self._attention_weights = []
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GATv2 convolution with attention
            if return_attention_weights:
                x, (edge_idx, alpha) = conv(
                    x, edge_index, 
                    edge_attr=edge_weight.unsqueeze(-1) if edge_weight is not None else None,
                    return_attention_weights=True
                )
                self._attention_weights.append((edge_idx, alpha))
            else:
                x = conv(
                    x, edge_index,
                    edge_attr=edge_weight.unsqueeze(-1) if edge_weight is not None else None,
                )
            
            x = norm(x)
            
            if i < len(self.convs) - 1:  # Not last layer
                x = F.elu(x)  # ELU often works better with attention
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_attention_weights:
            return x, self._attention_weights
        return x
    
    def get_attention_weights(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Extract attention weights for explainability.
        
        This method enables XAI by showing which neighbors the model
        focused on when making predictions.
        
        Example:
            >>> attention = model.encoder.get_attention_weights(x, edge_index)
            >>> for layer_idx, (edges, alpha) in enumerate(attention):
            ...     print(f"Layer {layer_idx}: max attention = {alpha.max():.3f}")
        
        Returns:
            List of (edge_index, alpha) tuples per layer.
            alpha shape: (n_edges, heads) for intermediate layers, (n_edges, 1) for last.
        """
        self.eval()
        with torch.no_grad():
            _, attention = self.forward(x, edge_index, edge_weight, return_attention_weights=True)
        return attention


# Keep GraphSAGEEncoder as fallback option
class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for node-level representations (Legacy).
    
    Uses mean aggregation (most stable for financial data) and
    residual connections for gradient flow.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        """Initialize GraphSAGE encoder.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer dimension.
            out_channels: Output embedding dimension.
            num_layers: Number of SAGE convolution layers.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.norms.append(nn.LayerNorm(out_channels))
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through GraphSAGE layers.
        
        Args:
            x: Node features (n_nodes, in_channels).
            edge_index: Edge connectivity (2, n_edges).
            edge_weight: Optional edge weights (n_edges,).
            
        Returns:
            Node embeddings (n_nodes, out_channels).
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # SAGE convolution (edge_weight used implicitly via neighborhood)
            x = conv(x, edge_index)
            x = norm(x)
            
            if i < len(self.convs) - 1:  # Not last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class PositionHead(nn.Module):
    """MLP head to predict position sizes from node embeddings.
    
    Economic Logic:
        The output is a position size in [-1, 1] for each asset:
        - +1 = maximum long position
        - -1 = maximum short position
        - 0 = no position
        
        We use Tanh activation to bound the output, ensuring positions
        don't explode during backpropagation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),  # Bound output to [-1, 1]
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Predict position sizes.
        
        Args:
            x: Node embeddings (n_nodes, in_channels).
            
        Returns:
            Position sizes (n_nodes,) in range [-1, 1].
        """
        return self.mlp(x).squeeze(-1)


class ScoreHead(nn.Module):
    """MLP head to predict ranking scores from node embeddings.
    
    For Learning to Rank (ListMLE/ListFold):
        The output is an UNBOUNDED score for each asset.
        Higher score = model predicts higher return.
        
        Unlike PositionHead, we do NOT use Tanh:
        - Scores are used for ranking, not as positions
        - Unbounded scores allow better gradient flow
        - Actual positions are derived from score ranking
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            # NO Tanh - unbounded scores for ranking
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Predict ranking scores.
        
        Args:
            x: Node embeddings (n_nodes, in_channels).
            
        Returns:
            Ranking scores (n_nodes,) - unbounded.
        """
        return self.mlp(x).squeeze(-1)


class CointegrationGNN(pl.LightningModule):
    """PyTorch Lightning module for cointegration-based trading.
    
    This model learns to generate trading signals from the dynamic
    cointegration graph by maximizing risk-adjusted returns (Sharpe ratio).
    
    Economic Interpretation:
        The model learns which spreads to trade and when. Information
        propagates through the graph: if XLF deviates from equilibrium,
        the model considers its cointegrated neighbors (XLK, XLE) to
        decide on the optimal portfolio adjustment.
    
        propagates through cointegration edges.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        transaction_cost_bps: float = 10.0,
        turnover_penalty: float = 0.5,
        encoder_type: Literal["gatv2", "sage"] = "gatv2",
        attention_heads: int = 4,
        use_ranking_loss: bool = True,
    ) -> None:
        """Initialize CointegrationGNN.
        
        Args:
            node_feature_dim: Dimension of input node features.
            hidden_dim: Hidden dimension for convolutions.
            num_layers: Number of GNN layers.
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization weight.
            transaction_cost_bps: Transaction cost in basis points.
            turnover_penalty: Penalty weight for position changes (0.0-1.0).
            encoder_type: "gatv2" (attention) or "sage" (uniform).
            attention_heads: Number of attention heads (only for GATv2).
            use_ranking_loss: If True, use ListFold loss for Learning to Rank.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder_type = encoder_type
        self.use_ranking_loss = use_ranking_loss
        
        if encoder_type == "gatv2":
            self.encoder = GATv2Encoder(
                in_channels=node_feature_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                num_layers=num_layers,
                heads=attention_heads,
                dropout=dropout,
            )
        else:
            self.encoder = GraphSAGEEncoder(
                in_channels=node_feature_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                num_layers=num_layers,
                dropout=dropout,
            )
        
        # Determine Head & Loss Function
        if use_ranking_loss:
            self.score_head = ScoreHead(hidden_dim // 2, 16)
            self.loss_fn = ListFoldLoss()  # 🆕 Modular Loss
        else:
            self.position_head = PositionHead(hidden_dim // 2, 8)
            # Legacy loss is computed inline
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.transaction_cost = transaction_cost_bps / 10000  # Convert to decimal
        self.turnover_penalty = turnover_penalty
    
    def forward(self, data: Data) -> Tensor:
        """Generate scores or positions from graph.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (n_nodes, n_features)
                - edge_index: Edge connectivity (2, n_edges)
                - edge_attr: Edge weights (n_edges, 1)
                
        Returns:
            If use_ranking_loss: Ranking scores (n_nodes,) - unbounded
            Else: Position signals (n_nodes,) in range [-1, 1]
        """
        # Encode nodes using graph structure
        embeddings = self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr.squeeze(-1) if data.edge_attr is not None else None,
        )
        
        # Predict scores or positions based on mode
        if self.use_ranking_loss:
            return self.score_head(embeddings)
        else:
            return self.position_head(embeddings)
    
    def compute_portfolio_return(
        self,
        positions: Tensor,
        returns: Tensor,
        prev_positions: Tensor | None = None,
    ) -> Tensor:
        """Compute portfolio return with transaction costs.
        
        Economic Logic:
            r_portfolio = Σ w_i * r_i - TC * Σ |Δw_i|
            
            where TC is transaction cost and Δw is position change.
        
        Args:
            positions: Current positions (n_assets,).
            returns: Asset returns for this period (n_assets,).
            prev_positions: Previous positions (for turnover calculation).
            
        Returns:
            Scalar portfolio return.
        """
        # Normalize positions to sum to 1 (long-short portfolio)
        weights = positions / (torch.abs(positions).sum() + 1e-8)
        
        # Gross return
        gross_return = (weights * returns).sum()
        
        # Transaction costs
        if prev_positions is not None:
            prev_weights = prev_positions / (torch.abs(prev_positions).sum() + 1e-8)
            turnover = torch.abs(weights - prev_weights).sum()
            tc = self.transaction_cost * turnover
        else:
            tc = 0.0
        
        return gross_return - tc
    
    def differentiable_sharpe(
        self,
        returns: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Compute differentiable Sharpe ratio for training loss.
        
        Economic Logic:
            Sharpe = E[r] / σ[r]
            
            We differentiate through the mean and std to optimize
            the risk-adjusted return directly.
            
        **IMPORTANT: Overfitting Risk**
            Sharpe optimization can easily overfit. Use with:
            - Walk-forward validation
            - Randomization tests
            - Strong regularization
        
        Args:
            returns: Sequence of portfolio returns.
            eps: Small constant for numerical stability.
            
        Returns:
            Negative Sharpe (for minimization).
        """
        mean_return = returns.mean()
        std_return = returns.std() + eps
        
        sharpe = mean_return / std_return
        
        # Return negative for minimization
        return -sharpe
    
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Training step with ListFold or Direct Return Loss.
        
        If use_ranking_loss=True (Learning to Rank):
            Uses ListFold loss to optimize ranking accuracy.
            The model learns WHICH assets will outperform, not by how much.
            
        Else (Legacy mode):
            Uses Direct Return Loss to maximize portfolio return.
            
        Args:
            batch: Batched graph data with returns in batch.y.
            batch_idx: Batch index.
            
        Returns:
            Loss tensor.
        """
        # 1. Generate scores (if ranking) or positions (if legacy)
        outputs = self(batch)
        
        # 2. Reshape outputs and returns per graph (snapshot)
        num_graphs = batch.num_graphs
        num_nodes_per_graph = outputs.shape[0] // num_graphs
        
        # Reshape: [Batch_Size, Num_Assets]
        output_matrix = outputs.view(num_graphs, num_nodes_per_graph)
        
        # Get target returns (also stacked)
        returns = batch.y if hasattr(batch, "y") and batch.y is not None else torch.zeros_like(outputs)
        ret_matrix = returns.view(num_graphs, num_nodes_per_graph)
        
        if self.use_ranking_loss:
            # ===== LEARNING TO RANK MODE =====
            # Compute ListFold loss for each graph in batch, then average
            listfold_losses = []
            ics = []  # Information Coefficient (Spearman correlation)
            
            for i in range(num_graphs):
                scores_i = output_matrix[i]
                returns_i = ret_matrix[i]
                
                # Skip if all returns are the same (no ranking to learn)
                if returns_i.std() < 1e-8:
                    continue
                    
                # Use modular Loss Function
                loss_i = self.loss_fn(scores_i, returns_i)
                listfold_losses.append(loss_i)
                
                # Compute Information Coefficient (Spearman rank correlation)
                # This is the KEY metric for ranking quality
                with torch.no_grad():
                    score_ranks = scores_i.argsort().argsort().float()
                    return_ranks = returns_i.argsort().argsort().float()
                    # Spearman = Pearson on ranks
                    ic = torch.corrcoef(torch.stack([score_ranks, return_ranks]))[0, 1]
                    if not torch.isnan(ic):
                        ics.append(ic)
            
            if len(listfold_losses) == 0:
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
            total_loss = torch.stack(listfold_losses).mean()
            
            # Logging for Learning to Rank
            self.log("train/loss", total_loss, prog_bar=True)
            self.log("train/listfold", total_loss)
            if len(ics) > 0:
                mean_ic = torch.stack(ics).mean()
                self.log("train/IC", mean_ic, prog_bar=True)  # Information Coefficient
            self.log("train/score_std", output_matrix.std())  # Score spread
            
        else:
            # ===== LEGACY DIRECT RETURN MODE =====
            pos_matrix = output_matrix  # These are positions in legacy mode
            
            # Compute Portfolio Returns
            portfolio_returns = (pos_matrix * ret_matrix).sum(dim=1)
            
            # Transaction cost approximation
            if num_graphs > 1:
                turnover = torch.abs(pos_matrix[1:] - pos_matrix[:-1]).sum(dim=1)
                cost = self.transaction_cost * turnover
                portfolio_returns_net = portfolio_returns[1:] - cost
            else:
                portfolio_returns_net = portfolio_returns
            
            # Loss components
            return_loss = -100.0 * portfolio_returns_net.mean()
            mag_penalty = 0.01 * (pos_matrix ** 2).mean()
            
            if num_graphs > 1:
                turnover_loss = torch.abs(pos_matrix[1:] - pos_matrix[:-1]).mean()
            else:
                turnover_loss = torch.tensor(0.0, device=outputs.device)
            
            net_exposure = pos_matrix.sum(dim=1)
            net_exposure_loss = 0.1 * (net_exposure ** 2).mean()
            
            total_loss = (
                return_loss 
                + mag_penalty 
                + (self.turnover_penalty * turnover_loss) 
                + net_exposure_loss
            )
            
            # Logging for legacy mode
            self.log("train/loss", total_loss, prog_bar=True)
            self.log("train/return", -return_loss / 100.0, prog_bar=True)
            self.log("train/turnover", turnover_loss)
            self.log("train/position_magnitude", pos_matrix.abs().mean())
        
        return total_loss
    
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Validation step."""
        positions = self(batch)
        returns = batch.y if hasattr(batch, "y") and batch.y is not None else torch.zeros_like(positions)
        
        if self.use_ranking_loss:
            # ===== LEARNING TO RANK MODE =====
            # Compute IC (Spearman Correlation) per graph in batch
            num_graphs = batch.num_graphs
            num_nodes_per_graph = positions.shape[0] // num_graphs
            
            # Reshape
            scores_matrix = positions.view(num_graphs, num_nodes_per_graph)
            ret_matrix = returns.view(num_graphs, num_nodes_per_graph)
            
            ics = []
            for i in range(num_graphs):
                s = scores_matrix[i]
                r = ret_matrix[i]
                
                if r.std() < 1e-8:
                    continue
                    
                s_rank = s.argsort().argsort().float()
                r_rank = r.argsort().argsort().float()
                
                ic = torch.corrcoef(torch.stack([s_rank, r_rank]))[0, 1]
                if not torch.isnan(ic):
                    ics.append(ic)
            
            val_ic = torch.stack(ics).mean() if ics else torch.tensor(0.0, device=positions.device)
            self.log("val/IC", val_ic, prog_bar=True)
            self.log("val/return", val_ic) # Use IC as return proxy for EarlyStopping
            return val_ic
            
        else:
            # ===== LEGACY MODE =====
            portfolio_return = self.compute_portfolio_return(positions, returns)
            self.log("val/return", portfolio_return, prog_bar=True)
            self.log("val/position_magnitude", positions.abs().mean())
            return portfolio_return
    
    def configure_optimizers(self):
        """Configure Adam optimizer with weight decay."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        return optimizer
    
    def predict_positions(self, data: Data) -> dict[str, float]:
        """Generate human-readable position predictions.
        
        Args:
            data: Graph data with node features and edges.
            
        Returns:
            Dictionary mapping asset names to positions.
        """
        self.eval()
        with torch.no_grad():
            positions = self(data).cpu().numpy()
        
        # Assuming data has node names in data.tickers attribute
        tickers = getattr(data, "tickers", [f"Asset_{i}" for i in range(len(positions))])
        
        return {ticker: float(pos) for ticker, pos in zip(tickers, positions)}


class RandomizedBaselineModel:
    """Baseline model that generates random signals for comparison.
    
    Economic Logic:
        Any profitable strategy must beat a random baseline. If the
        GNN doesn't significantly outperform this, it's likely fitting
        noise rather than a real signal.
    
    Usage in validation:
        >>> baseline = RandomizedBaselineModel(n_assets=9)
        >>> baseline_sharpe = backtest(baseline.generate_positions())
        >>> gnn_sharpe = backtest(gnn.generate_positions())
        >>> assert gnn_sharpe > baseline_sharpe + 0.5  # Significant edge
    """
    
    def __init__(self, n_assets: int, seed: int = 42) -> None:
        self.n_assets = n_assets
        self.rng = torch.Generator().manual_seed(seed)
    
    def generate_positions(self) -> Tensor:
        """Generate random positions."""
        # Random uniform in [-1, 1]
        positions = 2 * torch.rand(self.n_assets, generator=self.rng) - 1
        return positions
    
    def generate_shuffled_positions(self, real_positions: Tensor) -> Tensor:
        """Shuffle real positions (preserves distribution, breaks timing).
        
        This is a powerful test: if shuffled positions have similar
        Sharpe to real positions, the timing signal is worthless.
        """
        perm = torch.randperm(len(real_positions), generator=self.rng)
        return real_positions[perm]
