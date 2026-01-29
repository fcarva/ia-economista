"""
Configuration Module for Cointegration GNN
==========================================

This module defines all hyperparameters and settings using Pydantic for
runtime validation. The design follows the "Configuration as Code" principle,
allowing reproducible experiments.

Economic Logic:
- Asset universe is sector ETFs to capture macro-level cointegration
- Rolling windows are calibrated to capture regime changes (252 days ~ 1 year)
- Train/Val/Test splits are regime-aware (bull/crisis/inflation)
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data fetching and preprocessing.
    
    Economic Rationale:
        We use the top 9 most liquid Brazilian stocks (Blue Chips) because
        they represent the structure of the Brazilian economy. Cointegration
        between sectors is economically interpretable.
    """
    
    tickers: list[str] = Field(
        default=[
            "PETR4.SA",  # Petrobras PN - Energy/Oil
            "VALE3.SA",  # Vale ON - Mining/Commodities
            "ITUB4.SA",  # Itaú Unibanco PN - Financials
            "BBDC4.SA",  # Bradesco PN - Financials
            "ABEV3.SA",  # Ambev ON - Consumer Staples
            "B3SA3.SA",  # B3 ON - Financials/Exchange
            "WEGE3.SA",  # WEG ON - Industrials
            "RENT3.SA",  # Localiza ON - Consumer Discretionary
            "BBAS3.SA",  # Banco do Brasil ON - Financials
        ],
        description="Universe of Brazilian Blue Chips to analyze for cointegration."
    )
    
    # Brazil regime-aware data splits
    # Brazil regime-aware data splits
    train_start: date = Field(default=date(2016, 1, 1))   # Post Dilma impeachment
    train_end: date = Field(default=date(2023, 12, 31))   # Extended Training (Includes COVID + Post-COVID)
    val_start: date = Field(default=date(2024, 1, 1))     # Recent Validation
    val_end: date = Field(default=date(2024, 12, 31))     # 1 Year Val
    test_start: date = Field(default=date(2025, 1, 1))    # Recent Test
    test_end: date = Field(default=date(2026, 1, 28))     # Up to Today
    
    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        """Ensure we have at least 3 tickers for meaningful cointegration."""
        if len(v) < 3:
            raise ValueError("Need at least 3 assets for multi-asset cointegration")
        return [t.upper() for t in v]


class CointegrationConfig(BaseModel):
    """Configuration for cointegration graph construction.
    
    Economic Rationale:
        - Rolling window of 252 days (1 trading year) captures regime dynamics
        - Johansen test handles multivariate cointegration (APT factor structure)
        - Significance level of 5% balances Type I/II errors for trading
    """
    
    rolling_window: int = Field(
        default=252,
        ge=60,
        le=504,
        description="Rolling window for Johansen test (trading days)."
    )
    
    johansen_det_order: Literal[-1, 0, 1] = Field(
        default=0,
        description="Deterministic term: -1=no const, 0=const in coint, 1=const+trend."
    )
    
    significance_level: float = Field(
        default=0.05,
        ge=0.01,
        le=0.10,
        description="Significance level for cointegration tests."
    )
    
    min_edge_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum normalized trace statistic to form an edge."
    )


class CausalConfig(BaseModel):
    """Configuration for NOTEARS causal discovery.
    
    Economic Rationale:
        NOTEARS learns a Directed Acyclic Graph (DAG) representing causal
        relationships. In market microstructure, this captures information
        flow (e.g., Financials -> Tech during rate hikes).
        
        The acyclicity constraint is enforced via: h(W) = tr(e^{W∘W}) - d = 0
        
    Note on Daily Data:
        Daily returns are noisy (nearly Brownian). To reveal causal structure:
        1. Lower threshold (0.005 vs 0.1) - small effects are meaningful
        2. Higher L1 penalty (0.05 vs 0.01) - enforce sparsity
        3. Apply EMA smoothing before NOTEARS - reduces HF noise
    """
    
    lambda_l1: float = Field(
        default=0.01,  # Moderate L1 - balances sparsity vs edge detection
        ge=0.0,
        description="L1 regularization for sparsity in causal graph."
    )
    
    max_iter: int = Field(
        default=100,
        ge=10,
        description="Maximum iterations for NOTEARS optimization."
    )
    
    h_tol: float = Field(
        default=1e-8,
        description="Tolerance for acyclicity constraint violation."
    )
    
    w_threshold: float = Field(
        default=0.001,  # Very low threshold - any edge above noise
        ge=0.0,
        le=1.0,
        description="Threshold for edge weight to be considered causal."
    )
    
    ema_smoothing_span: int = Field(
        default=5,  # 5-day EMA for more smoothing of daily noise
        ge=1,
        le=10,
        description="EMA span for smoothing returns before NOTEARS."
    )


class GNNConfig(BaseModel):
    """Configuration for GraphSAGE model.
    
    Economic Rationale:
        GraphSAGE is inductive (generalizes to new assets) and learns to
        aggregate information from neighboring assets in the cointegration
        graph. This naturally captures multi-asset spread relationships.
    """
    
    node_feature_dim: int = Field(
        default=16,  # Increased to accommodate macro features (usdbrl, vix, etc.)
        description="Dimension of node features (returns, vol, z-score, macro, etc.)."
    )
    
    hidden_dim: int = Field(
        default=32,
        ge=16,
        le=128,
        description="Hidden dimension for SAGE convolutions."
    )
    
    num_layers: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of SAGE convolutional layers."
    )
    
    dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Dropout rate for regularization."
    )
    
    learning_rate: float = Field(
        default=1e-3,
        description="Learning rate for Adam optimizer."
    )
    
    weight_decay: float = Field(
        default=1e-4,
        description="L2 regularization (weight decay) to prevent overfitting."
    )


class BacktestConfig(BaseModel):
    """Configuration for backtesting engine.
    
    Economic Rationale:
        Transaction costs and slippage are critical for realistic PnL.
        10bps round-trip is a reasonable estimate for liquid ETFs.
        Position limits prevent over-concentration in single sectors.
    """
    
    transaction_cost_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description="Round-trip transaction cost in basis points."
    )
    
    slippage_bps: float = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Slippage estimate in basis points."
    )
    
    max_position_pct: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Maximum position size as fraction of portfolio."
    )
    
    initial_capital: float = Field(
        default=1_000_000.0,
        ge=10_000.0,
        description="Initial capital in USD."
    )


class Config(BaseModel):
    """Master configuration aggregating all sub-configs.
    
    Usage:
        config = Config()
        config.data.tickers  # ['XLK', 'XLF', ...]
        config.gnn.hidden_dim  # 32
    """
    
    data: DataConfig = Field(default_factory=DataConfig)
    cointegration: CointegrationConfig = Field(default_factory=CointegrationConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    # Random seed for reproducibility
    seed: int = Field(default=42)
    
    # Device configuration
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")


# Default configuration instance (IBOVESPA Blue Chips)
default_config = Config()

