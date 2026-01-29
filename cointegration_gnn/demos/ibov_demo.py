"""
Demo Script: IBOVESPA Data Pipeline & Graph
============================================

Versão do demo Phase 1 adaptada para ações brasileiras (B3).

Uso:
    python -m cointegration_gnn.demos.ibov_demo
"""

import warnings
from datetime import date
from pathlib import Path

# Use non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suprimir warnings
warnings.filterwarnings("ignore")

# Imports internos
from cointegration_gnn.config import default_config
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph
from cointegration_gnn.features.causal_discovery import GrangerLassoDiscovery


def main() -> None:
    """Executa demo IBOVESPA."""
    
    print("=" * 70)
    print("COINTEGRATION GNN - IBOVESPA BLUE CHIPS")
    print("=" * 70)
    
    config = default_config
    tickers = config.data.tickers
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Universe: {len(tickers)} ações brasileiras")
    for t in tickers:
        print(f"   - {t}")
    
    # =========================================================================
    # PARTE 1: FETCH DE DADOS
    # =========================================================================
    print("\n[PARTE 1] FETCHING DATA FROM B3")
    print("-" * 50)
    
    loader = DataLoader(tickers=tickers)
    
    prices, returns = loader.fetch_data(
        start=config.data.train_start,
        end=config.data.test_end,
    )
    
    print(f"✓ Período: {prices.index[0].strftime('%Y-%m-%d')} a {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Trading days: {len(prices):,}")
    print(f"✓ Ativos: {len(tickers)}")
    
    # =========================================================================
    # PARTE 2: STATIONARITY TESTS
    # =========================================================================
    print("\n[PARTE 2] STATIONARITY TESTS")
    print("-" * 50)
    
    stationarity = loader.test_stationarity(prices)
    
    print("\n| Ticker    | I(d) | ADF p-val | KPSS p-val |")
    print("|-----------|------|-----------|------------|")
    
    for ticker, result in stationarity.items():
        status = "✓" if result.integration_order == 1 else "✗"
        print(f"| {ticker:9} | I({result.integration_order}) | {result.adf_pvalue:9.4f} | {result.kpss_pvalue:10.4f} |")
    
    all_i1 = all(r.integration_order == 1 for r in stationarity.values())
    print(f"\n{'✓' if all_i1 else '✗'} Todos os ativos são I(1): {all_i1}")
    
    # =========================================================================
    # PARTE 3: COINTEGRATION MATRIX
    # =========================================================================
    print("\n[PARTE 3] COINTEGRATION MATRIX (JOHANSEN)")
    print("-" * 50)
    
    coint_graph = CointegrationGraph(
        tickers=tickers,
        rolling_window=252,
        min_edge_weight=0.3,
    )
    
    # Calcular adjacência no final de 2019 (pré-COVID)
    adjacency_2019 = coint_graph.compute_adjacency(
        prices[prices.index <= "2019-12-31"],
        as_of_date=pd.Timestamp("2019-12-31"),
    )
    
    print(f"✓ Edges em 2019: {(adjacency_2019 > 0).sum() // 2}")
    density = (adjacency_2019 > 0).sum() / (len(tickers) * (len(tickers) - 1))
    print(f"✓ Densidade: {density:.2%}")
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Remover .SA para labels mais limpos
    clean_tickers = [t.replace(".SA", "") for t in tickers]
    
    sns.heatmap(
        adjacency_2019,
        xticklabels=clean_tickers,
        yticklabels=clean_tickers,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        vmin=0,
        vmax=2,
    )
    ax.set_title("Matriz de Cointegração - IBOVESPA Blue Chips (Dez/2019)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ibov_cointegration_matrix.png", dpi=150)
    print(f"✓ Gráfico salvo: {output_dir}/ibov_cointegration_matrix.png")
    plt.close()
    
    # =========================================================================
    # PARTE 4: CAUSAL DISCOVERY (Granger-Lasso)
    # =========================================================================
    print("\n[PARTE 4] CAUSAL DISCOVERY (Granger-Lasso)")
    print("-" * 50)
    
    returns_train = returns[
        (returns.index >= str(config.data.train_start)) & 
        (returns.index <= str(config.data.train_end))
    ]
    
    print(f"  Período: {returns_train.index[0].strftime('%Y-%m-%d')} a {returns_train.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Model: r_{{j,t}} = Σ_i β_{{i,j}} * r_{{i,t-1}} + ε")
    
    granger = GrangerLassoDiscovery(
        n_lags=1,
        cv_folds=5,
        standardize=True,
    )
    causal_result = granger.fit(returns_train, tickers=clean_tickers)
    
    print(f"✓ Causal edges detected: {causal_result.n_edges}")
    print(f"✓ Mean R² across targets: {causal_result.r2_scores.mean():.4f}")
    
    # Plot heatmap
    granger.plot_adjacency_heatmap(causal_result, save_path=str(output_dir / "ibov_causal_heatmap.png"))
    
    # Plot network
    granger.plot_graph(
        causal_result, 
        save_path=str(output_dir / "ibov_causal_network.png"),
        min_edge_weight=0.0005,
        edge_scale=500.0,
    )
    
    # Show top edges
    print("\nTop 10 Causal Edges (Lead → Lag):")
    top_edges = granger.get_top_edges(causal_result, n_top=10)
    for i, (cause, effect, weight) in enumerate(top_edges, 1):
        sign = "+" if weight > 0 else "-"
        print(f"  {i:2}. {cause:6} → {effect:6} ({sign}{abs(weight):.4f})")
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESUMO IBOVESPA")
    print("=" * 70)
    print(f"""
✓ Dados carregados: {len(prices)} dias, {len(tickers)} ativos
✓ Todos os ativos são I(1): {all_i1}
✓ Grafo de cointegração: {(adjacency_2019 > 0).sum() // 2} edges
✓ Grafo causal: {causal_result.n_edges} edges

ARQUIVOS GERADOS em {output_dir}/:
- ibov_cointegration_matrix.png
- ibov_causal_heatmap.png
- ibov_causal_network.png

OBSERVAÇÃO ECONÔMICA BRASIL:
- Alta concentração em commodities (PETR4, VALE3) → alta correlação com dólar
- Bancos (ITUB4, BBDC4, BBAS3) tendem a se mover juntos
- Regimes brasileiros: Dilma→Temer→Bolsonaro→Lula afetam estrutura
""")


if __name__ == "__main__":
    main()
