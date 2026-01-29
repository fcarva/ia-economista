# 🇧🇷 Cointegration GNN - IBOVESPA

**Pairs Trading com GNN + Descoberta Causal para Ações Brasileiras**

Sistema de trading quantitativo que combina:
- 📈 **Cointegração** (Teste de Johansen) para encontrar pares estáveis
- 🔍 **Descoberta Causal** (Granger-Lasso) para lead-lag
- 🧠 **Graph Neural Networks** (GraphSAGE) para sinais multi-ativos
- 🤖 **Workflow Agêntico** (LangGraph) para orquestração

---

## 🚀 Quick Start

```bash
# Instalar dependências
pip install -e .

# Rodar demo IBOVESPA
python -m cointegration_gnn.demos.ibov_demo

# Ver resultados em outputs/
```

### 🧰 Workflow Rápido (Makefile)

```bash
make install        # install projeto (editable)
make dashboard      # Streamlit UI
make backtest       # rodar backtest
make train          # rodar treinamento
```

---

## 📁 Estrutura do Projeto

```
cointegration_gnn/
├── config.py              # Configuração (IBOV default)
├── main.py                # Entry point
├── workflow.py            # LangGraph orchestration
│
├── data/                  # Data pipeline
│   └── data_loader.py     # YFinance + stationarity tests
│
├── features/              # Feature engineering
│   ├── cointegration_graph.py  # Rolling Johansen
│   └── causal_discovery.py     # Granger-Lasso
│
├── models/                # Deep learning
│   └── gnn_model.py       # GraphSAGE + Sharpe loss
│
├── strategy/              # Trading logic
│   └── strategy.py        # Kelly sizing
│
├── backtest/              # Backtesting
│   └── backtest_engine.py # Event-driven engine
│
├── agents/                # LangGraph agents
│   ├── analyst.py         # Signal generation
│   ├── risk_manager.py    # Risk filtering
│   └── executor.py        # Order execution
│
├── validation/            # Sanity checks
│   └── sanity_checks.py   # Stationarity, Randomization
│
└── demos/                 # Demos & examples
    └── ibov_demo.py       # IBOVESPA demo
```

---

## 🇧🇷 Universe (Blue Chips B3)

| Ticker | Empresa | Setor |
|--------|---------|-------|
| PETR4 | Petrobras | Energia |
| VALE3 | Vale | Mineração |
| ITUB4 | Itaú | Financeiro |
| BBDC4 | Bradesco | Financeiro |
| ABEV3 | Ambev | Consumo |
| B3SA3 | B3 | Exchange |
| WEGE3 | WEG | Industrial |
| RENT3 | Localiza | Consumo Disc |
| BBAS3 | Banco do Brasil | Financeiro |

---

## 📊 Data Splits (Regime-Aware)

| Split | Período | Regime |
|-------|---------|--------|
| **Train** | 2016-2020 | Temer + COVID |
| **Validation** | 2021-2022 | Bolsonaro final |
| **Test** | 2023-2026 | Lula 3 |

---

## 🔬 Resultados Fase 1

| Métrica | Valor |
|---------|-------|
| Trading Days | 2,509 |
| Cointegração | 100% densidade |
| Causal Edge | `ITUB4 → BBAS3` |

---

## 📈 Próximos Passos

- [ ] Treinar GraphSAGE com grafo combinado
- [ ] Backtesting com custos de transação
- [ ] Validação walk-forward
- [ ] Deploy com API REST

---

## 🛠️ Tech Stack

- **Data**: yfinance, pandas
- **Econometria**: statsmodels (ADF, KPSS, Johansen)
- **ML**: PyTorch, PyTorch Geometric, Lightning
- **Agents**: LangGraph
- **Viz**: matplotlib, seaborn, networkx

---

## 📜 License

MIT
