# Workspace Setup — AI Economist

This guide makes the workspace ready for research, backtests, and dashboard work.

## 1) Python Environment (Poetry)
```bash
poetry install
poetry shell
```

## 2) Quick Commands (Makefile)
Use the Makefile at the repo root to standardize workflows:

```bash
make install        # install project (editable)
make install-dev    # install dev dependencies
make dashboard      # run Streamlit UI
make backtest       # run backtest script
make train          # run training script
make lint           # run ruff + black check
make format         # run ruff + black format
make test           # run pytest
```

## 3) Data & Outputs
- Backtests output to `results/`.
- Dashboard uses data from `data/` and `results/`.

## 4) Suggested Local Env Vars
Create a `.env` file if you need to override defaults (e.g., data paths or API keys).
