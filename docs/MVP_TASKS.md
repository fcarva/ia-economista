# MVP Task Board — AI Economist

## Goal
Ship a **Bloomberg-lite, minimal Flexoki/Kepano** dashboard with credible backtests and a robust IBOV benchmark, suitable for investor demos.

---

## Phase 0 — Foundations (1–2 weeks)
**Data & Benchmark**
- [ ] Add IBOV benchmark ingestion (`^BVSP` or BOVA11) and align it with strategy returns.
- [ ] Log benchmark stats (return, Sharpe, drawdown, tracking error) alongside strategy.

**Telemetry**
- [ ] Save a canonical `latest_metrics.json` after each backtest (to power the dashboard).
- [ ] Add data freshness metadata (last pull, last training, last backtest).

**UX/UI**
- [ ] Define all CSS theme tokens in `dashboard/utils.py` (Flexoki minimal palette).
- [ ] Replace placeholder KPIs with live data from `latest_metrics.json`.

---

## Phase 1 — Core Model & Validation (3–6 weeks)
**Model & Research**
- [ ] Walk-forward training + evaluation (rolling windows with reset).
- [ ] Evaluate regime-specific performance and feature attribution stability.
- [ ] Add benchmark-aware bootstrap validation (probability Sharpe > benchmark).

**Risk & Execution**
- [ ] Implement risk caps (max position size, leverage, sector caps).
- [ ] Add transaction cost sensitivity analysis (5–30 bps).

---

## Phase 2 — Productization (6–10 weeks)
**Dashboard**
- [ ] Introduce a top-of-screen market tape (prices, signals, regime).
- [ ] Add portfolio drill-down (weights, exposures, turnover, PnL attribution).
- [ ] Add explainability panel (graph edges + top contributing signals).

**Infra**
- [ ] Add experiment registry (log runs, configs, model versioning).
- [ ] Automate nightly data pulls + backtest runs (cron or CI).

---

## Definition of Done (MVP)
- [ ] Strategy backtest vs IBOV with tracking error + IR displayed.
- [ ] Dashboard KPIs reflect the latest run (no placeholders).
- [ ] One-command run for dashboard + backtest.
- [ ] UI follows Flexoki/Kepano minimal style guide.
