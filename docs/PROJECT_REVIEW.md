# Project Review — AI Economist (IBOVESPA)

## Executive Summary
- The core architecture is coherent (data → features → model → backtest → validation), but benchmark design and UI telemetry are currently weak, which risks overstating performance and undermining user trust. The benchmark in `scripts/run_backtest.py` is an equal-weight proxy and not a true IBOVESPA index proxy, and dashboard KPIs are mostly static placeholders. This makes it hard to interpret the model’s real-world edge. In addition, the dashboard theme is close to the “Flexoki/Kepano minimal” goal, but several CSS tokens are undefined, limiting consistency and polish.

## Benchmarking Assessment (Why the benchmark feels uninteresting)
**Current state**
- The backtest compares the strategy to a simple equal-weight buy-and-hold benchmark derived from the same universe (`returns.mean(axis=1)`), which is not an IBOV index proxy and may produce misleading alpha in trending or concentrated regimes. This is implemented in `scripts/run_backtest.py`. 

**Risks**
- Equal-weight benchmarks overweight small/mid caps compared to IBOV’s value-weighted index; results may not be comparable to the market you intend to beat. 
- If the universe is the same as the strategy’s tradable basket, the benchmark is mechanically correlated, which can compress meaningful relative performance differences.

**Recommendations**
1. **Add a true IBOV proxy**: use IBOV index data (e.g., `^BVSP` from Yahoo) or BOVA11 ETF as the primary benchmark. Keep equal-weight as a *secondary* diagnostic. 
2. **Report multi-benchmark diagnostics**: compare vs. IBOV, vs. equal-weight, and vs. sector-neutral (or risk-parity) to understand whether gains are alpha or factor exposure.
3. **Include tracking error + information ratio**: explicitly show how your strategy diverges from IBOV and whether the divergence is rewarded.
4. **Benchmark assumptions match trading costs**: align transaction cost/slippage assumptions for benchmark if it’s a tradable proxy (e.g., BOVA11). 

## Model & Data Pipeline Review
**Strengths**
- The pipeline components are clearly separated (data loader, cointegration graph, causal discovery, GNN model, backtest). This is maintainable and supports iterative research.
- Validation via stationary bootstrap is a strong step toward avoiding single-path overfitting.

**Gaps**
- The bootstrap validation includes only the strategy series; there is a placeholder for benchmarking but it’s not used. This is an opportunity to provide probabilistic statements like “Sharpe > benchmark with X% probability.”
- The demo and dashboard rely on snapshots and static metrics; you will need a data-refresh layer to support live model telemetry.

## Backtesting & Risk
**Strengths**
- The backtest engine enforces time ordering and includes execution delay and transaction costs. 

**Gaps & Next Steps**
- Add **portfolio constraints** (max position size, max leverage, sector caps) to avoid unrealistic exposures.
- Add **walk-forward evaluation** with rolling re-train windows to align with the regime-based splits.
- Add **out-of-sample stress tests** (volatile regimes, 2020 COVID shock) to confirm stability.

## UX / UI (Flexoki / Kepano minimal)
**Current state**
- The dashboard already follows a minimalist typography-first aesthetic, but several CSS variables used in the style sheet are undefined (e.g., `--accent-color`, `--border-color`, `--sidebar-bg`). This causes inconsistent colors and prevents a polished, cohesive UI. 
- The KPI row on the home page is static, which breaks the “Bloomberg-lite” mental model of live, data-driven telemetry.

**Recommendations (visual + UX)**
1. **Define all CSS tokens** (Flexoki/Kepano minimal palette), including background, accent, and border variables. 
2. **Replace static KPIs with computed values** from the latest backtest summary JSON, or from a cached metrics store.
3. **Introduce a compact “market tape”** (top-of-page ticker strip) for quick state awareness.
4. **Use consistent data labeling** (e.g., add units, date ranges, and benchmark context to each metric).
5. **Add a “status & freshness” chip** for data and model recency (last data pull, last model retrain).

## Engineering & Dev Experience
**What’s good**
- Poetry + `pyproject.toml` is clean for dependency management.

**Improve for a more MVP-ready workflow**
- Provide a single-entry developer workflow (Makefile or task runner) to run dashboards, backtests, training, and linting consistently.
- Add a workspace checklist with “Definition of Done” criteria for each major module (data, model, backtest, UI).

---

## Suggested Benchmarks (Quick Reference)
- **Primary**: `^BVSP` (IBOV) or BOVA11 ETF
- **Secondary**: Equal-weight basket (current)
- **Tertiary**: Risk-parity or sector-neutral baseline

## Suggested Metrics (MVP)
- Tracking Error, Information Ratio
- Alpha vs IBOV + Beta to IBOV
- Drawdown stats (max, avg, duration)
- Turnover + costs impact
- Prob(Sharpe > 0) + Prob(Sharpe > Benchmark)
