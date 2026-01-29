
import streamlit as st
import pandas as pd
import glob
from pathlib import Path
import json
import plotly.express as px
from dashboard.utils import load_css, make_flexoki_chart

st.set_page_config(page_title="Model Metrics", page_icon="📈", layout="wide")
load_css()

st.title("📈 Model Metrics & Backtest")

# 1. Training Analysis
st.header("1. Training Convergence")
results_dir = Path("results")
analysis_plots = sorted(list(results_dir.glob("training_analysis_*.png")), reverse=True)

if analysis_plots:
    # Pick latest
    latest_plot = analysis_plots[0]
    st.image(str(latest_plot), caption=f"Training Metrics ({latest_plot.name})", use_container_width=True)
    
    with st.expander("History"):
        for p in analysis_plots[1:]:
            st.image(str(p), caption=p.name, width=400)
else:
    st.info("No training analysis plots found in results/.")

st.divider()

# 2. Backtest Results
st.header("2. Backtest Performance (Out-of-Sample)")
backtest_files = sorted(list(results_dir.glob("backtest_summary_*.json")), reverse=True)

if backtest_files:
    latest_backtest = backtest_files[0]
    with open(latest_backtest, 'r') as f:
        metrics = json.load(f)
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
    c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    c3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    c4.metric("Alpha vs Benchmark", f"{metrics.get('alpha', 0):.2%}")
    
    st.caption(f"Model: {metrics.get('model')} | Period: {metrics.get('period_start')} to {metrics.get('period_end')}")
    
    # Plot NAV
    # Find matching NAV pickle
    timestamp = latest_backtest.stem.split('_')[-1] # backtest_summary_2023... -> 2023...
    # split might be tricky if timestamp has underscores.
    # Actually timestamp is at the end: backtest_summary_YYYYMMDD_HHMMSS
    # So split('_')[-2:] joined?
    # Better: glob for *nav*{timestamp}.pkl
    nav_files = list(results_dir.glob(f"*nav*{timestamp}*.pkl"))
    
    if nav_files:
        nav_series = pd.read_pickle(nav_files[0])
        nav_df = nav_series.to_frame(name="Strategy")
        
        # Plot
        fig = px.line(nav_df, y="Strategy", title="Cumulative Return (NAV)")
        fig.update_layout(xaxis_title="Date", yaxis_title="NAV (Start=1.0)")
        fig = make_flexoki_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("NAV data not found for plotting.")

else:
    st.info("No backtest results found. Run scripts/run_backtest.py")

st.divider()
if st.button("🔄 Refresh Metrics"):
    st.rerun()
