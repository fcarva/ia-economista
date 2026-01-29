
import streamlit as st
import pandas as pd
import plotly.io as pio
from dashboard.utils import load_css, make_flexoki_chart
from dashboard.components.order_flow import plot_order_flow, create_trades_df_from_positions
from dashboard.components.risk_cone import plot_risk_cone, generate_forecast_cone


import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import glob
import torch

from cointegration_gnn.config import default_config
from cointegration_gnn.models.gnn_model import CointegrationGNN
from cointegration_gnn.backtest.backtest_engine import BacktestEngine
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph

st.set_page_config(page_title="Strategy Backtest", page_icon="🧪", layout="wide")
load_css()

st.title("🧪 Strategy Backtest Lab")

# Sidebar
st.sidebar.header("Backtest Configuration")

# Model Selection
model_dir = Path(__file__).parent.parent.parent / "models"
model_files = sorted(glob.glob(str(model_dir / "*.ckpt")), reverse=True)
model_names = [Path(f).name for f in model_files]

selected_model_name = st.sidebar.selectbox(
    "Select Trained Model",
    options=model_names if model_names else ["No models found"],
    index=0 if model_names else 0
)

st.sidebar.divider()

# Backtest Parameters
initial_capital = st.sidebar.number_input("Initial Capital (USD)", 10_000, 10_000_000, 1_000_000)
cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 50, 10)
slippage_bps = st.sidebar.slider("Slippage (bps)", 0, 50, 5)

start_date = st.sidebar.date_input("Start Date", default_config.data.test_start)
end_date = st.sidebar.date_input("End Date", default_config.data.test_end)

@st.cache_data
def load_test_data(start, end):
    loader = DataLoader(tickers=default_config.data.tickers)
    # Fetch extra history for window
    fetch_start = start - pd.Timedelta(days=default_config.cointegration.rolling_window * 2)
    prices, returns = loader.fetch_data(start=fetch_start, end=end)
    features = loader.compute_features(prices, returns)
    return prices, returns, features

def run_backtest_logic(model_path, prices, returns, features, start_date, end_date):
    # 1. Load Model - detect encoder type from checkpoint
    import torch
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check if this is a legacy GraphSAGE checkpoint by inspecting state dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    is_sage = 'encoder.convs.0.lin_l.weight' in state_dict and state_dict['encoder.convs.0.lin_l.weight'].shape[0] == 32
    
    if is_sage:
        # Legacy GraphSAGE checkpoint
        model = CointegrationGNN.load_from_checkpoint(model_path, encoder_type="sage")
    else:
        # GATv2 checkpoint
        model = CointegrationGNN.load_from_checkpoint(model_path)
    
    model.eval()
    
    # 2. Filter Data for Test Period
    test_mask = (features.index >= pd.Timestamp(start_date)) & (features.index <= pd.Timestamp(end_date))
    test_features = features[test_mask]
    test_returns = returns.loc[test_features.index]
    test_prices = prices.loc[test_features.index]
    
    if test_features.empty:
        return None, None, "No data in selected range."
    
    # 3. Generate Signals
    # Needs Cointegration Graph for each day
    coint_graph = CointegrationGraph(
        tickers=default_config.data.tickers, 
        rolling_window=default_config.cointegration.rolling_window
    )
    
    adj_cache = coint_graph.compute_rolling_adjacencies(prices, recompute_freq=5)
    
    signals = []
    dates = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    total_steps = len(test_features)
    
    for i, (date, row_features) in enumerate(test_features.iterrows()):
        # Update progress
        if i % 10 == 0 and total_steps > 0:
            progress_bar.progress(i / total_steps)
            
        # Get Adjacency (Look-ahead safe)
        closest_date = max([d for d in adj_cache.keys() if d <= date], default=None)
        if closest_date is None:
            # Fallback to zeros or skip
            adjacency = np.zeros((len(default_config.data.tickers), len(default_config.data.tickers)))
        else:
            adjacency = adj_cache[closest_date]
            
        # Prepare Graph Data
        x_rows = []
        for ticker in default_config.data.tickers:
             x_rows.append(row_features[ticker].values)
        x = np.stack(x_rows) # (n_assets, n_features)
        
        data = coint_graph.to_pyg_data(adjacency, x)
        data.tickers = default_config.data.tickers # Attach tickers
        
        # Predict
        position_dict = model.predict_positions(data)
        signals.append(position_dict)
        dates.append(date)
        
    progress_bar.progress(1.0)
    
    # Convert list of dicts to DataFrame
    signal_df = pd.DataFrame(signals, index=dates)
    
    # 4. Run Backtest Engine
    engine = BacktestEngine(
        tickers=default_config.data.tickers,
        initial_capital=initial_capital,
        transaction_cost_bps=cost_bps,
        slippage_bps=slippage_bps
    )
    
    # Align signals with prices index (fill missing with 0 or ffill)
    aligned_signals = signal_df.reindex(test_prices.index).fillna(0)
    
    # IMPORTANT: signal_df[t] is signal generated at t.
    # In 'predict_positions', we use 'data_loader' features which are shifted.
    # So features[t] uses info from t-1.
    # So signal generated at t uses info from t-1.
    # We trade at t Close.
    # This works with engine.
    
    def signal_func(historical_data, current_date):
        # Engine calls this for each date
        # We look up pre-computed signal
        ts = pd.Timestamp(current_date)
        if ts in aligned_signals.index:
            return aligned_signals.loc[ts].to_dict()
        return {t: 0.0 for t in default_config.data.tickers}

    result = engine.run(test_prices, signal_generator=signal_func)
    
    return result, signal_df, None

if st.sidebar.button("🚀 Run Backtest", type="primary"):
    if not model_files:
        st.error("No trained models found. Please train a model first via 'scripts/train.py'.")
    else:
        model_path = str(model_dir / selected_model_name)

        # --- ASCII LOADING ART ---
        loading_placeholder = st.empty()

        def ascii_loader(step, msg):
            art = ["▖", "▘", "▝", "▗"]
            loading_placeholder.markdown(
                f"""
            ```text
            {art[step % 4]} SYSTEM PROCESS: {msg}
            [{'=' * (step+1) + ' ' * (10-step)}] {step*10}%
            ```
            """
            )

        ascii_loader(1, "INITIALIZING ENVIRONMENT")

        ascii_loader(3, "CONNECTING TO DATA FEED")
        try:
            prices, returns, features = load_test_data(start_date, end_date)
            if features.empty:
                raise ValueError("No features available for the selected range.")
            ascii_loader(6, "COMPUTING TENSORS")

            result, signal_df, error = run_backtest_logic(
                model_path, prices, returns, features, start_date, end_date
            )

            ascii_loader(9, "FINALIZING METRICS")
            loading_placeholder.empty()

            if error:
                st.error(f"Backtest Failed: {error}")
                if "No data" in error:
                    st.warning(
                        "Try adjusting the 'Start Date' to earlier (e.g., 2024-01-01) "
                        "or check if yfinance has 2025 data available."
                    )
            else:
                st.success("Simulation Complete!")

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{result.total_return:.2%}")
                col2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                col3.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                col4.metric("Turnover", f"{result.turnover:.2%}")

                # Equity Curve
                st.subheader("Equity Curve")
                fig = px.line(result.equity_curve, title="Portfolio Value Over Time")
                fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray")
                fig = make_flexoki_chart(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Positions Area Chart
                st.subheader("Position Exposure")
                pos_df = result.positions
                fig_pos = px.area(pos_df, title="Net Exposure per Asset")
                fig_pos = make_flexoki_chart(fig_pos)
                st.plotly_chart(fig_pos, use_container_width=True)

                # ========== TRACK 5: Bloomberg Terminal UX ==========
                st.divider()
                st.subheader("🕵️ Trade Analysis (Order Flow)")

                selected_ticker = st.selectbox(
                    "Inspect Asset:",
                    options=default_config.data.tickers,
                    index=0,
                )

                trades_df = create_trades_df_from_positions(
                    positions_df=result.positions,
                    prices_df=prices.loc[result.positions.index],
                    signals_df=signal_df.reindex(result.positions.index)
                    if signal_df is not None
                    else None,
                )

                fig_flow = plot_order_flow(
                    prices_df=prices.loc[result.positions.index],
                    trades_df=trades_df,
                    ticker=selected_ticker,
                )
                fig_flow = make_flexoki_chart(fig_flow)
                st.plotly_chart(fig_flow, use_container_width=True)

                if not trades_df.empty:
                    ticker_trades = trades_df[trades_df["Ticker"] == selected_ticker]
                    if not ticker_trades.empty:
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Trades", len(ticker_trades))
                        col_b.metric("Avg Confidence", f"{ticker_trades['Confidence'].mean():.2f}")
                        col_c.metric("Avg Position Size", f"{ticker_trades['Size'].mean():.1%}")

                st.divider()
                st.subheader("📈 Forecast Cone (Model Uncertainty)")

                historical_prices = prices.loc[result.positions.index]
                forecast_cone = generate_forecast_cone(
                    model=None,
                    prices_df=historical_prices,
                    ticker=selected_ticker,
                    horizon=5,
                    n_simulations=100,
                )

                recent_history = historical_prices.tail(30)
                fig_cone = plot_risk_cone(recent_history, forecast_cone, selected_ticker)
                fig_cone = make_flexoki_chart(fig_cone)
                st.plotly_chart(fig_cone, use_container_width=True)

                st.caption(
                    "💡 **Interpretation:** If price exits the shaded cone, it indicates a "
                    "regime shift or unexpected shock."
                )

        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Critical Runtime Error: {e}")
            import traceback

            st.code(traceback.format_exc())

else:
    if not model_files:
        st.warning("⚠️ No trained models found in `models/` directory.")
        st.info("Run `python scripts/train.py` in your terminal to train a model.")
    else:
        st.info("Ready. Select a model and click 'Run Backtest'.")
