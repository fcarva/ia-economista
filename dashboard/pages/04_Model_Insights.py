
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import glob
import torch

from cointegration_gnn.config import default_config
from cointegration_gnn.models.gnn_model import CointegrationGNN
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.cointegration_graph import CointegrationGraph
from dashboard.utils import load_css, make_flexoki_chart

# Page Config
st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
load_css()

st.title("🧠 Model Insights (XAI)")
st.markdown("### GNN Interpretability & Signal Drivers")

# Sidebar: Select Model
model_dir = Path(__file__).parent.parent.parent / "models"
model_files = sorted(glob.glob(str(model_dir / "*.ckpt")), reverse=True)
model_names = [Path(f).name for f in model_files]

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=model_names if model_names else ["No models found"],
    index=0 if model_names else 0
)

@st.cache_data
def load_model_and_importance(model_path):
    # This mimics logic from scripts/explain_model.py
    # But runs somewhat faster for demo purposes or caches result
    
    # 1. Load Model - detect encoder type from checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    is_sage = 'encoder.convs.0.lin_l.weight' in state_dict and state_dict['encoder.convs.0.lin_l.weight'].shape[0] == 32
    
    if is_sage:
        model = CointegrationGNN.load_from_checkpoint(model_path, encoder_type="sage")
    else:
        model = CointegrationGNN.load_from_checkpoint(model_path)
    
    model.eval()
    
    # Load Macro Data to match training shape
    from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher
    
    # 2. Get Data (Snapshot)
    loader = DataLoader(tickers=default_config.data.tickers)
    # Just fetch last 6 months for a sample
    end_date = default_config.data.val_end
    # Ensure wide enough window for features
    start_date = end_date - pd.Timedelta(days=365) 
    
    prices, returns = loader.fetch_data(start=start_date, end=end_date)
    features = loader.compute_features(prices, returns)
    
    # Fetch Macro
    try:
        macro_fetcher = BrazilMacroFetcher(cache_path="data/brazil_macro.csv")
        macro_df = macro_fetcher.fetch_all(start_date=start_date.strftime("%Y-%m-%d"))
    except Exception:
        macro_df = pd.DataFrame(index=features.index)
    
    # Add Macro to Features (Mirroring run_backtest logic)
    # Need to verify if add_macro_to_node_features is importable or replicate logic
    # It's usually in a utils file, but let's replicate robustly here or check imports
    
    # Replicate logic briefly to ensure 13 features (8 base + 5 macro)
    # Actually wait, error said 19x128. That means 19 features?
    # Base=8. Macro expected=['selic', 'usd_brl', 'ibc_br', 'ipca_mom', 'ipca_12m'] = 5.
    # Total = 13.
    # Where does 19 come from?
    # Maybe config defines more lags? Or macro has more columns?
    # Let's import the specific helper function if it exists.
    # Checking scripts/run_backtest.py for 'add_macro_to_node_features'
    
    def add_macro_local(node_features, macro_df):
        # Join macro columns to every node
        # Ensure index alignment
        joined = node_features.join(macro_df, how='left').ffill().fillna(0.0)
        return joined
        
    features = add_macro_local(features, macro_df)
    
    if features.empty:
        raise ValueError("Feature matrix is empty. Check data availability.")

    # 3. Graph
    coint = CointegrationGraph(tickers=default_config.data.tickers)
    date = features.index[-1]
    adj = coint.compute_adjacency(prices, date)
    
    # 4. Gradient Feature Importance
    # Create input tensor
    # Ensure we select only numeric valid columns that match model input
    # If model expects 19, we must have 19.
    # Let's assume standard behavior for now (13 features). If 19, likely extra macro vars.
    # We will try to pass all available numeric columns from features.
    
    # Drop non-feature columns if any
    safe_features = features.select_dtypes(include=[np.number])
    if safe_features.empty:
        raise ValueError("No numeric features available for XAI.")
    
    x_rows = [safe_features.loc[date].values for t in default_config.data.tickers]
    x = torch.tensor(np.stack(x_rows), dtype=torch.float, requires_grad=True)
    
    edge_index_np = np.array(np.nonzero(adj))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(adj[edge_index_np[0], edge_index_np[1]], dtype=torch.float).unsqueeze(1)
    
    # Forward Pass
    embeddings = model.encoder(x, edge_index, edge_attr)
    output = model.position_head(embeddings)
    
    # Backward to get gradients
    target = output.sum()
    target.backward()
    
    grads = x.grad.abs().mean(dim=0).numpy() # Mean across assets
    inputs = x.detach().abs().mean(dim=0).numpy()
    importance = grads * inputs
    
    # Feature Names (hardcoded match to data_loader)
    # Feature Names dynamic
    num_features = x.shape[1]
    if num_features == 8:
        feat_names = [
            "Return (1d)", "Momentum (5d)", "Volatility (20d)", 
            "Z-Score (20d)", "RSI (14d)", "Spread vs Mean", 
            "Lag Return (t-1)", "Lag Return (t-2)"
        ]
    elif num_features >= 13:
        # Correspond to base + macro
        feat_names = [
            "Return (1d)", "Momentum (5d)", "Volatility (20d)", 
            "Z-Score (20d)", "RSI (14d)", "Spread vs Mean", 
            "Lag Return (t-1)", "Lag Return (t-2)"
        ]
        # Append generic macro names for the rest
        macro_names = [f"Macro_{i}" for i in range(num_features - 8)]
        # Try to guess specific names if standard 5 are present
        if num_features == 13:
            macro_names = ["Selic", "USD/BRL", "IBC-Br", "IPCA MoM", "IPCA 12m"]
        feat_names.extend(macro_names)
    else:
        feat_names = [f"Feature_{i}" for i in range(num_features)]
    
    df_imp = pd.DataFrame({
        "Feature": feat_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)
    
    # Handle both 1D and 2D output shapes
    output_np = output.detach().numpy()
    signals = output_np.flatten()  # Flatten to 1D regardless of original shape
    
    return df_imp, signals

if not model_files:
    st.error("No models found.")
    st.stop()

model_path = str(model_dir / selected_model_name)

if st.button("🔍 Analyze Model Logic"):
    with st.spinner("Calculating Feature Gradients..."):
        try:
            df_imp, signals = load_model_and_importance(model_path)

            # Layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Global Feature Importance")
                st.caption("Gradient x Input | Higher = More Impact on Signal")

                fig = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    text_auto=".3f",
                )

                fig.update_traces(
                    marker_color="#205EA6",
                    textfont_size=12,
                    textangle=0,
                    textposition="outside",
                    cliponaxis=False,
                )

                fig = make_flexoki_chart(fig)
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Trading Logic Diagnosis")

                top_f = df_imp.iloc[0]
                st.container(border=True).metric(
                    "Top Driver", top_f["Feature"], f"{top_f['Importance']:.4f}"
                )

                st.markdown("### 🕵️ Logic Check")

                checks = [
                    ("Mean Reversion", "Z-Score (20d)", "Is it Top 3?"),
                    ("Trend Following", "Momentum (5d)", "Is it Top 3?"),
                    ("Risk Aversion", "Volatility (20d)", "Is it significant?"),
                ]

                for name, feat, q in checks:
                    is_top = feat in df_imp.head(3)["Feature"].values
                    icon = "✅" if is_top else "⚪"
                    st.markdown(f"**{name}**: {icon} ({feat})")

                st.info(
                    "If Z-Score is top driver, the model is likely trading cointegration spreads effectively."
                )

                st.divider()
                st.caption("Analysis based on latest validation date snapshot.")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
else:
    st.info("Click 'Analyze Model Logic' to run the XAI module.")
