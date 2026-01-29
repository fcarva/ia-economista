import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.config import default_config
from dashboard.utils import load_css, make_flexoki_chart

# Page Config
st.set_page_config(page_title="Data Universe", page_icon="📊", layout="wide")
load_css()

# --- 1. HELPER FUNCTIONS ---
def run_adf_test(series):
    """Roda ADF e retorna (p-value, is_stationary)."""
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value, p_value < 0.05

@st.cache_data
def get_stationarity_diagnosis(prices_df):
    """
    Diagnóstico completo de ordem de integração I(d).
    Para Cointegração, queremos I(1): Preço Non-Stat, Retorno Stat.
    """
    results = []
    for ticker in prices_df.columns:
        price = prices_df[ticker]
        ret = price.pct_change().dropna()
        
        # Teste em Nível (Preço)
        p_price, price_is_stat = run_adf_test(price)
        
        # Teste em Diferença (Retorno)
        p_ret, ret_is_stat = run_adf_test(ret)
        
        # Classificação Econômica & CALL TO ACTION
        if not price_is_stat and ret_is_stat:
            order = "I(1)"
            status = "✅ DEPLOY ALPHA" # CTA Agressivo: "Execute a Estratégia"
            color = "#24837B" # Green (Flexoki)
        elif price_is_stat:
            order = "I(0)"
            status = "⚠️ AVOID (Mean Rev)" # CTA: "Evite"
            color = "#DA702C" # Orange
        else:
            order = "I(2+)"
            status = "⛔ HALT (Explosive)" # CTA: "Pare"
            color = "#D14D41" # Red
            
        results.append({
            "Ticker": ticker,
            "Price p-val": p_price,
            "Return p-val": p_ret,
            "Order": order,
            "Action": status, # Renomeado de Status para Action
            "_color": color # Coluna oculta para styling
        })
    
    return pd.DataFrame(results).set_index("Ticker")

# --- 2. MAIN UI ---
st.title("📊 Data Universe & Stationarity")
st.markdown("### Asset Qualification Engine")

# Sidebar
with st.sidebar:
    st.header("⚙️ Universe Config")
    selected_tickers = st.multiselect(
        "Select Assets", 
        default_config.data.tickers,
        default=default_config.data.tickers
    )
    
    # Date Range
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if not selected_tickers:
    st.warning("Please select at least one asset to analyze.")
    st.stop()

# Load Data
with st.spinner("Fetching market telemetry..."):
    loader = DataLoader(tickers=selected_tickers)
    prices, returns = loader.fetch_data(start=start_date, end=end_date)
    
    # Filter selection
    prices = prices[selected_tickers]
    returns = returns[selected_tickers]

# --- 3. CHARTS SECTION ---
col_chart, col_stat = st.columns([2, 1])

with col_chart:
    st.subheader("Price Action (Normalized)")
    # Normalize to 100 for comparable visuals
    normalized = (prices / prices.iloc[0]) * 100
    
    fig = px.line(normalized, x=normalized.index, y=normalized.columns)
    fig.update_layout(yaxis_title="Rel. Performance (Base 100)")
    fig = make_flexoki_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

# --- 4. STATIONARITY DIAGNOSIS (CTA MODE) ---
with col_stat:
    st.subheader("🧬 Signal Qualification")
    st.markdown("ADF Root Test Results:")
    
    # Run Diagnosis
    diag_df = get_stationarity_diagnosis(prices)
    
    # Apply Pandas Styler for the "Badge" look
    def highlight_status(row):
        # Lookup color from original dataframe using index
        color = diag_df.loc[row.name, '_color']
        # Aplica a cor na coluna 'Action' com peso bold
        return [f'color: {color}; font-weight: 800; letter-spacing: 0.05em;' if col == 'Action' else '' for col in row.index]

    # Display clean table
    st.dataframe(
        diag_df[["Price p-val", "Return p-val", "Order", "Action"]].style.apply(highlight_status, axis=1)
        .format({"Price p-val": "{:.3f}", "Return p-val": "{:.1e}"}),
        use_container_width=True,
        height=400
    )
    
    st.caption("**Protocolo:** Ativos **I(1)** são elegíveis para formação de pares (Cointegração). Ativos **I(0)** já reverteram à média e não possuem prêmio de risco.")

# --- 5. RISK METRICS ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.subheader("📉 Volatility Regime (20d)")
    vol = returns.rolling(20).std() * (252**0.5)
    fig_vol = px.line(vol, title="Annualized Volatility")
    fig_vol.update_layout(showlegend=False)
    fig_vol = make_flexoki_chart(fig_vol)
    st.plotly_chart(fig_vol, use_container_width=True)

with c2:
    st.subheader("🔗 Correlation Structure")
    corr = returns.corr()
    fig_corr = px.imshow(
        corr, 
        text_auto=".2f", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_corr = make_flexoki_chart(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)
