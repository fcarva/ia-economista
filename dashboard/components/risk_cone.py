# -*- coding: utf-8 -*-
"""
Risk Cone Visualization Component
=================================

Visualizes model uncertainty with forecast confidence intervals.
Shows predicted price range (Mean ± 2σ) vs actual price.

Track 5: Bloomberg Terminal UX
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_risk_cone(historical_df, forecast_df, ticker):
    """
    Plota o preço histórico e o 'Cone de Incerteza' futuro.
    
    Args:
        historical_df: Historical data (Index: Datetime, Columns: prices per ticker).
        forecast_df: DataFrame with columns ['Date', 'Mean', 'Upper', 'Lower'].
                     Upper/Lower are typically Mean ± 2*StdDev.
        ticker: Asset symbol to plot.
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # 1. Historical Price
    if ticker in historical_df.columns:
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df[ticker],
            mode='lines',
            name='Historical',
            line=dict(color='#100F0F', width=2)
        ))

    if forecast_df is not None and not forecast_df.empty:
        # 2. Confidence Interval (Shaded Area)
        # Plotly trick: Draw Upper line (invisible) and fill to Lower
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',  # Fill to previous trace (Upper)
            fillcolor='rgba(32, 94, 166, 0.2)',  # Flexoki Blue transparent
            name='95% Confidence Interval'
        ))

        # 3. Forecast Mean (Dashed Line)
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Mean'],
            mode='lines',
            name='GNN Forecast',
            line=dict(color='#205EA6', width=2, dash='dash')
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"GNN Forecast Cone (5-Day Horizon): {ticker}",
            font=dict(family="Fraunces", size=18)
        ),
        paper_bgcolor="#FFFCF0",
        plot_bgcolor="#FFFCF0",
        font=dict(family="JetBrains Mono", color="#100F0F"),
        xaxis=dict(gridcolor="#E6E4D9", title="Date"),
        yaxis=dict(gridcolor="#E6E4D9", title="Price"),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15)
    )

    return fig


def generate_forecast_cone(model, prices_df, ticker, horizon=5, n_simulations=100):
    """
    Generate forecast cone data using Monte Carlo simulation.
    
    Args:
        model: Trained GNN model (CointegrationGNN).
        prices_df: Historical prices DataFrame.
        ticker: Asset to forecast.
        horizon: Forecast horizon in days.
        n_simulations: Number of Monte Carlo paths.
        
    Returns:
        DataFrame with ['Date', 'Mean', 'Upper', 'Lower'] columns.
    """
    # Placeholder implementation - in production, this would:
    # 1. Run model inference multiple times with dropout enabled (MC Dropout)
    # 2. Or use historical volatility to estimate uncertainty
    
    last_price = prices_df[ticker].iloc[-1]
    last_date = prices_df.index[-1]
    
    # Simple volatility-based forecast
    returns = np.log(prices_df[ticker]).diff().dropna()
    daily_vol = returns.std()
    
    forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='B')[1:]
    
    # Generate paths
    np.random.seed(42)
    paths = np.zeros((n_simulations, horizon))
    
    for i in range(n_simulations):
        cumulative_return = 0
        for t in range(horizon):
            daily_return = np.random.normal(0, daily_vol)
            cumulative_return += daily_return
            paths[i, t] = last_price * np.exp(cumulative_return)
    
    # Calculate statistics
    mean_path = paths.mean(axis=0)
    upper_path = np.percentile(paths, 97.5, axis=0)  # 95% CI
    lower_path = np.percentile(paths, 2.5, axis=0)
    
    return pd.DataFrame({
        'Date': forecast_dates,
        'Mean': mean_path,
        'Upper': upper_path,
        'Lower': lower_path
    })
