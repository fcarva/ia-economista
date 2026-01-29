# -*- coding: utf-8 -*-
"""
Order Flow Visualization Component
==================================

Bloomberg-style price chart with trade execution overlay.
Shows agent confidence (opacity) and position size (marker size).

Track 5: Bloomberg Terminal UX
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_order_flow(prices_df, trades_df, ticker, theme_color="#205EA6"):
    """
    Plota o gráfico de preços com overlay de execuções (Buy/Sell).
    
    Args:
        prices_df: DataFrame with datetime index and price columns per ticker.
        trades_df: DataFrame with columns ['Date', 'Ticker', 'Action', 'Size', 'Confidence', 'Price'].
                   Action should be 'BUY' or 'SELL'.
        ticker: Asset symbol (e.g., 'PETR4.SA').
        theme_color: Primary color for buy markers.
        
    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # 1. Price Line (Base)
    if ticker in prices_df.columns:
        fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=prices_df[ticker],
            mode='lines',
            name=f'{ticker} Price',
            line=dict(color='#100F0F', width=1.5),  # Flexoki Black
            opacity=0.6
        ))

    if trades_df is not None and not trades_df.empty:
        # Filter trades for this asset
        asset_trades = trades_df[trades_df['Ticker'] == ticker].copy()
        
        if not asset_trades.empty:
            # 2. Buy Markers (Triangle up)
            buys = asset_trades[asset_trades['Action'] == 'BUY']
            if not buys.empty:
                # Scale size: abs(Size) * multiplier, clamp to reasonable range
                marker_sizes = np.clip(buys['Size'].abs() * 100, 5, 30)
                
                fig.add_trace(go.Scatter(
                    x=buys['Date'],
                    y=buys['Price'],
                    mode='markers',
                    name='Buy Order',
                    marker=dict(
                        symbol='triangle-up',
                        size=marker_sizes,
                        color=theme_color,
                        opacity=buys['Confidence'].clip(0.3, 1.0),  # Min opacity 0.3
                        line=dict(width=1, color='#100F0F')
                    ),
                    hovertemplate=(
                        "<b>BUY %{x}</b><br>"
                        "Price: %{y:.2f}<br>"
                        "Size: %{customdata[0]:.1%}<br>"
                        "Confidence: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.stack([buys['Size'], buys['Confidence']], axis=1)
                ))

            # 3. Sell Markers (Triangle down)
            sells = asset_trades[asset_trades['Action'] == 'SELL']
            if not sells.empty:
                marker_sizes = np.clip(sells['Size'].abs() * 100, 5, 30)
                
                fig.add_trace(go.Scatter(
                    x=sells['Date'],
                    y=sells['Price'],
                    mode='markers',
                    name='Sell Order',
                    marker=dict(
                        symbol='triangle-down',
                        size=marker_sizes,
                        color='#D14D41',  # Flexoki Red
                        opacity=sells['Confidence'].clip(0.3, 1.0),
                        line=dict(width=1, color='#100F0F')
                    ),
                    hovertemplate=(
                        "<b>SELL %{x}</b><br>"
                        "Price: %{y:.2f}<br>"
                        "Size: %{customdata[0]:.1%}<br>"
                        "Confidence: %{customdata[1]:.2f}<extra></extra>"
                    ),
                    customdata=np.stack([sells['Size'], sells['Confidence']], axis=1)
                ))

    # Bloomberg Minimalist Layout
    fig.update_layout(
        title=dict(
            text=f"Order Flow Analysis: {ticker}",
            font=dict(family="Fraunces", size=18)
        ),
        template="plotly_white",
        paper_bgcolor="#FFFCF0",  # Flexoki Paper
        plot_bgcolor="#FFFCF0",
        font=dict(family="JetBrains Mono", color="#100F0F"),
        xaxis=dict(gridcolor="#E6E4D9", showgrid=True),
        yaxis=dict(gridcolor="#E6E4D9", showgrid=True, title="Price"),
        legend=dict(orientation="h", y=1.02, x=0.8),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )

    return fig


def create_trades_df_from_positions(positions_df, prices_df, signals_df=None):
    """
    Convert position changes to a trades DataFrame for visualization.
    
    Args:
        positions_df: DataFrame with positions (index: Date, columns: Tickers).
        prices_df: DataFrame with prices (index: Date, columns: Tickers).
        signals_df: Optional DataFrame with raw signals (for confidence).
        
    Returns:
        DataFrame with columns ['Date', 'Ticker', 'Action', 'Size', 'Confidence', 'Price'].
    """
    trades = []
    
    # Calculate position changes
    position_changes = positions_df.diff()
    
    for date in position_changes.index[1:]:  # Skip first row (NaN)
        for ticker in position_changes.columns:
            change = position_changes.loc[date, ticker]
            
            if abs(change) > 0.001:  # Significant position change
                action = 'BUY' if change > 0 else 'SELL'
                size = abs(change)
                
                # Get price at trade date
                price = prices_df.loc[date, ticker] if date in prices_df.index else None
                
                # Get confidence from signals (or use position magnitude as proxy)
                if signals_df is not None and date in signals_df.index:
                    confidence = abs(signals_df.loc[date, ticker])
                else:
                    confidence = min(abs(positions_df.loc[date, ticker]), 1.0)
                
                if price is not None:
                    trades.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Action': action,
                        'Size': size,
                        'Confidence': confidence,
                        'Price': price
                    })
    
    return pd.DataFrame(trades)
