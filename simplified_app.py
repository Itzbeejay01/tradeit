import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import custom modules
import data_service as ds
import trading_signals as ts
import technical_indicators as ti
import visualization as viz

# Page configuration
st.set_page_config(
    page_title="Quick Forex & Crypto Trading Signals",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'assets' not in st.session_state:
    st.session_state.assets = ['AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD', 'BTC/ADA', 'BTC/DOGE', 'BTC/ETH', 'BTC/FDUSD', 'BTC/LTC', 'BTC/USDT', 'BTC/USD', 'BTC/XLM', 'ETH/ADA', 'ETH/BCH', 'ETH/BTC', 'ETH/DOGE', 'ETH/FDUSD', 'ETH/LINK', 'ETH/USDT', 'ETH/USD', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY', 'EUR/NZD', 'EUR/USD', 'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/JPY', 'GBP/NZD', 'GBP/USD', 'NZD/CAD', 'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'USD/CAD', 'USD/CHF', 'USD/JPY', 'USD/MXN', 'USD/SEK', 'USD/SGD']
    
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(minutes=5)

# Main content
st.title("Quick Trading Signals")

# Top section for inputs
col1, col2, col3 = st.columns(3)

with col1:
    # Asset selection
    selected_asset = st.selectbox(
        "Select Asset",
        st.session_state.assets,
        key="signal_asset"
    )
    
    # Add custom asset option
    new_asset = st.text_input("Or add a new asset (e.g., BTC-USD, EUR/USD)")
    if st.button("Add Asset"):
        if new_asset and new_asset not in st.session_state.assets:
            try:
                # Validate asset by trying to fetch data
                ds.get_asset_data(new_asset, '1d', '1d')
                st.session_state.assets.append(new_asset)
                st.success(f"{new_asset} added!")
                # Auto-select the new asset
                st.session_state.signal_asset = new_asset
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    # Timeframe selection
    timeframe = st.radio(
        "Select Timeframe",
        ["1m", "2m", "5m", "15m", "30m", "60m"],
        horizontal=True
    )
    
    # Period selection
    period = st.radio(
        "Data Period",
        ["1d", "5d", "1mo"],
        horizontal=True,
        help="How far back to analyze data"
    )

with col3:
    # Current price display
    try:
        current_data = ds.get_current_price(selected_asset)
        st.metric(
            label=f"Current Price ({selected_asset})", 
            value=f"{current_data['price']:.4f}",
            delta=f"{current_data['change_percent']:.2f}%"
        )
    except Exception as e:
        st.error(f"Error fetching current price: {str(e)}")
    
    # Generate signal button
    signal_button = st.button("🔄 GET TRADING SIGNAL", type="primary", use_container_width=True)
    
    # Last updated timestamp
    st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Manual refresh button
    if st.button("Refresh Data", use_container_width=True):
        st.session_state.last_update = datetime.now()
        st.rerun()

# Display signal if button is clicked
if signal_button:
    try:
        with st.spinner(f"Analyzing {selected_asset} with {timeframe} timeframe..."):
            # Fetch data
            df = ds.get_asset_data(selected_asset, timeframe, period)
            
            if len(df) < 20:
                st.warning(f"Limited data available for {selected_asset} with {timeframe} timeframe. Results may be less reliable.")
            
            # Generate trading advice
            advice = ts.generate_trading_advice(selected_asset, df)
            
            # Display recommendation
            rec = advice["recommendation"]
            strength = advice["strength"]
            
            # Layout for signal display
            signal_col1, signal_col2 = st.columns([2, 1])
            
            # Determine color based on recommendation
            if rec == "BUY":
                color = "green"
                emoji = "🟢"
            elif rec == "SELL":
                color = "red"
                emoji = "🔴"
            else:
                color = "orange"
                emoji = "🟠"
            
            with signal_col1:
                # Display recommendation card
                st.markdown(f"""
                <div style="padding: 25px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};">
                    <h1 style="text-align: center; color: {color}; font-size: 48px;">{emoji} {rec} {emoji}</h1>
                    <h3 style="text-align: center;">Signal Strength: {strength:.1f}%</h3>
                    <p style="text-align: center;">Based on analysis of {timeframe} data over {period}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show reasoning
                st.subheader("Analysis Reasoning")
                for reason in advice["reasoning"]:
                    st.markdown(f"- {reason}")
            
            with signal_col2:
                # Price chart
                st.subheader("Price Chart")
                fig = viz.plot_candlestick(df, f"{selected_asset} ({timeframe})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Current indicators
                st.subheader("Current Indicator Values")
                
                # Prepare indicators for display
                analysis_df = df.copy()
                analysis_df = ti.add_sma(analysis_df, 20, 50)
                analysis_df = ti.add_rsi(analysis_df, 14)
                analysis_df = ti.add_macd(analysis_df)
                
                latest = analysis_df.iloc[-1]
                
                # Create metrics for important indicators
                st.metric("RSI (14)", f"{latest['RSI_14']:.2f}", 
                          delta="Overbought" if latest['RSI_14'] > 70 else ("Oversold" if latest['RSI_14'] < 30 else "Neutral"))
                
                if "MACD" in latest:
                    macd_signal = latest["MACD"] - latest["MACD_Signal"]
                    st.metric("MACD", f"{latest['MACD']:.4f}", 
                              delta=f"{macd_signal:.4f}" if macd_signal else "0")
            
            # Technical analysis charts
            st.subheader("Technical Analysis")
            
            # Create tabs for different indicators
            indicator_tabs = st.tabs(["SMA", "RSI", "MACD", "Bollinger Bands"])
            
            with indicator_tabs[0]:
                analysis_df = ti.add_sma(df, 20, 50)
                fig_sma = viz.plot_with_sma(analysis_df, f"{selected_asset} with SMA", 20, 50)
                st.plotly_chart(fig_sma, use_container_width=True)
                
            with indicator_tabs[1]:
                analysis_df = ti.add_rsi(df, 14)
                fig_rsi = viz.plot_rsi(analysis_df, 14)
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            with indicator_tabs[2]:
                analysis_df = ti.add_macd(df)
                fig_macd = viz.plot_macd(analysis_df)
                st.plotly_chart(fig_macd, use_container_width=True)
                
            with indicator_tabs[3]:
                analysis_df = ti.add_bollinger_bands(df, 20)
                fig_bb = viz.plot_with_bollinger_bands(analysis_df, f"{selected_asset} with Bollinger Bands", 20)
                st.plotly_chart(fig_bb, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating trading signal: {str(e)}")
        st.error("If timeframe is too short, try using a longer period to get more data.")
else:
    # Default display when page loads
    st.info("👆 Select your asset and timeframe, then click 'GET TRADING SIGNAL' to see if you should buy or sell.")
    
    # Show a sample list of available assets
    st.subheader("Available Assets")
    st.write("You can analyze any of these assets or add your own:")
    
    # Display available assets in a grid
    asset_cols = st.columns(3)
    for i, asset in enumerate(st.session_state.assets):
        asset_cols[i % 3].button(
            asset, 
            key=f"asset_button_{i}",
            on_click=lambda asset=asset: st.session_state.update({"signal_asset": asset}),
            use_container_width=True
        )

# Auto-refresh data every 2 minutes
if (datetime.now() - st.session_state.last_update).total_seconds() > 120:
    st.session_state.last_update = datetime.now()
    st.rerun()