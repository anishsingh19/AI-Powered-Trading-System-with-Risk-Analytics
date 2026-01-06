# Essential imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import logging
from typing import List
from datetime import datetime, timedelta
import asyncio
import streamlit as st
import time
import threading

# Alpaca API imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Technical Analysis ---
class TechnicalAnalysis:
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty or len(df) < 26:  # Need at least 26 periods for EMA_26
                return df
                
            df = df.copy()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

# --- Enhanced Trading Strategy ---
class TradingStrategy:
    def __init__(self):
        self.position_size = 0.02  # 2% of portfolio per trade
        self.max_positions = 3
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.04  # 4% take profit
        
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate trading signal with confidence score"""
        if df.empty or len(df) < 50:
            return {'signal': 0, 'confidence': 0, 'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        reasons = []
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append(1)
            reasons.append('MACD bullish crossover')
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append(-1)
            reasons.append('MACD bearish crossover')
        
        # RSI Signal
        if latest['RSI'] < 30 and prev['RSI'] >= 30:
            signals.append(1)
            reasons.append('RSI oversold')
        elif latest['RSI'] > 70 and prev['RSI'] <= 70:
            signals.append(-1)
            reasons.append('RSI overbought')
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower'] and latest['Close'] > prev['Close']:
            signals.append(1)
            reasons.append('Bollinger bounce')
        elif latest['Close'] > latest['BB_Upper'] and latest['Close'] < prev['Close']:
            signals.append(-1)
            reasons.append('Bollinger resistance')
        
        # Volume confirmation
        volume_confirmed = latest['Volume_Ratio'] > 1.2
        
        if signals:
            signal = np.mean(signals)
            confidence = len(signals) / 3  # Max 3 signals
            if volume_confirmed:
                confidence *= 1.2
            
            return {
                'signal': 1 if signal > 0.3 else (-1 if signal < -0.3 else 0),
                'confidence': min(confidence, 1.0),
                'reason': ', '.join(reasons)
            }
        
        return {'signal': 0, 'confidence': 0, 'reason': 'No clear signal'}

# --- Risk Management ---
class RiskManager:
    def __init__(self, max_portfolio_risk=0.05):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = 0.02
        self.max_positions = 3
        
    def calculate_position_size(self, account_value: float, entry_price: float, stop_loss_price: float) -> int:
        """Calculate position size based on risk management"""
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0
        
        risk_amount = account_value * self.max_portfolio_risk
        position_size = int(risk_amount / risk_per_share)
        
        return max(1, position_size)
    
    def should_trade(self, current_positions: int, daily_pnl: float, account_value: float) -> bool:
        """Check if we should continue trading"""
        if current_positions >= self.max_positions:
            return False
        
        max_daily_loss_amount = account_value * self.max_daily_loss
        if daily_pnl < -max_daily_loss_amount:
            return False
        
        return True

# --- Enhanced Live Trading System ---
class LiveTradingSystem:
    def __init__(self, symbols: List[str], trading_client: TradingClient):
        self.symbols = symbols
        self.trading_client = trading_client
        self.historical_client = StockHistoricalDataClient(
            st.secrets["API_KEY"], 
            st.secrets["SECRET_KEY"]
        )
        self.technical_analysis = TechnicalAnalysis()
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager()
        self.stream = None
        self.is_running = False

    def initialize_data(self):
        """Initialize historical data for all symbols"""
        if 'data' not in st.session_state:
            st.session_state.data = {}
        
        for symbol in self.symbols:
            try:
                # Get more historical data for better indicators
                request_params = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Hour,
                    start=datetime.now() - timedelta(days=30)
                )
                
                bars = self.historical_client.get_stock_bars(request_params).df
                
                if not bars.empty:
                    # Reset index to get timestamp as a column
                    bars = bars.reset_index()
                    bars = bars.rename(columns={
                        'close': 'Close', 'open': 'Open', 
                        'high': 'High', 'low': 'Low', 'volume': 'Volume'
                    })
                    
                    # Set timestamp as index
                    bars = bars.set_index('timestamp')
                    
                    # Add technical indicators
                    bars = self.technical_analysis.add_indicators(bars)
                    st.session_state.data[symbol] = bars
                    
                    logger.info(f"Initialized data for {symbol} with {len(bars)} bars.")
                else:
                    st.session_state.data[symbol] = pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Failed to initialize data for {symbol}: {e}")
                st.session_state.data[symbol] = pd.DataFrame()

    def update_data(self, bar):
        """Update data with new bar"""
        symbol = bar.symbol
        if symbol in st.session_state.data:
            try:
                new_row = pd.DataFrame([{
                    'Open': bar.open, 'High': bar.high, 'Low': bar.low,
                    'Close': bar.close, 'Volume': bar.volume
                }], index=[pd.to_datetime(bar.timestamp)])
                
                # Add to existing data
                st.session_state.data[symbol] = pd.concat([st.session_state.data[symbol], new_row])
                
                # Keep only last 1000 bars to manage memory
                if len(st.session_state.data[symbol]) > 1000:
                    st.session_state.data[symbol] = st.session_state.data[symbol].tail(1000)
                
                # Recalculate indicators
                st.session_state.data[symbol] = self.technical_analysis.add_indicators(st.session_state.data[symbol])
                
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")

    def execute_trade(self, symbol: str, signal_info: dict):
        """Execute trade based on signal"""
        try:
            # Get current account info
            account = self.trading_client.get_account()
            account_value = float(account.portfolio_value)
            
            # Get current position
            try:
                position = self.trading_client.get_open_position(symbol)
                current_qty = int(position.qty)
            except:
                current_qty = 0
            
            # Get current positions count
            positions = self.trading_client.get_all_positions()
            current_positions = len(positions)
            
            # Check if we should trade
            daily_pnl = float(account.equity) - float(account.last_equity)
            if not self.risk_manager.should_trade(current_positions, daily_pnl, account_value):
                return "Risk limits exceeded"
            
            signal = signal_info['signal']
            current_price = st.session_state.data[symbol]['Close'].iloc[-1]
            
            if signal == 1 and current_qty == 0:  # Buy signal
                # Calculate position size
                stop_loss_price = current_price * (1 - self.strategy.stop_loss)
                qty = self.risk_manager.calculate_position_size(account_value, current_price, stop_loss_price)
                
                if qty > 0:
                    order_data = MarketOrderRequest(
                        symbol=symbol, 
                        qty=qty, 
                        side=OrderSide.BUY, 
                        time_in_force=TimeInForce.DAY
                    )
                    
                    order = self.trading_client.submit_order(order_data=order_data)
                    return f"BUY {qty} shares at ${current_price:.2f}"
                    
            elif signal == -1 and current_qty > 0:  # Sell signal
                order_data = MarketOrderRequest(
                    symbol=symbol, 
                    qty=abs(current_qty), 
                    side=OrderSide.SELL, 
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_data=order_data)
                return f"SELL {abs(current_qty)} shares at ${current_price:.2f}"
                
            return "HOLD"
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return f"Trade failed: {str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("🤖 AI-Powered Live Trading System")

# --- Initialize Alpaca Connection ---
try:
    # Check if secrets are available
    if "API_KEY" not in st.secrets or "SECRET_KEY" not in st.secrets:
        st.error("Please add your Alpaca API credentials to secrets.toml")
        st.stop()
    
    trading_client = TradingClient(st.secrets["API_KEY"], st.secrets["SECRET_KEY"], paper=True)
    
    # Test connection
    account = trading_client.get_account()
    st.success(f"✅ Connected to Alpaca Paper Trading API")
    
except Exception as e:
    st.error(f"❌ Failed to connect to Alpaca: {e}")
    st.info("Please check your API credentials in secrets.toml")
    st.stop()

# --- Initialize Session State ---
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = LiveTradingSystem(
        symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'], 
        trading_client=trading_client
    )
    st.session_state.trading_system.initialize_data()
    st.session_state.trade_log = []
    st.session_state.is_running = False

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎮 Trading Controls")
    
    # Trading status
    if st.session_state.is_running:
        st.success("🟢 System Running")
        if st.button("⏹️ Stop Trading", type="primary"):
            st.session_state.is_running = False
            st.rerun()
    else:
        st.info("🔴 System Stopped")
        if st.button("▶️ Start Trading", type="primary"):
            st.session_state.is_running = True
            st.rerun()
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    selected_symbols = st.multiselect(
        "Select Trading Symbols",
        options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META'],
        default=['AAPL', 'GOOGL', 'MSFT']
    )
    
    refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Manual refresh
    if st.button("🔄 Refresh Data"):
        st.session_state.trading_system.initialize_data()
        st.success("Data refreshed!")

# --- Main Dashboard ---
# Account Overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Portfolio Value", f"${float(account.portfolio_value):,.2f}")
    
with col2:
    st.metric("💵 Available Cash", f"${float(account.cash):,.2f}")
    
with col3:
    st.metric("📊 Buying Power", f"${float(account.buying_power):,.2f}")
    
with col4:
    daily_pnl = float(account.equity) - float(account.last_equity)
    st.metric("📈 Today's P&L", f"${daily_pnl:,.2f}", delta=f"{(daily_pnl/float(account.last_equity)*100):.2f}%")

st.divider()

# Live Trading Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Live Charts")
    
    # Symbol selector
    if st.session_state.data:
        symbol_to_chart = st.selectbox(
            "Select Symbol to Chart:",
            options=list(st.session_state.data.keys()),
            index=0
        )
        
        if symbol_to_chart and not st.session_state.data[symbol_to_chart].empty:
            df = st.session_state.data[symbol_to_chart]
            
            # Create candlestick chart
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Add moving averages
            if 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['SMA_20'], 
                    mode='lines', 
                    name='SMA 20',
                    line=dict(color='orange')
                ))
                
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['SMA_50'], 
                    mode='lines', 
                    name='SMA 50',
                    line=dict(color='blue')
                ))
            
            # Add Bollinger Bands
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Upper'], 
                    mode='lines', 
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_Lower'], 
                    mode='lines', 
                    name='BB Lower',
                    line=dict(color='gray', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{symbol_to_chart} - Live Price Action",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            if len(df) > 0:
                latest = df.iloc[-1]
                
                ind_col1, ind_col2, ind_col3 = st.columns(3)
                
                with ind_col1:
                    if 'RSI' in df.columns:
                        rsi_color = "red" if latest['RSI'] > 70 else ("green" if latest['RSI'] < 30 else "gray")
                        st.metric("RSI", f"{latest['RSI']:.2f}", delta=None)
                        
                with ind_col2:
                    if 'MACD' in df.columns:
                        macd_signal = "🔴" if latest['MACD'] < latest['MACD_Signal'] else "🟢"
                        st.metric("MACD", f"{latest['MACD']:.4f}", delta=f"{macd_signal}")
                        
                with ind_col3:
                    if 'Volume_Ratio' in df.columns:
                        vol_signal = "🔊" if latest['Volume_Ratio'] > 1.5 else "🔇"
                        st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}", delta=f"{vol_signal}")

with col2:
    st.header("📋 Trading Signals")
    
    # Generate signals for all symbols
    if st.session_state.data:
        for symbol in st.session_state.data.keys():
            if not st.session_state.data[symbol].empty:
                signal_info = st.session_state.trading_system.strategy.generate_signal(
                    st.session_state.data[symbol], symbol
                )
                
                # Display signal
                if signal_info['signal'] == 1:
                    st.success(f"🟢 **{symbol}** - BUY")
                elif signal_info['signal'] == -1:
                    st.error(f"🔴 **{symbol}** - SELL")
                else:
                    st.info(f"⚪ **{symbol}** - HOLD")
                
                st.caption(f"Confidence: {signal_info['confidence']:.2f} | {signal_info['reason']}")
                st.divider()

# Positions and Orders
st.header("💼 Current Positions")
try:
    positions = trading_client.get_all_positions()
    if positions:
        pos_data = []
        for pos in positions:
            pos_data.append({
                'Symbol': pos.symbol,
                'Quantity': pos.qty,
                'Market Value': f"${float(pos.market_value):,.2f}",
                'Unrealized P&L': f"${float(pos.unrealized_pl):,.2f}",
                'Unrealized P&L %': f"{float(pos.unrealized_plpc) * 100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No open positions")
except Exception as e:
    st.error(f"Error fetching positions: {e}")

# Trading Log
st.header("📜 Trading Log")
if st.session_state.trade_log:
    log_df = pd.DataFrame(st.session_state.trade_log)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No trades executed yet")

# Auto-refresh when running
if st.session_state.is_running:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.divider()
st.caption("⚠️ This is a paper trading system for educational purposes only. Not financial advice.")