import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from backend import (
    safe_combine, fetch_stock, fetch_news, compute_indicators,
    predict_lstm, predict_lr, make_insights, risk_level
)

st.set_page_config(page_title="Pro Stock Predictor", layout="wide")
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

def inject_css():
    if theme == "Dark":
        st.markdown("""
        <style>
        .reportview-container {background-color:#101217;}
        .sidebar .sidebar-content {background-color:#181a22;}
        .big-header {background: linear-gradient(90deg,#6168ff,#44e9ff); padding: 26px; border-radius:16px; color:white; font-size:28px; text-align:center;}
        .metric {background: linear-gradient(135deg,#fc8e8e,#b6f0e9); color:#222; padding:12px; border-radius:10px;}
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .big-header {background: linear-gradient(90deg,#8ee3fc,#e3fc8e); padding:26px; border-radius:16px; color:#085; font-size:28px; text-align:center;}
        </style>""", unsafe_allow_html=True)
inject_css()

st.markdown('<div class="big-header">📈 Stock Price Predictor: LSTM & LR</div>', unsafe_allow_html=True)
st.write(" ")

with st.sidebar:
    st.header("🛠 Controls")
    ticker = st.text_input("💹 Stock ticker", value="INFY.NS").upper()
    days = st.slider("📅 History (days)", 180, 1460, 365)
    method = st.selectbox("🧠 Prediction method", ["Both Models", "LSTM Only", "Linear Regression Only"])
    run = st.button("🔮 Predict Next 7 Days")
    st.markdown("---")
    st.markdown("## 🗂 Export and Info")
    st.write("After data loads, export full history below.")

df = fetch_stock(ticker, period_days=days)
if df is None:
    st.error("No data found — check ticker or internet.")
    st.stop()
df_ind = compute_indicators(df)
with st.sidebar:
    st.download_button(
        "⬇ Download History (CSV)", 
        data=df.to_csv(), 
        file_name=f"{ticker}_history.csv",
        mime="text/csv"
    )

st.markdown("### 📊 Market Stats")
col1, col2, col3, col4 = st.columns(4)
try:
    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2])
    daily_change = (current_price - prev_price) / prev_price * 100
    week_high = float(df["High"].iloc[-7:].max())
    week_low = float(df["Low"].iloc[-7:].min())
except Exception:
    current_price = float(df["Close"].iat[-1])
    prev_price = float(df["Close"].iat[-2])
    daily_change = (current_price - prev_price) / prev_price * 100
    week_high = float(df["High"].tail(7).max())
    week_low = float(df["Low"].tail(7).min())
col1.metric("Current Price (₹)", f"{current_price:.2f}", delta=f"{daily_change:.2f}%")
col2.metric("Daily Change", f"{daily_change:.2f}%")
col3.metric("7d High (₹)", f"{week_high:.2f}")
col4.metric("7d Low (₹)", f"{week_low:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Position","🧠 LSTM Model","📊 Linear Regression","📰 News & Export"])

with tab1:
    st.subheader(f"{ticker} - Chart ({days} days)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#008fd5")))
    fig.add_trace(go.Scatter(x=df.index, y=df_ind["SMA_20"], name="SMA20", line=dict(dash="dot", color="#6f4be7")))
    fig.add_trace(go.Scatter(x=df.index, y=df_ind["SMA_50"], name="SMA50", line=dict(dash="dash", color="#f95d6a")))
    fig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=450, xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Indicators")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("*RSI (14)*")
        rfig = go.Figure(go.Scatter(x=df.index, y=df_ind["RSI"], name="RSI", line=dict(color="#FE7F2D")))
        rfig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=250, yaxis=dict(range=[0,100]))
        st.plotly_chart(rfig, use_container_width=True)
    with c2:
        st.markdown("*Bollinger Bands*")
        bfig = go.Figure()
        bfig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
        bfig.add_trace(go.Scatter(x=df.index, y=df_ind["BB_up"], name="BB Up", line=dict(dash="dash")))
        bfig.add_trace(go.Scatter(x=df.index, y=df_ind["BB_low"], name="BB Low", line=dict(dash="dash")))
        bfig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=250)
        st.plotly_chart(bfig, use_container_width=True)
    st.markdown("### News")
    news = fetch_news(ticker)
    if news:
        for n in news:
            st.markdown(f"[{n['title']}]({n['url']})")
            if n.get("desc"):
                st.caption(n["desc"])
            else:
                st.caption(n.get("source", ""))
            st.markdown("---")
    else:
        st.info("No news available.")

with tab2:
    st.subheader("🧠 LSTM Model Performance")
    if method in ["Both Models", "LSTM Only"]:
        with st.spinner("Training LSTM..."):
            preds_lstm, model_lstm, scaler = predict_lstm(df["Close"].values, lookback=60, epochs=3)
        if preds_lstm is None:
            st.warning("Not enough data to train LSTM.")
        else:
            lstm_avg = float(np.mean(preds_lstm))
            future_dates = pd.date_range(df.index[-1] + timedelta(1), periods=7)
            pred_df_lstm = pd.DataFrame({"Date": future_dates, "LSTM Predicted": preds_lstm})
            st.dataframe(pred_df_lstm.style.format({"LSTM Predicted":"{:.2f}"}), height=240)
            vol = float(np.std(preds_lstm))
            st.metric("LSTM 7-Day Avg", f"{lstm_avg:.2f} ₹")
            st.metric("Volatility (std)", f"{vol:.2f}")
            st.markdown(f"*Risk Level:* {risk_level(preds_lstm)}")
            pct_change = (lstm_avg - current_price) / current_price * 100
            if pct_change <= -5:
                st.error(f"⚠ Large drop forecast: {pct_change:.2f}%")
            elif pct_change >= 5:
                st.success(f"📈 Strong rise forecast: {pct_change:.2f}%")
            else:
                st.info(f"Predicted change: {pct_change:.2f}%")
            overlay_dates = list(df.index[-60:]) + list(future_dates)
            overlay_actual = safe_combine(df["Close"], None)
            overlay_lstm = safe_combine(df["Close"], preds_lstm)
            overlay_fig = go.Figure()
            overlay_fig.add_trace(go.Scatter(x=overlay_dates, y=overlay_actual, name="Recent Close", mode="lines+markers", line=dict(color="royalblue")))
            overlay_fig.add_trace(go.Scatter(x=overlay_dates, y=overlay_lstm, name="LSTM Forecast", mode="lines+markers", line=dict(dash="dash",color="firebrick")))
            overlay_fig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=450)
            st.plotly_chart(overlay_fig, use_container_width=True)

with tab3:
    st.subheader("📊 Linear Regression Performance")
    if method in ["Both Models", "Linear Regression Only"]:
        preds_lr, lr_model = predict_lr(df["Close"].values)
        lr_avg = float(np.mean(preds_lr))
        st.metric("LR 7-Day Avg", f"{lr_avg:.2f} ₹")
        st.dataframe(pd.DataFrame({"Date": pd.date_range(df.index[-1] + timedelta(1), periods=7), "LR Predicted": preds_lr}).style.format({"LR Predicted":"{:.2f}"}), height=200)
        pct_lr = (lr_avg - current_price) / current_price * 100
        if pct_lr <= -5:
            st.error(f"⚠ Drop forecast: {pct_lr:.2f}%")
        elif pct_lr >= 5:
            st.success(f"📈 Rise forecast: {pct_lr:.2f}%")
        else:
            st.info(f"Predicted change: {pct_lr:.2f}%")
        future_dates = pd.date_range(df.index[-1] + timedelta(1), periods=7)
        actual_segment = df["Close"]
        lstm_segment = preds_lstm if 'preds_lstm' in locals() and preds_lstm is not None else None
        lr_segment = preds_lr if 'preds_lr' in locals() and preds_lr is not None else None
        comp_dates = list(df.index[-60:]) + list(future_dates)
        comp_actual = safe_combine(actual_segment, None)
        comp_lstm = safe_combine(actual_segment, lstm_segment)
        comp_lr = safe_combine(actual_segment, lr_segment)
        comp_df = pd.DataFrame({
            "date": comp_dates,
            "Actual": comp_actual,
            "LSTM": comp_lstm,
            "LR": comp_lr
        })
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["Actual"], name="Actual"))
        comp_fig.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["LSTM"], name="LSTM"))
        comp_fig.add_trace(go.Scatter(x=comp_df["date"], y=comp_df["LR"], name="LR"))
        comp_fig.update_layout(template="plotly_dark" if theme=="Dark" else "plotly_white", height=480)
        st.plotly_chart(comp_fig, use_container_width=True)

with tab4:
    st.subheader("📰 Latest News")
    news = fetch_news(ticker)
    if news:
        for n in news:
            st.markdown(f"[{n['title']}]({n['url']})")
            if n.get("desc"):
                st.caption(n["desc"])
            st.markdown("---")
    else:
        st.info("No news available.")
    out_df = None
    if 'preds_lstm' in locals() and preds_lstm is not None:
        out_df = pd.DataFrame({
            "Date": pd.date_range(df.index[-1] + timedelta(1), periods=7),
            "LSTM_Pred": preds_lstm
        })
    if 'preds_lr' in locals():
        lr_df = pd.DataFrame({
            "Date": pd.date_range(df.index[-1] + timedelta(1), periods=7),
            "LR_Pred": preds_lr
        })
        out_df = out_df.merge(lr_df, on="Date", how="outer") if out_df is not None else lr_df
    if out_df is not None:
        out_df = out_df.set_index("Date")
        st.download_button("⬇ Download Forecast (CSV)", data=out_df.to_csv(), file_name=f"{ticker}_7day_forecast.csv", mime="text/csv")
    lstm_avg = float(np.mean(preds_lstm)) if 'preds_lstm' in locals() and preds_lstm is not None else None
    lr_avg = float(np.mean(preds_lr)) if 'preds_lr' in locals() else None
    vol = float(np.std(preds_lstm)) if 'preds_lstm' in locals() and preds_lstm is not None else 0.0
    insights = make_insights(current_price, lstm_avg, lr_avg, vol)
    st.subheader("🧾 AI-Generated Insights")
    for ins in insights:
        st.write(f"- {ins}")

st.markdown("---")
st.caption("Built with Streamlit • Data from Yahoo Finance • News via NewsAPI")
