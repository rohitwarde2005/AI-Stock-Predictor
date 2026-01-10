import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

NEWS_API_KEY = "15a39ffbd0ee4a22b40bd6e7a56a6351"

def safe_combine(base_array, forecast_array, hist_len=60, forecast_len=7):
    base_list = list(base_array) if base_array is not None else []
    forecast_list = list(forecast_array) if forecast_array is not None else []
    base_list = (base_list[-hist_len:] if len(base_list) > hist_len else [np.nan]*(hist_len - len(base_list)) + base_list)
    forecast_list = (forecast_list[:forecast_len] if len(forecast_list) > forecast_len else forecast_list + [np.nan]*(forecast_len - len(forecast_list)))
    return base_list + forecast_list

def fetch_stock(ticker, period_days=365):
    end = datetime.today()
    start = end - timedelta(days=period_days)
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def fetch_news(query, topn=5):
    ticker_to_name = {"INFY.NS": "Infosys"}
    search_query = ticker_to_name.get(query.upper(), query)
    try:
        url = f"https://newsapi.org/v2/everything?q={search_query}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize={topn}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            items = r.json().get("articles", [])
            if items:
                return [{"title": a.get("title"), "desc": a.get("description"), "url": a.get("url"), "source": a.get("source", {}).get("name")} for a in items]
    except Exception:
        pass
    try:
        yurl = f"https://query1.finance.yahoo.com/v1/finance/search?q={search_query}"
        r = requests.get(yurl, timeout=8).json()
        news = []
        for n in r.get("news", [])[:topn]:
            news.append({"title": n.get("title"), "desc": n.get("summary"), "url": n.get("link"), "source": n.get("provider")})
        if news:
            return news
    except Exception:
        pass
    return []

def compute_indicators(df):
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_low"] = df["BB_mid"] - 2 * df["BB_std"]
    return df

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def predict_lstm(df_close, lookback=60, epochs=3):
    values = df_close.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i,0])
        y.append(scaled[i,0])
    X = np.array(X); y = np.array(y)
    if len(X) < 10:
        return None, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm((X.shape[1],1))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    last_seq = scaled[-lookback:].reshape(1,lookback,1)
    preds = []
    for _ in range(7):
        p = model.predict(last_seq, verbose=0)[0][0]
        preds.append(p)
        last_seq = np.append(last_seq[:,1:,:], [[[p]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return preds, model, scaler

def predict_lr(df_close):
    df_close = np.asarray(df_close).flatten()
    df = pd.DataFrame({"y": df_close})
    df["t"] = np.arange(len(df))
    X = df[["t"]].values
    y = df["y"].values
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(df), len(df)+7).reshape(-1,1)
    preds = model.predict(future_t)
    return preds, model

def make_insights(today_price, lstm_avg, lr_avg, volatility):
    insights = []
    if lstm_avg is not None:
        pct = (lstm_avg - today_price) / today_price * 100
        insights.append(f"LSTM expects {pct:+.2f}% change (7-day avg).")
    if lr_avg is not None:
        pct2 = (lr_avg - today_price) / today_price * 100
        insights.append(f"Linear model expects {pct2:+.2f}% change (7-day avg).")
    if volatility > 0.03 * today_price:
        insights.append("High volatility detected — risk elevated.")
    else:
        insights.append("Volatility moderate — model predictions more stable.")
    return insights

def risk_level(preds):
    std = np.std(preds)
    if std / np.mean(preds) > 0.06:
        return "High"
    elif std / np.mean(preds) > 0.03:
        return "Medium"
    else:
        return "Low"
