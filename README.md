# 📈 AI-Powered 7-Day Stock Price Forecasting System


An intelligent stock market prediction web application built using Python, Streamlit, LSTM Neural Networks, and Linear Regression.  
The system fetches real-time stock data, computes technical indicators, predicts future prices, analyzes risk, and displays financial news.

---

## 🔹 Features

- Real-time stock price fetching (Yahoo Finance)
- LSTM deep learning model for short-term prediction
- Linear Regression model for trend comparison
- Technical indicators (SMA, RSI, Bollinger Bands)
- Live financial news integration
- Risk level & volatility analysis
- CSV export for historical & forecasted data
- Dark / Light theme UI
- Interactive charts using Plotly

---

## 🛠️ Technologies Used

- Python 3.9 – 3.11
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Yahoo Finance API
- NewsAPI
- Plotly

---

## 📂 Project Structure

Stock-Predictor/
│
├── app.py
├── backend.py
├── requirements.txt
└── README.md


---

## ⚙️ Step-by-Step Setup Guide

### Step 1: Install Python

python --version


or


python3 --version


---

### Step 2: Create Virtual Environment (Recommended)



python -m venv venv


Activate virtual environment:

**Windows**


venv\Scripts\activate


**Linux / macOS**


source venv/bin/activate


---

### Step 3: Install Required Libraries

Create `requirements.txt`:


streamlit
pandas
numpy
yfinance
scikit-learn
tensorflow
plotly
requests


Install dependencies:


pip install -r requirements.txt


---

### Step 4: Configure News API Key

Open `backend.py` and replace:


NEWS_API_KEY = "YOUR_API_KEY_HERE"


Get your key from:
https://newsapi.org/

---

### Step 5: Run the Application



streamlit run app.py


---

### Step 6: Open in Browser

Open the following URL in your browser:


http://localhost:8501


---

## 🧪 How to Use

1. Enter stock ticker (example: INFY.NS, TCS.NS)
2. Select historical data range
3. Choose prediction model
4. Click "Predict Next 7 Days"
5. View charts, predictions, risk level, and news
6. Download CSV reports if needed

---

## 📊 Prediction Models

### LSTM Model
- Uses past 60 days data
- Captures time-series patterns
- Suitable for short-term forecasting

### Linear Regression Model
- Identifies overall trend
- Fast and simple baseline model

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
Stock market investments involve risk.  
Predictions should not be considered financial advice.

---

## 🎓 Academic Use

Suitable for:
- Final Year Project
- Mini Project
- AI / ML Portfolio
- Resume & GitHub Showcase

---

## 👨‍💻 Author

Developed by Rohit  
Project Type: AI-Based Stock Prediction System  
Platform: Python + Streamlit