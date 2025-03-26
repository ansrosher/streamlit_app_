#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import tempfile
import os
import json
from PIL import Image
import tempfile
from datetime import datetime, timedelta
import plotly.io as pio
import mplfinance as mpf 
# Import the GoogleNews package to fetch the news articles
from GoogleNews import GoogleNews
import pandas as pd
import numpy as np
import statistics
import yfinance as yf
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import time
# Ignore all warnings
import warnings
from tabulate import tabulate
from termcolor import colored
warnings.filterwarnings("ignore")
from termcolor import colored
# Ia libriries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from itertools import product
from datetime import datetime, timezone, timedelta
import time

import os
import io
import time
import math
import asyncio
import pickle
import subprocess
import warnings
from datetime import datetime, timezone, timedelta, date

# Web & API Handling
import requests
import telegram
from flask import request
from markupsafe import Markup
import investpy

# Data Handling
import pandas as pd
import numpy as np
import statistics
from pandas.errors import SettingWithCopyWarning
import statsmodels.api as sm
import statsmodels.tsa.api as ts
# Finance & Crypto APIs
import yfinance as yf

# Technical & Financial Analysis
from ta.trend import MACD
from ta.momentum import RSIIndicator

# News Scraping
from GoogleNews import GoogleNews

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tabulate import tabulate
from termcolor import colored

# Machine Learning & AI
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Suppress Warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



# Tu API Key de FRED (reemplázala con la tuya)
API_KEY = "48cb461760a41705b9d9a26fcbd97dad"


pio.kaleido.scope.default_format = "png"

# Configure the API key - IMPORTANT: Use Streamlit secrets or environment variables for security
# For now, using hardcoded API key - REPLACE WITH YOUR ACTUAL API KEY SECURELY
GOOGLE_API_KEY = "AIzaSyB0fvpnzxWXib63banV-Uf8bEcfJwhBLrI"
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash' # or other model
gen_model = genai.GenerativeModel(MODEL_NAME)

# Configuración inicial
st.set_page_config(layout="wide", page_title="Quanthika")

# CSS personalizado inspirado en Robinhood (basado en la imagen)
import streamlit as st

st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Questrial&display=swap');
        
        /* Main App Background */
        .stApp {
            background-color: #F5F5F5;  /* Fondo gris claro para neutralidad */
            color: #000000;            /* Texto negro para legibilidad */
            font-family: 'Questrial', sans-serif;
            margin: 0;
            padding: 0;
        }
        
        /* Remove default margins and paddings */
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #0097b2;  /* Turquesa para la barra lateral */
            border-right: 1px solid #007a8f; /* Tono más oscuro de turquesa */
            padding-top: 0;
        }
        .sidebar-title {
            color: #FFFFFF;            /* Texto blanco en la sidebar */
            font-size: 24px;
            padding: 1rem;
            border-bottom: 1px solid #007a8f;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        /* Buttons */
        .stButton>button {
            width: 100%;
            background-color: #7ed957; /* Verde lima para botones principales */
            color: #000000;           /* Texto negro para contraste */
            border: none;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.3rem 0;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #6cc744; /* Verde lima más oscuro al pasar el ratón */
            color: #000000;
            transform: translateY(-2px);
        }
        
        /* Secondary Button (Sign Up) */
        .secondary-signup-button {
            background-color: #cbfd0b; /* Amarillo neón para destacar */
            color: #000000;           /* Texto negro */
            border: 1px solid #a8cc09;
        }
        .secondary-signup-button:hover {
            background-color: #b8e009; /* Amarillo un poco más oscuro */
            color: #000000;
        }
        
        /* Secondary Button (Log In) */
        .secondary-login-button {
            background-color: transparent;
            color: #000000;           /* Texto Negro */
            border: 1px solid #000000;
        }
        .secondary-login-button:hover {
            background-color: rgba(255, 255, 255, 0.1);
            color: #000000;
        }
        
        /* Inputs */
        .stTextInput>div>div>input, 
        .stDateInput>div>div>input,
        .stSelectbox>div>div>select {
            background-color: #FFFFFF; /* Fondo blanco */
            color: #000000;           /* Texto negro */
            border: 1px solid #0097b2; /* Borde turquesa */
            border-radius: 5px;
            padding: 0.5rem;
        }
        
        /* Fix for input and selectbox labels */
        .stTextInput>label,
        .stSelectbox>label {
            color: #000000 !important; /* Texto negro para etiquetas */
        }
        
        /* Content Boxes */
        .content-box {
            background-color: #FFFFFF; /* Fondo blanco para las cajas */
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 0;
            width: 100%;
            box-sizing: border-box;
            color: #000000;           /* Texto negro */
        }
        
        /* Headings */
        h1 {
            color: #0097b2;           /* Turquesa para títulos principales */
            font-weight: 700;
            margin-top: 0;
        }
        h2 {
            color: #7ed957;           /* Verde lima para subtítulos */
            font-weight: 400;
        }
        
        /* Sidebar Section Text */
        .section-title {
            color: #cbfd0b;           /* Amarillo neón para secciones en sidebar */
            font-size: 18px;
            font-weight: 400;
            margin: 1rem 0 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Inicialización de Session State al inicio
if 'pagina' not in st.session_state:
    st.session_state.pagina = "principal"
if 'subpagina' not in st.session_state:
    st.session_state.subpagina = "basica"
# Funciones aplicaciones
def get_economic_data():
    # Diccionario con los indicadores económicos y sus nombres en FRED
    economic_indicators = {
        "REAL_GDP": "GDP",
        "TREASURY_YIELD": "DGS10",
        "FEDERAL_FUNDS_RATE": "FEDFUNDS",
        "CPI": "CPIAUCSL",
        "INFLATION": "FPCPITOTLZGUSA",
        "UNEMPLOYMENT": "UNRATE",
        "NONFARM_PAYROLL": "PAYEMS",
        "ICC": "UMCSENT",
        "RETAIL_SALES": "RSXFS",
        "YIELD_CURVE": "T10Y2Y"
    }
    
    # URL base de la API de FRED
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    data = []
    
    for name, series_id in economic_indicators.items():
        params = {
            "series_id": series_id,
            "api_key": API_KEY,
            "file_type": "json",
            "limit": 2,  # Obtener los dos datos más recientes
            "sort_order": "desc"
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            json_data = response.json()
            
            try:
                observations = json_data.get("observations", [])
                if len(observations) >= 2:
                    date_now, value_now = observations[0]["date"], round(float(observations[0]["value"]),2)
                    date_last, value_last = observations[1]["date"], round(float(observations[1]["value"]),2)
                    percent_change = round(((value_now - value_last) / value_last) * 100,2) if value_last != 0 else None
                    data.append({
                        "Indicator": name,
                        "Date_last": date_last,
                        "Value_last": value_last,
                        "Date_now": date_now,
                        "Value_now": value_now,
                        "Percent_change %": percent_change
                    })
                else:
                    raise ValueError("No hay suficientes datos disponibles")
            except Exception as e:
                print(f"Error al obtener {name}: {e}")
        else:
            print(f"Error en {name}: {response.status_code}")
    
    # Convertir a DataFrame
    df = pd.DataFrame(data)
    return df
def get_ticker_from_word(search_word):
    try:
        # Yahoo Finance search endpoint
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={search_word}"
        
        # Make the request
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers)
        data = response.json()
        
        # Check if we got any results
        if 'quotes' in data and len(data['quotes']) > 0:
            # Get the first matching result
            for quote in data['quotes']:
                if quote['quoteType'] == 'EQUITY':
                    symbol = quote['symbol']
                    # Verify the ticker exists
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if 'symbol' in info:
                        return symbol
            return None, f"No valid equity ticker found for {search_word}"
        else:
            return None, f"No results found for {search_word}"
            
    except Exception as e:
        return None, f"Error searching for ticker: {str(e)}"
        
# Get top economic news from TradingEconomics (requires API key)
def get_economic_news():
    url = "https://api.tradingeconomics.com/news?c=guest:guest&f=json"
    response = requests.get(url)
    if response.status_code == 200:
        news = response.json()
        return news[:5]  # Get top 5 important news items
    return "Failed to fetch economic news"

def get_economic_calendar():
    # Get economic calendar data
    df = investpy.news.economic_calendar(
        time_zone=None,  # investpy outputs in UTC by default
        time_filter='time_only',
        countries=["united states","england"],
        importances=["high"],
        categories=None,
        from_date=None,
        to_date=None
    )
    df= df.drop(columns=["id", "currency","date","importance","zone"])

    return df

def get_price_changes():
    # Lista de activos
    assets = {
        # Índices financieros (símbolos de Yahoo Finance)
        "S&P 500": "^GSPC",
        "Nikkei 225": "^N225",
        "EURO STOXX 50": "^STOXX50E",
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
        # Criptomonedas (IDs para CoinGecko)
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum",
        "BNB": "binancecoin",
        "XRP": "ripple"
    }

    # Lista para almacenar resultados
    data = []

    # Fecha de inicio: retroceder lo suficiente para asegurar al menos 2 días hábiles
    start_date = (datetime.now() - timedelta(days=4 if datetime.now().weekday() == 0 else 2)).strftime('%Y-%m-%d')

    # Obtener datos de índices financieros desde yfinance
    for name, symbol in assets.items():
        if name in ["S&P 500", "Nikkei 225", "EURO STOXX 50", "Dow Jones", "VIX"]:
            try:
                #ticker = yf.Ticker(symbol)
                # Descargar datos desde el inicio calculado hasta hoy
                hist = yf.download(symbol, start=start_date, progress=False)
                #hist = ticker.history(start=start_date, interval="1d")
                if not hist.empty and len(hist) >= 2:
                    # Últimos dos cierres disponibles
                    latest_close = hist["Close"].iloc[-1][0]  # Cierre más reciente (lunes si disponible, viernes si no)
                    prev_close = hist["Close"].iloc[-2][0]    # Cierre de la jornada anterior
                    change_percent = ((latest_close - prev_close) / prev_close) * 100
                    data.append([name, round(prev_close, 2), round(latest_close, 2), round(change_percent, 2)])
                else:
                    data.append([name, "N/A", "N/A", "Datos insuficientes"])
            except Exception as e:
                data.append([name, "N/A", "N/A", f"Error: {str(e)}"])

    # Obtener datos de criptomonedas desde CoinGecko
    crypto_ids = ",".join([assets[name] for name in ["Bitcoin", "Ethereum", "BNB", "XRP"]])
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={crypto_ids}"
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        for coin in json_data:
            name = next(k for k, v in assets.items() if v == coin["id"])
            current_price = coin["current_price"]
            change_percent = coin["price_change_percentage_24h"]
            # Calcular precio anterior: current_price / (1 + change_percent/100)
            prev_price = current_price / (1 + change_percent / 100)
            data.append([name, round(prev_price, 2), round(current_price, 2), round(change_percent, 2)])
    else:
        print("Error al conectar con CoinGecko API")
        data.append(["Criptos", "N/A", "N/A", "Error en API"])

    # Crear DataFrame
    df = pd.DataFrame(data, columns=["Activo", "Precio Anterior", "Precio Actual", "% Cambio"])
    # Ajustar título de la columna de cambio según contexto
    df.columns = ["Activo", "Precio Anterior", "Precio Actual", 
                  "% Cambio"]
    return df

# Función de backtesting genérica
def backtest_strategy(df, signal_func, **params):
    # Generar señales
    df = signal_func(df, **params)
    
    # Calcular retornos
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    # Retornos acumulados
    df['cum_strategy'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    df['cum_bh'] = (1 + df['returns'].fillna(0)).cumprod()
    
    # Métricas de rendimiento
    retorno_estrategia = df['cum_strategy'].iloc[-1] - 1
    retorno_bh = df['cum_bh'].iloc[-1] - 1
    sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)) if df['strategy_returns'].std() != 0 else 0
    volatilidad = df['strategy_returns'].std() * np.sqrt(252)
    df['cum_max'] = df['cum_strategy'].cummax()
    df['drawdown'] = df['cum_strategy'] / df['cum_max'] - 1
    max_drawdown = df['drawdown'].min()
    
    return {
        'params': params,
        'retorno_estrategia': retorno_estrategia,
        'retorno_bh': retorno_bh,
        'sharpe': sharpe,
        'volatilidad': volatilidad,
        'drawdown': max_drawdown
    }, df

# Funciones de señal para cada indicador
def signal_sma(df, short_window, long_window):
    df['SMA_short'] = SMAIndicator(df['Close'], window=short_window).sma_indicator()
    df['SMA_long'] = SMAIndicator(df['Close'], window=long_window).sma_indicator()
    df['signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
    return df

def signal_rsi(df, window, buy_level, sell_level):
    df['rsi'] = RSIIndicator(df['Close'], window=window).rsi()
    df['signal'] = np.where(df['rsi'] < buy_level, 1, np.where(df['rsi'] > sell_level, 0, np.nan))
    df['signal'] = df['signal'].ffill().fillna(0)
    return df

def signal_macd(df, fast, slow, signal):
    macd = MACD(df['Close'], fast, slow, signal)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['signal'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
    return df

def signal_bollinger(df, window, std):
    bb = BollingerBands(df['Close'], window=window, window_dev=std)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['signal'] = np.where(df['Close'] < df['bb_low'], 1, np.where(df['Close'] > df['bb_high'], 0, np.nan))
    df['signal'] = df['signal'].ffill().fillna(0)
    return df

def signal_stochastic(df, k, d, buy_level, sell_level):
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=k, smooth_window=d)
    df['stoch_k'] = stoch.stoch()
    df['signal'] = np.where(df['stoch_k'] < buy_level, 1, np.where(df['stoch_k'] > sell_level, 0, np.nan))
    df['signal'] = df['signal'].ffill().fillna(0)
    return df


# Función para probar un indicador
def test_indicator(indicador, df, param_grid):
    signal_funcs = {
        'sma': signal_sma,
        'rsi': signal_rsi,
        'macd': signal_macd,
        'bollinger': signal_bollinger,
        'stochastic': signal_stochastic
    }
    
    if indicador not in signal_funcs:
        raise ValueError("Indicador no soportado")
    
    signal_func = signal_funcs[indicador]
    params_list = [dict(zip(param_grid[indicador].keys(), values)) for values in product(*param_grid[indicador].values())]
    
    resultados = []
    for params in params_list:
        metrics, _ = backtest_strategy(df.copy(), signal_func, **params)
        resultados.append(metrics)
    
    df_resultados = pd.DataFrame(resultados)
    mejor = df_resultados.loc[df_resultados['sharpe'].idxmax()]
    
    print(f"\nMejor combinación para {indicador}:")
    print(mejor)
    
    # Ejecutar nuevamente para graficar
    _, df_mejor = backtest_strategy(df.copy(), signal_func, **mejor['params'])
    image=plot_results(df_mejor, indicador, mejor['params'])
    
    return mejor,image

# Función para graficar
def plot_results(df, indicador, params):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot closing price
        ax.plot(df.index, df['Close'], color='blue', label='Close Price')
        
        # Plot indicator-specific lines
        if indicador == 'sma':
            ax.plot(df.index, df['SMA_short'], label=f"SMA {params['short_window']}")
            ax.plot(df.index, df['SMA_long'], label=f"SMA {params['long_window']}")
        elif indicador == 'rsi':
            ax.plot(df.index, df['rsi'], label=f"RSI {params['window']}")
            ax.axhline(y=params['buy_level'], color='g', linestyle='--', label='Buy Level')
            ax.axhline(y=params['sell_level'], color='r', linestyle='--', label='Sell Level')
        elif indicador == 'macd':
            ax.plot(df.index, df['macd'], label='MACD')
            ax.plot(df.index, df['macd_signal'], label='Signal')
        elif indicador == 'bollinger':
            ax.plot(df.index, df['bb_high'], label='BB High')
            ax.plot(df.index, df['bb_low'], label='BB Low')
        elif indicador == 'stochastic':
            ax.plot(df.index, df['stoch_k'], label='Stoch %K')
            ax.axhline(y=params['buy_level'], color='g', linestyle='--', label='Buy Level')
            ax.axhline(y=params['sell_level'], color='r', linestyle='--', label='Sell Level')
        
        # Plot buy/sell signals
        compras = df[df['signal'].diff() == 1]
        ventas = df[df['signal'].diff() == -1]
        ax.scatter(compras.index, compras['Close'], marker='^', color='g', label='Buy')
        ax.scatter(ventas.index, ventas['Close'], marker='v', color='r', label='Sell')
        
        # Customize plot
        ax.set_title(f"Backtesting {indicador.upper()}")
        ax.legend()
                    
        plt.savefig(tmp.name)
        tmpfile_path = tmp.name
    
        # Open with PIL if needed
        image = Image.open(tmpfile_path)

        return image
        
def find_best_strategy_indicator(ticker,indicator,days):
    Time_days=days
    start=datetime.now() -  timedelta(days=Time_days)
    start=start.strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start)
    data.columns = [col[0] for col in data.columns]
    df = data[['Close', 'High', 'Low']].copy()
    df.rename(columns={'Close': 'Close'}, inplace=True)
    
    # Diccionario de parámetros a probar para cada indicador
    param_grid = {
        'sma': {'short_window': [5, 10, 14, 21, 30], 'long_window': [50, 100, 150, 200]},
        'rsi': {'window': [7, 14, 21, 30], 'buy_level': [20, 30], 'sell_level': [70, 80]},
        'macd': {'fast': [5, 12,14], 'slow': [26,30, 35], 'signal': [5,7, 9]},
        'bollinger': {'window': [7,14,20, 30], 'std': [2, 2.5,3]},
        'stochastic': {'k': [14, 20], 'd': [3, 5], 'buy_level': [20, 30], 'sell_level': [70, 80]}}
    mejor,image = test_indicator(indicator, df, param_grid)
    return mejor, image
    
# Gathering all the data of the current page to one dataframe
def newsfeed1(raw_dictionary):
    # Dataframe to store the news article information
    article_info = pd.DataFrame(columns=['Date','Title','Link'])
    for i in range(len(raw_dictionary)-1):
        if raw_dictionary is not None:
            # Fetch the date and time and convert it into datetime format
            date = str(raw_dictionary[i]['date'])
            #date = pd.to_datetime(date)
            # Fetch the title, time, description and source of the news articles
            title = str(raw_dictionary[i]['title'])
            link = raw_dictionary[i]['link']
            # Append all the above information in a single dataframe
            article_info = article_info._append({'Date': date,'Title': title,'Link': link}, ignore_index=True)
        else:
            break

    return article_info

def sentiment_analysis(keywords):
    print("Obteniendo las noticias y articulos relacionados........")
    googlenews = GoogleNews()
    googlenews.set_period('1d') 
    googlenews = GoogleNews(lang='en', region='US')
    #googlenews.set_time_range(start, end)      
    # Dataframe containing the news of all the keywords searched
    articles = pd.DataFrame()
    
    # Each keyword will be searched seperately and results will be saved in a dataframe
    for steps in range(len(keywords)):
        string = (keywords[steps])
        googlenews.get_news(string)
    
        # Fetch the results
        result = googlenews.results()  
        feed = newsfeed1(result)    
        articles = articles._append(feed)
    
        # Clear off the search results of previous keyword to avoid duplication
        googlenews.clear()
    
    shape = articles.shape[0]
    
    # Resetting the index of the final result
    articles.index = np.arange(shape)
    #Sentiment analysis mean()
    print("Obteniendo sentiment analysis score........")
    vader_scores=[]
    textblob_scores=[]
    for i in range(len(articles)):
        title=articles["Title"][i]
        vader_scores.append(SentimentIntensityAnalyzer().polarity_scores(title)["compound"])
        textblob_scores.append(TextBlob(title).sentiment.polarity)
    return ((statistics.mean(vader_scores)+statistics.mean(textblob_scores))/2)

def get_stock_data(ticker,start,end):
    #stock=yf.Ticker(ticker)
    #df=stock.history(start=start,end=end)
    df = yf.download(ticker, start=start_date, progress=False)
    return df

def create_features(df,threshold):
    df["SMA_50"]= ta.sma(df["Close"],lenght=50)
    df["SMA_200"]= ta.sma(df["Close"],lenght=200)
    df["RSI"]= ta.rsi(df["Close"],lenght=14)
    df["Volatility"]= df["Close"].rolling(window=50).std()/df["Close"]
    df["Return"]=df["Close"].pct_change(1)
    df["dev_std"]=df["Close"].rolling(window=5).std()
    # Add Drawdowns
    df["Drawdown"] = (df["Close"] - df["Close"].cummax()) / df["Close"].cummax()
    
    # Add Volatility Indicator
    df["Volatility_Indicator"] = np.where(df["Drawdown"].abs() < df["Volatility"], 1, 0)
    
    df.dropna(inplace=True)
    
    # Create feature variable X
    X = df.drop(columns=['High', 'Low'])
    
    # Create target variable
    df["Crash"]=np.where(df["Return"].shift(-1) <    threshold,1,0)
    df['target'] = df["Return"].shift(-1)
    y = np.where(df.target > 0, 1, 0)
    y2= np.where(df.Crash > 0, 1, 0)
    return X, y, y2

def train_model(X, y):
    # Split the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False)

    # Convert to DataFrame to retain feature names
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # Scale the features data of both train and test dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------- Classifier 1: XGBoost ------------------------------------------
    xgb = XGBClassifier(max_depth=5, n_estimators=30,
                        random_state=42, eval_metric='logloss')
    
    # --------------- Classifier 2: Logistic------------------------------------------
    lr = LogisticRegression(random_state=42)
    
    # --------------- Classifier 3: SVM------------------------------------------
    svc = svm.SVC(kernel='rbf', probability=True, random_state=42)

    # --------------- Classifier 4: RandomForestClassifier------------------------------------------
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    # --------------- Classifier 5: MLPClassifier (Neural Network) ------------------------------------------
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(5,), 
                        random_state=42, solver='sgd')

    # Define a list to store the different models
    estimator = []
    estimator.append(('LR', lr))
    estimator.append(('SVC', svc))
    estimator.append(('XGB', xgb))
    estimator.append(('RFC', rfc))
    estimator.append(('MLP', mlp)) 

    # Implement voting classifier with hard voting
    model_vot_hard = VotingClassifier(estimators=estimator, voting='hard')

    # Fit the voting classifier model
    model_vot_hard.fit(X_train_scaled, y_train)

    # Implement voting classifier with soft voting 
    model_vot_soft = VotingClassifier(estimators=estimator, voting='soft')
    
    # Fit the voting classifier model
    model_vot_soft.fit(X_train_scaled, y_train)
    
    return X_test_scaled,X_test, model_vot_hard,model_vot_soft
    
def create_models(X,y,y2):
    X_test1_scaled,X_test1,model1_hard,model1_soft=train_model(X, y)
    X_test2_scaled,X_test2,model2_hard,model2_soft=train_model(X, y2)
    return X_test1_scaled,X_test2_scaled,X_test1,X_test2,model1_hard, model2_hard,model1_soft,model2_soft
def get_company_name(ticker):
    try:
        results = search(ticker)
        for result in results['quotes']:
            if result.get('symbol', '').upper() == ticker.upper():
                return result.get('longname', "Nombre no encontrado")
    except Exception as e:
        print(f"Error obteniendo el nombre de la empresa: {e}")
    return "Nombre no encontrado"

def print_widget_explanation():
    explanation = """   
    El Widget AI Predictive Analytics para Mercados Financieros utiliza modelos de Machine Learning
    y procesamiento de lenguaje natural (NLP) para analizar datos históricos, noticias y tendencias
    del mercado en tiempo real.
    
    Cómo se deben interpretar los valores:
    1. Predicción:
       - "Comprar": Se espera una tendencia alcista en el próximo período.
       - "Vender": Se espera una tendencia bajista en el próximo período.
       - "Neutral": No se detecta una tendencia clara en el próximo período.
    
    2. Probabilidad:
       - Indica la confianza del modelo en la predicción realizada.
       - Valores más altos representan mayor certeza en la decisión.
    
    3. Predicción de Crash:
       - Si la posibilidad de caída supera el umbral (5% para Bitcoin, 2% para otros índices),
         se emite una alerta de posible movimiento fuerte a la baja.
    
    4. Puntaje de Sentimiento:
       - Valores > 0.05 indican sentimiento positivo en las noticias analizadas.
       - Valores entre -0.05 y 0.05 indican sentimiento neutral.
       - Valores < -0.05 indican sentimiento negativo.
    
    Nota Importante:
    - Este widget proporciona predicciones basadas en modelos estadísticos y de Machine Learning,
      pero NO constituye un consejo de inversión. Las decisiones financieras deben tomarse
      considerando múltiples factores y análisis adicionales.
    """
    
    return explanation
# Funciones
def pagina_principal():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("Bienvenido")
        st.write("Utilice el panel lateral para acceder a las herramientas de inversión")
        st.markdown('</div>', unsafe_allow_html=True)

def submenu_predictor_ia():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("Predictor IA")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predicción Básica", key="basica"):
                st.session_state.subpagina = "basica"
            if st.session_state.subpagina == "basica":
                st.markdown('<span class="active-indicator">Activo</span>', unsafe_allow_html=True)
        with col2:
            if st.button("Predicción Avanzada", key="avanzada"):
                st.session_state.subpagina = "avanzada"
            if st.session_state.subpagina == "avanzada":
                st.markdown('<span class="active-indicator">Activo</span>', unsafe_allow_html=True)
        
        fecha_inicio = datetime.now() - timedelta(days=365)
        fecha_fin = datetime.now()
        simbolo = st.text_input("Símbolo", "AAPL")
        simbolo = get_ticker_from_word(simbolo)
        if st.button("Predecir"):        
            if st.session_state.subpagina == "basica":
                st.subheader("Predicción Básica")
                st.write(f"Configurado para {simbolo} desde {fecha_inicio} hasta {fecha_fin}")
                st.write(f"Prediciendo para {simbolo}...")
            
                # Download stock data
                data = yf.download(simbolo, start=fecha_inicio, end=fecha_fin)
                data.columns = [col[0] for col in data.columns]
                if not data.empty:
                    stock_data = data
                else:
                    st.warning(f"No data found for {simbolo}.")
                    st.stop()
            
                def analyze_ticker(ticker, data):
                    # Calculate indicators
                    sma = data['Close'].rolling(window=20).mean()
                    std = data['Close'].rolling(window=20).std()
                    bb_upper = sma + 2 * std
                    bb_lower = sma - 2 * std
                    VWAP = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                    
                    # Prepare additional plots
                    apds = [
                        mpf.make_addplot(sma, color='blue', label='SMA (20)'),
                        mpf.make_addplot(bb_upper, color='green', label='BB Upper'),
                        mpf.make_addplot(bb_lower, color='red', label='BB Lower'),
                        mpf.make_addplot(VWAP, color='purple', label='VWAP'),
                    ]
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        # Plot candlestick with additional indicators
                        mpf.plot(
                            data,
                            type='candle',
                            style='yahoo',
                            addplot=apds,
                            volume=True,  # Optional: adds volume subplot
                            savefig=tmp.name,
                            show_nontrading=False  # Similar to hiding rangeslider
                        )
                        tmpfile_path = tmp.name
                        
                        # Open with PIL if needed
                        image = Image.open(tmpfile_path)
            
                    prompt = """
                    Eres un Operador de Bolsa especializado en Análisis Técnico en una institución financiera de primer nivel.
                    Analiza el gráfico de acciones basándote en su gráfico de velas y los indicadores técnicos mostrados.
                    Proporciona una justificación detallada de tu análisis, explicando qué patrones, señales y tendencias observas.
                    Luego, basándote únicamente en el gráfico, proporciona una recomendación de las siguientes opciones:
                    'Compra Fuerte', 'Compra', 'Compra Débil', 'Mantener', 'Venta Débil', 'Venta', o 'Venta Fuerte'. responde en español en nombre de QuantStrategyAI.
                    """
                    
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content([prompt, image])
                    
                    result = response.text
            
                    return image, result
           
                fig, result = analyze_ticker(simbolo, stock_data)
            
            
    
                st.subheader(f"Analisis para {simbolo}")
                st.image(fig)
                st.write("**Justificación Detallada:**")
                st.write(result)
                
            if st.session_state.subpagina == "avanzada":
                st.subheader("Predicción Avanzada")
                st.write(f"Configurado para {simbolo} desde {fecha_inicio} hasta {fecha_fin}")
                st.write(f"Prediciendo para {simbolo}...")
                # Import the GoogleNews package to fetch the news articles    
    
                keywords= [simbolo]
                keywords.append(get_company_name(simbolo))
                threshold= -0.02 
                #Sentiment analysis, return the average score between two models vader and textblob
                
                sentiment=sentiment_analysis(keywords)
                interval="1d"
                
                #get data from yfinance
                Time_days=500
                start=datetime.now() -  timedelta(days=Time_days)
                start=start.strftime("%Y-%m-%d")
                print("Descargando datos de precios historicos....")
                Stock_data = yf.download(simbolo, start=start)
                Stock_data.columns = [col[0] for col in Stock_data.columns]
                
                #create machine learning models to predict
                X, y, y2=create_features(Stock_data,threshold)
            
                print("Creando modelo predictivo utilizando inteligencia artificial........")
                X_test1_scaled,X_test2_scaled,X_test1,X_test2,model1_hard, model2_hard,model1_soft,model2_soft=create_models(X,y,y2)
            
                # Get probability and prediction estimates 
                y_pred1 = model1_hard.predict(X_test1_scaled)
                y_pred_proba1 = model1_soft.predict_proba(X_test1_scaled)
                y_pred2 = model2_hard.predict(X_test2_scaled)
                y_pred_proba2 = model2_soft.predict_proba(X_test2_scaled)
            
                #summary
                df=pd.DataFrame()
                df["Prediccion IA"]=0
                df["Probabilidad Comprar"]=0
                df["Probabilidad Vender"]=0
                df["Prediccion Crash IA"]=0
                df["Sentiment Score"]=0
                # New row to add
                if y_pred1[-1]==1 and y_pred_proba1[-1][1]>=0.5:
                    Prediccion="Comprar"
                elif y_pred1[-1]==0 and y_pred_proba1[-1][0]>0.5:
                    Prediccion="Vender"
                else:
                    Prediccion="Neutral"
                Probabilidad=y_pred_proba1[-1]
                if y_pred2[-1]==1:
                    Prediccion_Crash="Posible movimiento pronunciado a la baja"
                elif y_pred2[-1]==0:
                    Prediccion_Crash="Movimientos dentro de margenes normales"
                Sentiment_Score=sentiment
                new_row = {'Prediccion IA': Prediccion, 'Probabilidad Comprar': y_pred_proba1[-1][1],'Probabilidad Vender': y_pred_proba1[-1][0],"Prediccion Crash IA": Prediccion_Crash,"Sentiment Score": Sentiment_Score}
                # Add row using loc
                df.loc[len(df)] = new_row
                #print predictions
            
                # Asignar las predicciones al DataFrame X_test1
                X_test1["Prediction"] = y_pred1
                X_test1["Crash"] = y_pred2
                
    
                
                # Última predicción
                last_prediction = Prediccion  # Debe ser "Comprar", "Vender" o "Neutral"
                last_price = X_test1["Close"].iloc[-1]
                last_index = X_test1.index[-1]
                
                # Define the last prediction and corresponding icon/color
                if last_prediction == "Comprar":
                    icon = "⬆️"
                    color = "green"
                elif last_prediction == "Vender":
                    icon = "⬇️"
                    color = "red"
                else:  # Neutral
                    icon = "⏸️"
                    color = "gray"
                
                # Prepare additional plots (if needed)
                apds = []
                
                # Create temporary file to save the plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    fig, ax = plt.subplots()
                    
                    # Plot only close values
                    ax.plot(X_test1.index, X_test1["Close"], color='blue', label='Close Price')
                    ax.legend()
                    
                    # Add annotation with icon
                    ax.annotate(
                        icon,
                        xy=(last_index, last_price),
                        xytext=(last_index, last_price * 1.02),  # Adjust position
                        fontsize=14,
                        color=color,
                        ha='center'
                    )
                    
                    plt.savefig(tmp.name)
                    tmpfile_path = tmp.name
                
                    # Open with PIL if needed
                    image = Image.open(tmpfile_path)
    
                
                widget_explanation=print_widget_explanation()
            
                st.subheader(f"Analisis para {simbolo}")
                st.image(image)
                st.dataframe(df)
                st.write("**A considerar:**")
                st.write(widget_explanation)
            

        st.markdown('</div>', unsafe_allow_html=True)           

def submenu_backtesting():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("Backtesting")
        
        # Create selectbox with time period options
        period_options = ["3 Meses", "6 Meses", "1 Año", "2 Años"]
        selected_period = st.selectbox("Período de tiempo", period_options)
        
        # Calculate fecha_inicio based on selected period
        days_dict = {
            "3 Meses": 90,    # Approximately 3 months
            "6 Meses": 180,   # Approximately 6 months
            "1 Año": 365,     # 1 year
            "2 Años": 730     # 2 years
        }
        
        # Set fecha_fin as current date
        fecha_fin = datetime.now()
        
        # Calculate fecha_inicio based on selected period
        days_to_subtract = days_dict[selected_period]
        fecha_inicio = fecha_fin - timedelta(days=days_to_subtract)
        # Calculate the difference in days
        diferencia_dias = (fecha_fin - fecha_inicio).days
        simbolo = st.text_input("Símbolo", "AAPL")
        simbolo = get_ticker_from_word(simbolo)
        
        indicador = st.selectbox(
            "Indicador Técnico",
            ["RSI", "MACD", "Media Móvil", "Bandas de Bollinger", "Stochastic"]
        )
        
        st.write(f"Configurado para {simbolo} con {indicador} desde {fecha_inicio} hasta {fecha_fin}")
        
        if st.button("Encontrar los Mejores Parámetros"):
            if indicador=="RSI":
                indicador='rsi'
            elif indicador=="MACD":
                indicador='macd'
            elif indicador=="Media Móvil":
                indicador='sma'
            elif indicador=="Bandas de Bollinger":  
                indicador='bollinger'
            elif indicador=="Stochastic":
                indicador='stochastic'

            st.write(f"Buscando parámetros óptimos para {simbolo} con {indicador}...")
            mejor, image= find_best_strategy_indicator(simbolo,indicador,diferencia_dias)
            parametros = mejor.loc['params']

            prompt = f"""
            eres un agente ia que crea codigo usando datos de yfinance en python para hacer backtesting,
            crea un backtesting teniendo en cuenta el simbolo, el indicador tecnico y los parametros (parametros).
            Importante: no des a entender que eres una ia, como titulo, escribe solamente: "codigo en python para backtesting de {simbolo}
            y indicador {indicador} y luego el codigo, Bajo este codigo agrega codigo en metatrader 4 para de {simbolo}
            y indicador {indicador} y luego el codigo "
            """
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content([prompt])
            
            result = response.text

            
            st.image(image)
            st.dataframe(pd.DataFrame(mejor))
            st.write("**Codigo en python**")
            st.write(result)

            
            
        st.markdown('</div>', unsafe_allow_html=True)


def submenu_ia_macro_analisis():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("IA Macro Análisis")
               
        st.write("Reporte Macroeconomico con Inteligencia Artificial ")
        if st.button("Reporte IA Macroeconomia"):
            st.write(f"Procesando análisis macroeconómico con inteligencia artificial.")

            # Obtener los datos
            economic_data=get_economic_data()
            economic_news=get_economic_news()
            price_changes=get_price_changes()
                        
            # Convertir DataFrames a string para el prompt
            economic_data_st = economic_data.to_string()
            economic_news_st  = str(economic_news)
            price_changes_st  = price_changes.to_string()
            
            # Crear el prompt
            prompt = f"""
            Eres un analista economico especializado en analisis macro aplicado a mercados financieros. Con las tablas proporcionadas, crea un análisis bastante detallado de la situación 
            macroeconimca actual (puedes agregar tambien otros datos actualizados  si es necesario). Por último, basándote únicamente en la información proporcionada, 
            proporciona una recomendación sobre el estado de la economia. Recalca que deben tomarse más datos en cuenta para una decisión final.
            
            Datos macro:
            {economic_data_st}
            
            Noticias economicas:
            {economic_news_st}
            
            Variacion de precios:
            {price_changes_st}
            Por ultimo y muy importante, no des a entender que eres una ia, solo entrega los solicitado
            """
            

            response = gen_model.generate_content(prompt)
            result = response.text
            st.write("**Analisis IA Macro**")
            st.write("Datos Macro")
            st.dataframe(economic_data)
            st.write("Precios")
            st.dataframe(price_changes)
            st.write(result)
        st.markdown('</div>', unsafe_allow_html=True)

def submenu_ia_fundamental_analisis():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("Analisis Fundamental con IA")
        
        simbolo = st.text_input("Símbolo", "GOOGL")
        simbolo = get_ticker_from_word(simbolo)
        
        temporalidad = st.selectbox(
            "Temporalidad",
            ["Trimestral", "Anual"]
        )

        if st.button("Analizar"):
            if temporalidad=="Anual":
                temp="annual"
            elif temporalidad=="Trimestral":
                temp="quarterly"
            st.write("Analizando Estado de resultados, Balance general y Flujo de caja")
            # Obtener datos financieros (usando la función anterior)
            def get_financial_statements(ticker, period):
                stock = yf.Ticker(ticker)
                if period not in ["annual", "quarterly"]:
                    raise ValueError("Period debe ser 'annual' o 'quarterly'")
                
                if period == "annual":
                    income_statement = stock.financials
                    balance_sheet = stock.balance_sheet
                    cash_flow = stock.cashflow
                else:
                    income_statement = stock.quarterly_financials
                    balance_sheet = stock.quarterly_balance_sheet
                    cash_flow = stock.quarterly_cashflow
                
                return (income_statement.dropna(how='all'), 
                        balance_sheet.dropna(how='all'), 
                        cash_flow.dropna(how='all'))
            
            # Obtener los datos
            income_stmt, balance, cash = get_financial_statements(simbolo, temp)
            
            # Convertir DataFrames a string para el prompt
            income_str = income_stmt.to_string()
            balance_str = balance.to_string()
            cash_str = cash.to_string()
            
            # Crear el prompt
            prompt = f"""
            Eres un analista financiero de la bolsa especializado en análisis fundamental. Con las tablas proporcionadas, crea un análisis bastante detallado de la situación 
            actual de la empresa {simbolo} en base a los datos financieros: Estado de resultados, Balance general y Flujo de caja. Muestra ratios financieros relevantes y, 
            por último, basándote únicamente en la información proporcionada, proporciona una recomendación de las siguientes opciones: 
            'Compra Fuerte', 'Compra', 'Compra Débil', 'Venta Débil', 'Venta', o 'Venta Fuerte'. Recalca que deben tomarse más datos en cuenta para una decisión final.
            
            Datos financieros:
            Estado de Resultados:
            {income_str}
            
            Balance General:
            {balance_str}
            
            Flujo de Caja:
            {cash_str}
            Por ultimo y muy importante, no des a entender que eres una ia, solo entrega los solicitado
            """
            

            response = gen_model.generate_content(prompt)
            result = response.text
            st.write("**Analisis fundamental**")
            st.write(result)
  
        st.markdown('</div>', unsafe_allow_html=True)

def submenu_portafolio_minima_varianza():
    with st.container():
        st.markdown('</div>', unsafe_allow_html=True)
        st.title("Portafolio Mínima Varianza")
        
        # Create selectbox with time period options
        period_options = ["3 Meses", "6 Meses", "1 Año", "2 Años"]
        selected_period = st.selectbox("Período de tiempo", period_options)
        
        # Calculate fecha_inicio based on selected period
        days_dict = {
            "3 Meses": 90,    # Approximately 3 months
            "6 Meses": 180,   # Approximately 6 months
            "1 Año": 365,     # 1 year
            "2 Años": 730     # 2 years
        }
        
        # Set fecha_fin as current date
        fecha_fin = datetime.now()
        
        # Calculate fecha_inicio based on selected period
        days_to_subtract = days_dict[selected_period]
        fecha_inicio = fecha_fin - timedelta(days=days_to_subtract)

                                      
        simbolos = st.text_input("Símbolos (separados por coma)", "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, WMT, DIS")
        inversion = st.text_input("Monto invertido USD", "1000")
        
        st.write(f"Optimizando portafolio para {simbolos} desde {fecha_inicio} hasta {fecha_fin}")
        if st.button("Optimizar Portafolio"):
            st.write(f"Calculando portafolio de mínima varianza para {simbolos}...")
            # Split the input string into a list of symbols
            simbolos = [s.strip() for s in simbolos.split(",")]
            data = yf.download(simbolos, start=fecha_inicio, end=fecha_fin)['Close']
            # Calcular retornos diarios
            returns = data.pct_change().dropna()
            
            # Calcular matriz de covarianza y retornos esperados (anualizados)
            cov_matrix = returns.cov() * 252  # Anualizamos la covarianza
            expected_returns = returns.mean() * 252  # Anualizamos los retornos
            
            # Función para calcular la varianza del portafolio
            def portfolio_variance(weights, cov_matrix):
                return weights.T @ cov_matrix @ weights
            
            # Restricciones: los pesos suman 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Límites: pesos entre 0 y 1 (sin ventas en corto)
            bounds = tuple((0, 1) for _ in range(len(simbolos)))
            
            # Inicialización de pesos (distribución uniforme)
            init_weights = np.array([1/len(simbolos)] * len(simbolos))
            
            # Optimización para mínima varianza
            opt_result = minimize(
                fun=portfolio_variance, 
                x0=init_weights, 
                args=(cov_matrix,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            # Pesos óptimos del portafolio de mínima varianza
            optimal_weights = opt_result.x
            
            # Filtrar activos con pesos >= 1% (0.01)
            significant_weights = optimal_weights[optimal_weights >= 0.01]
            significant_tickers = [crypto for crypto, weight in zip(simbolos, optimal_weights) if weight >= 0.01]
            
            # Normalizar los pesos significativos para que sumen 1
            normalized_weights = significant_weights / np.sum(significant_weights)
            
            # Pesos del portafolio equally weighted (1/n para cada activo)
            equal_weights = np.array([1/len(simbolos)] * len(simbolos))
            
            # Calcular retornos diarios de ambos portafolios
            min_var_returns_adjusted = returns[significant_tickers] @ normalized_weights  # Mínima varianza ajustado
            equal_weighted_returns = returns @ equal_weights  # Equally weighted
            
            # Calcular la evolución acumulada del valor del portafolio (valor inicial = 1)
            min_var_cumulative_adjusted = (1 + min_var_returns_adjusted).cumprod()
            equal_weighted_cumulative = (1 + equal_weighted_returns).cumprod()
            
            # Pedir al usuario el valor total a invertir
            total_investment = float(inversion)
            
            # Calcular el monto a invertir en cada acción para el portafolio de mínima varianza ajustado
            investment_per_asset = normalized_weights * total_investment
            
            # Crear tabla con los resultados para el portafolio de mínima varianza ajustado
            results_table = pd.DataFrame({
                'Activo': significant_tickers,
                'Peso': normalized_weights,
                'Monto a Invertir ($)': investment_per_asset
            })
            results_table['Peso (%)'] = results_table['Peso'] * 100

                        # Calcular retorno y volatilidad del portafolio ajustado
            portfolio_return = np.sum(expected_returns[significant_tickers] * normalized_weights)
            portfolio_var = portfolio_variance(normalized_weights, cov_matrix.loc[significant_tickers, significant_tickers])
            portfolio_std = np.sqrt(portfolio_var)

            # Calcular retorno y volatilidad del portafolio equally weighted
            equal_weighted_return = np.sum(expected_returns * equal_weights)
            equal_weighted_var = portfolio_variance(equal_weights, cov_matrix)
            equal_weighted_std = np.sqrt(equal_weighted_var)

            
            st.write("**Portafolio de mínima varianza ajustado (pesos >= 1%):**")
            st.dataframe(results_table)

            # Prepare the bar plot
            fig, ax = plt.subplots()
            ax.bar(significant_tickers, normalized_weights, width=0.35, color='blue', label='Mínima Varianza (ajustado)')
            ax.set_title('Pesos en el Portafolio Ajustado (pesos >= 1%)')
            ax.set_ylabel('Pesos')
            ax.set_xlabel('Activos')
            ax.legend()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                tmpfile_path = tmp.name
                image = Image.open(tmpfile_path)
            
            st.image(image)
            plt.close(fig)
            
            # Display metrics
            st.write(f"\nRetorno esperado anualizado (Mínima Varianza Ajustado): {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
            st.write(f"Volatilidad anualizada (Mínima Varianza Ajustado): {portfolio_std:.4f} ({portfolio_std*100:.2f}%)")
            st.write(f"\nRetorno esperado anualizado (Equally Weighted): {equal_weighted_return:.4f} ({equal_weighted_return*100:.2f}%)")
            st.write(f"Volatilidad anualizada (Equally Weighted): {equal_weighted_std:.4f} ({equal_weighted_std*100:.2f}%)")




            # Ensure all arrays align with data.index
            min_var_cumulative_adjusted = min_var_cumulative_adjusted.reindex(data.index, fill_value=0)  # Fill missing with 0 or interpolate
            equal_weighted_cumulative = equal_weighted_cumulative.reindex(data.index, fill_value=0)
            
            # Prepare the plot
            fig, ax = plt.subplots()
            ax.plot(data.index, min_var_cumulative_adjusted, color='blue', label='Portafolio de Mínima Varianza (ajustado)')
            ax.plot(data.index, equal_weighted_cumulative, color='orange', label='Portafolio Equally Weighted')
            ax.set_title('Evolución del Valor del Portafolio')
            ax.set_ylabel('Valor del Portafolio (normalizado)')
            ax.set_xlabel('Fecha')
            ax.legend()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                tmpfile_path2 = tmp.name
                image2 = Image.open(tmpfile_path2)
            
            st.image(image2)
            plt.close(fig)


        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="sidebar-title">
            <div class="logo">Quanthika</div>
            
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Inicio"):
        st.session_state.pagina = "principal"
        st.session_state.subpagina = None
    
    st.markdown('<div class="section-title">Herramientas IA</div>', unsafe_allow_html=True)
    if st.button("Predictor IA"):
        st.session_state.pagina = "predictor"
        st.session_state.subpagina = "basica"
    if st.button("IA Macro Análisis"):
        st.session_state.pagina = "macro"
        st.session_state.subpagina = None
    if st.button("Analisis Fundamental con IA"):
        st.session_state.pagina = "analisis"
        st.session_state.subpagina = None
    
    st.markdown('<div class="section-title">Crear Estrategias/Optimizar Portafolio</div>', unsafe_allow_html=True)
    if st.button("Backtesting"):
        st.session_state.pagina = "backtesting"
        st.session_state.subpagina = None
    if st.button("Portafolio Mínima Varianza"):
        st.session_state.pagina = "portafolio"
        st.session_state.subpagina = None
    
    st.markdown("<hr>", unsafe_allow_html=True)

# Mostrar contenido
if st.session_state.pagina == "principal":
    pagina_principal()
elif st.session_state.pagina == "predictor":
    submenu_predictor_ia()
elif st.session_state.pagina == "backtesting":
    submenu_backtesting()
elif st.session_state.pagina == "sentimiento":
    submenu_analisis_sentimiento()
elif st.session_state.pagina == "macro":
    submenu_ia_macro_analisis()
elif st.session_state.pagina == "portafolio":
    submenu_portafolio_minima_varianza()
elif st.session_state.pagina == "analisis":
    submenu_ia_fundamental_analisis()


# In[ ]:


# streamlit run C:\Users\andre\Downloads\streamlit_prueba.py

