import streamlit as st
from datetime import datetime
import yfinance as yf
import prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib_inline
import mplfinance as mpf

stock='^NSEI'
start="2023-08-20"
today=datetime.today().strftime('%Y-%m-%d')

def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data

data=load_data(stock)
data=data.set_index('Date')
#print(data)

mpf.plot(data,figratio=(18,10),type='candle',tight_layout=True,volume=True,style='yahoo')