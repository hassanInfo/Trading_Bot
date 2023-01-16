import streamlit as st
import pandas as pd
import requests
import numpy as np
import altair as alt
import seaborn as sns
import datetime
import logging
import coloredlogs

from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
from trading_bot.methods import evaluate_model
from trading_bot.agent import Agent
st.set_page_config(layout="wide")

# from datetime import timedelta

def check_consecutive_datetimes(date_time_list):
    for i in range(len(date_time_list)-1):
        if date_time_list[i] - date_time_list[i+1] != datetime.timedelta(days=1):
            return False
    return True


bitcoin_link = 'https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD'
etherium_link = 'https://finance.yahoo.com/quote/ETH-USD/history?p=ETH-USD'
google_link = 'https://finance.yahoo.com/quote/GOOG/history?p=GOOG'
apple_link = 'https://finance.yahoo.com/quote/AAPL/history?p=AAPL'

response_bitcoin = requests.get(bitcoin_link, headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
response_etherium = requests.get(etherium_link, headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
response_google = requests.get(google_link, headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
response_apple = requests.get(apple_link, headers ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

data_bitcoin, data_etherium = pd.read_html(response_bitcoin.text), pd.read_html(response_etherium.text)
data_google, data_apple = pd.read_html(response_google.text), pd.read_html(response_apple.text)

data_bitcoin, data_etherium = data_bitcoin[-1], data_etherium[-1]
data_google, data_apple = data_google[-1], data_apple[-1]

data_bitcoin = data_bitcoin.drop([len(data_bitcoin)-1])
data_etherium = data_etherium.drop([len(data_etherium)-1])
data_google = data_google.drop([len(data_google)-1])
data_apple = data_apple.drop([len(data_apple)-1])

data_bitcoin, data_etherium = data_bitcoin.dropna(), data_etherium.dropna()
data_google, data_apple = data_google.dropna(), data_apple.dropna()

data_bitcoin = data_bitcoin.drop(['Open', 'High', 'Low', 'Close*', 'Volume'], axis=1)
data_etherium = data_etherium.drop(['Open', 'High', 'Low', 'Close*', 'Volume'], axis=1)
data_google = data_google.drop(['Open', 'High', 'Low', 'Close*', 'Volume'], axis=1)
data_apple = data_apple.drop(['Open', 'High', 'Low', 'Close*', 'Volume'], axis=1)

data_bitcoin = data_bitcoin.rename(columns={'Date': 'date', 'Adj Close**': 'actual'})
data_etherium = data_etherium.rename(columns={'Date': 'date', 'Adj Close**': 'actual'})
data_google = data_google.rename(columns={'Date': 'date', 'Adj Close**': 'actual'})
data_apple = data_apple.rename(columns={'Date': 'date', 'Adj Close**': 'actual'})

data_bitcoin['date'] = pd.to_datetime(data_bitcoin['date'], infer_datetime_format=True)
data_etherium['date'] = pd.to_datetime(data_etherium['date'], infer_datetime_format=True)
data_google['date'] = pd.to_datetime(data_google['date'], infer_datetime_format=True)
data_apple['date'] = pd.to_datetime(data_apple['date'], infer_datetime_format=True)

data_bitcoin = data_bitcoin[data_bitcoin["actual"].str.contains("Dividend") == False].reset_index(drop=True)
data_bitcoin.loc[-1] = [data_bitcoin['date'].iloc[0]+datetime.timedelta(days=1), float('nan')] # adding a row
data_bitcoin.index = data_bitcoin.index+1 # shifting index
data_bitcoin.sort_index(inplace=True)
data_bitcoin = data_bitcoin.iloc[::-1]
data_bitcoin = data_bitcoin.astype({'actual': float})

data_etherium = data_etherium[data_etherium["actual"].str.contains("Dividend") == False].reset_index(drop=True)
data_etherium.loc[-1] = [data_etherium['date'].iloc[0]+datetime.timedelta(days=1), float('nan')] # adding a row
data_etherium.index = data_etherium.index+1 # shifting index
data_etherium.sort_index(inplace=True)
data_etherium = data_etherium.iloc[::-1]
data_etherium = data_etherium.astype({'actual': float})

data_google = data_google[data_google["actual"].str.contains("Dividend") == False].reset_index(drop=True)
if not check_consecutive_datetimes(list(data_google['date'].iloc[:5])):
    data_google.loc[-1] = [data_google['date'].iloc[0]+datetime.timedelta(days=1), float('nan')] # adding a row
else:
    data_google.loc[-1] = [data_google['date'].iloc[0]+datetime.timedelta(days=3), float('nan')]
data_google.index = data_google.index+1 # shifting index
data_google.sort_index(inplace=True)
data_google = data_google.iloc[::-1]
data_google = data_google.astype({'actual': float})

data_apple = data_apple[data_apple["actual"].str.contains("Dividend") == False].reset_index(drop=True)
if not check_consecutive_datetimes(list(data_apple['date'].iloc[:5])):
    data_apple.loc[-1] = [data_apple['date'].iloc[0]+datetime.timedelta(days=1), float('nan')] # adding a row
else:
    data_apple.loc[-1] = [data_apple['date'].iloc[0]+datetime.timedelta(days=3), float('nan')]
data_apple.index = data_apple.index+1 # shifting index
data_apple.sort_index(inplace=True)
data_apple = data_apple.iloc[::-1]
data_apple = data_apple.astype({'actual': float})



# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0]] + [x[0] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions
    
    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)
    
    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )
    
    points = alt.Chart(df).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale),
        shape='action',
        color='action',
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1500)
    
    return chart

def apply(model_name, data, title):
    window_size = 10
    debug = True
    agent = Agent(window_size, pretrained=True, model_name=model_name)
    coloredlogs.install(level='DEBUG')
    switch_k_backend_device()
    test_data = list(data['actual'])
    initial_offset = test_data[1] - test_data[0]
    test_result, history = evaluate_model(agent, test_data, window_size, debug)
    # show_eval_result(model_name, test_result, initial_offset)
    st.write(visualize(data, history, title=title+str(format_position(test_result))))

st.markdown('<h1 align="center" style="color:#3AFF00;">Welcome to the Trading world !!!</h1>', unsafe_allow_html=True)
st.markdown('<h1 align="center" style="color:#3AFF00;">We are here to provide you the best Market order </h1>', unsafe_allow_html=True)
st.markdown('<h1 align="center"></h1>', unsafe_allow_html=True)
st.markdown('<h1 align="center"></h1>', unsafe_allow_html=True)
st.markdown('<h1 align="center"></h1>', unsafe_allow_html=True)
st.markdown('<h4 align="center">Please select a currency or a stock:</h4>', unsafe_allow_html=True)
option = st.selectbox(
    '',
    ('Bitcoin', 'Etherium', 'Google', 'Apple')
)

if option=='Bitcoin':
    apply('model_t_dqn_BTC-USD_epidodes_50_30', data_bitcoin, 'BITCOIN Currency ')
elif option=='Etherium':
    apply('100_iter_change_target/model_t_dqn_ETH-USD_epidodes_50_20', data_etherium, 'ETHERIUM Currency ')
elif option=='Google':
    apply('Model_dqn_GOOG_epidodes_50_40', data_google, 'GOOGLE Stocks ')
else:
    apply('model_dqn_AAPL_epidodes_50_50', data_apple, 'APPLE Stocks ')
