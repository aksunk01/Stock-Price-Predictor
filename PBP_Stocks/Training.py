import pickle
from prophet import Prophet
from autots import AutoTS
from autots.datasets import load_hourly
import numpy as np
import pandas as pd

df_long = load_hourly(long=True)


#This is the historical stock data
data = pd.read_csv("AAPL.csv")

model = AutoTS(forecast_length = 5, frequency = 'infer', ensemble = 'simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col = None)
prediction = model.predict()
forecast = prediction.forecast


#This shows a visual of the stock price data
#figure = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close = data["Close"])])
#figure.update_layout(title = "Apple Stock Price Analysis", xaxis_rangeslider_visible=False)
#figure.show()

with open('model_training.pkl', 'wb') as f:
    pickle.dump(model, f)