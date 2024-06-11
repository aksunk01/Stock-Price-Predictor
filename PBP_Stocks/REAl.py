#This is staring to work, look at this again.
import pickle
from prophet import Prophet
from autots import AutoTS
from autots.datasets import load_hourly
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#This is to make all of the autots libraries work
df_long = load_hourly(long=True)

#This is the Stock data that the program is using
data = pd.read_csv("AAPL.csv")




#This is where the model's saved training is loaded onto this model
with open('model_training.pkl', 'rb') as f:
    model = pickle.load(f)


#This is where the prediction happens
prediction = model.predict()
forecast = prediction.forecast

#This prints out the forecasted data
print("predicted data")
print(forecast)

#This prints out some of the historical data
print("old data")
print(data.head())
