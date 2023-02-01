import streamlit as st
import pandas as pd
import matplotlib as plt
from datetime import date
from datetime import datetime
from prophet import Prophet
import plotly.express as px
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.graph_objs as go
import numpy as np
#import pandas_datareader as data

import warnings
warnings.filterwarnings('ignore')


def app():

    df = pd.read_excel('BrentOil.xlsx', parse_dates=['Date'])
    data = df.copy()
    data['year'] = data['Date'].dt.year

    st.title(":green[Brent Oil Price Prediction]")

    # Forecasting 
    df_train = df[['Date', 'Price']]
    df_train = df_train.rename(columns={"Date":"ds", "Price": "y"})
    df_train['ds'] = pd.to_datetime(df_train['ds']).dt.date

    st.sidebar.subheader(':orange[Brent Oil Price Prediction]')
    n_years = st.sidebar.selectbox('Years of data you want to predict', ('1','2','3','4')) 
   

    if n_years == "1":
       period = 1 * 365
    elif n_years == "2":
       period = 2 * 365
    elif n_years == "3":
       period = 3 * 365
    elif n_years == "4":
       period = 4 * 365
    else:
      period = 0

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future)

    if period >= 365:
        st.subheader(':blue[Raw data]')
        st.write(df.head(10))

        st.markdown('### :blue[Brent Oil Prices]')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = data['Date'], y = data['Price']))
        fig.layout.update(title_text="Time Series Data",height = 600, xaxis_rangeslider_visible = True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(':blue[Forecast dataframe]')
        st.write(forecast.head())

        st.subheader(':blue[Forecast Prices]')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        vis_df = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']].join(df_train.set_index('ds'))
        st.subheader(":blue[Predicted Values]")
        st.write(vis_df.head())
        st.markdown('##### yhat - Predicted Price')
        st.markdown('##### y_lower - Lower Range of Predicted Price')
        st.markdown('##### y_upper - Higher Range of Predicted Price')
        st.markdown('##### y - Actual Price')

        def convert_df(df1):
         return df1.to_csv().encode('utf-8')

        csv1 = convert_df(vis_df[9036:])

        st.sidebar.subheader(':orange[Download Predicted Data File]')
        st.sidebar.download_button(
           label="Download data as CSV",
           data=csv1,
           file_name='predict_data.csv',
           mime='text/csv',
)



        st.markdown('### :blue[Predicted Range of Brent Oil Prices]')
        fig = px.line(vis_df)
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown('#### :blue[Prophet Model:]')
        y_true=df_train['y'].iloc[:9034]
        y_predicted=vis_df['yhat'].iloc[:9034]
        mae=mean_absolute_error(y_true, y_predicted)
        r2=r2_score(y_true, y_predicted)
  
        st.write('Prophet accuracy using mean absolute error=', mae)
     
        st.write('Prophet accuracy using r2 score=', r2)
  


   


   

