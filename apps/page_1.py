import streamlit as st
import pandas as pd
import matplotlib as plt
from datetime import date
from datetime import datetime
from prophet import Prophet
import plotly.express as px
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import numpy as np
import pandas_datareader as data

import warnings
warnings.filterwarnings('ignore')

def app():
    df = pd.read_excel('BrentOil.xlsx', parse_dates=['Date'])
    data = df.copy()
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month

    st.sidebar.subheader(':orange[Data Analysis]')
    data_inf = st.sidebar.selectbox('Information about the data', ('No','Yes')) 

    def convert_df(df1):
        return df1.to_csv().encode('utf-8')

    csv = convert_df(df)
   
    
    st.sidebar.subheader(':orange[Download Raw Data File]')
    st.sidebar.download_button(
           label="Download data as CSV",
           data=csv,
           file_name='raw_data.csv',
           mime='text/csv',
)

    if data_inf == "No":
            st.title(":violet[About Dataset]")
            st.subheader("We have downloaded the Brent oil price dataset from U.S. Energy Information Administration (EIA) website.")
            st.subheader("Dataset consist of 20th May 1987 to 19th Dec 2022 (36 years) of daily oil price.")
            st.subheader("Our Dataset has 2 columns 'Date' and 'Price'.")
            st.subheader("We have used Prophet model to predict the upcoming years Oil Prices.")

    if data_inf == "Yes":
     st.title(":violet[Data Analysis]")
     st.write("### :blue[Enter the number of rows to view]")
     rows = st.number_input("", min_value=0,value=5)
     if rows > 0:
         st.dataframe(df.head(rows))
         st.subheader(":blue[Data Description]")
         st.write(df.describe().T)
    

    if data_inf == "Yes":
        st.markdown('### :blue[Variation of Price over Years]')
        st.bar_chart(
        data = data,
        x = 'year',
        y = 'Price',
        use_container_width=True)
        
        st.markdown('### :blue[Prices over Months]')
        fig = px.strip(data, x='month', y='Price')
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)
       

        st.markdown('### :blue[Line Chart of Brent Oil Prices]')
        fig = px.line(df, x='Date', y="Price")
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)


        st.markdown('### :blue[Boxplot of Price Column]')
        fig = px.box(df, x="Price")
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=500)
        st.plotly_chart(fig,use_container_width=True)

        st.markdown('### :blue[Year-Wise-Boxplot(1987 to 2022)]')
        fig = px.box(data, x="year", y="Price")
        fig.update_yaxes(tickfont=dict(size=8))
        fig.update_xaxes(tickfont=dict(size=8))
        fig.update_layout(height=600)
        st.plotly_chart(fig,use_container_width=True)



        
