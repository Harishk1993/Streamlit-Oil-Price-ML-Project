import streamlit as st
import pandas as pd
import matplotlib as plt
from datetime import date
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import numpy as np
#import pandas_datareader as data
from multiapp import MultiApp
from apps import page_1, page_2



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header(':green[Dashboard] `version 1`')


app = MultiApp()

# Add all your application here
st.sidebar.subheader(':orange[Select a Tab]')
app.add_app("About Dataset", page_1.app)
app.add_app("Time Series Forecasting", page_2.app)

app.run()


