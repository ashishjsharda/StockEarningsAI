import numpy as np
import streamlit as st
from src.data.data_fetcher import fetch_data
from src.model.model import train_model, predict_stock_movement
from src.features.feature_engineering import add_features
import pandas as pd

# Streamlit UI
st.title("Stock Earnings Movement Predictor")

# Input form for stock symbol and date range
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL")
start_date = st.date_input("Start Date:", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime("2023-01-01"))

if st.button("Fetch Stock and Earnings Data"):
    # Fetch the data
    data = fetch_data(stock_symbol, start_date, end_date)

    if not data.empty:
        st.write(f"Showing stock and earnings data for {stock_symbol}:")
        st.write(data[['Date', 'Close', 'EPSActual', 'EPSEstimate', 'Returns']])
        st.line_chart(data[['Close']])
    else:
        st.write("No data found for the provided symbol and date range.")

# Allow user to input new earnings data to predict stock movement
st.subheader("Predict Stock Movement")
eps_actual = st.number_input("Enter EPS Actual:")
eps_estimate = st.number_input("Enter EPS Estimate:")

if st.button("Predict Stock Movement"):
    X, y = add_features(data)
    model = train_model(X, y)

    # Predict stock movement
    new_data = np.array([[eps_actual - eps_estimate, eps_actual, eps_estimate]])
    prediction = predict_stock_movement(model, new_data)

    if prediction == 1:
        st.write("Prediction: The stock is likely to move **UP**.")
    else:
        st.write("Prediction: The stock is likely to move **DOWN**.")
