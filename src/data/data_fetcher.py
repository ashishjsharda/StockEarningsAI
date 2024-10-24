import yfinance as yf
import pandas as pd
import streamlit as st



def fetch_data(stock_symbol, start_date, end_date):
    """
    Fetch historical stock price data and earnings reports for the selected company using Yahoo Finance API.
    """
    # Download stock price data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # Get earnings history
    stock = yf.Ticker(stock_symbol)
    earnings = stock.earnings_history

    # Check if earnings data exists and print to see structure
    if earnings:
        earnings_data = pd.DataFrame(earnings)
        st.write("Earnings Data Preview:")
        st.write(earnings_data.head())  # Debugging: Print earnings data to inspect the structure
    else:
        st.write("No earnings data available for this stock.")
        return pd.DataFrame()  # Return an empty dataframe if no earnings data

    # Check if 'Date' column exists in earnings data
    if 'Date' not in earnings_data.columns:
        st.error("The 'Date' column is missing from the earnings data.")
        return pd.DataFrame()

    earnings_data['Date'] = pd.to_datetime(earnings_data['Date'])

    # Merge earnings data with stock price data
    merged_data = pd.merge(stock_data, earnings_data, left_index=True, right_on='Date', how='inner')

    return merged_data
