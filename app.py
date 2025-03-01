import streamlit as st
import yfinance as yf
import pandas as pd

# Title
st.title("Stock Price Prediction")

# User input
stock = st.text_input("Enter stock symbol:", "AAPL")

# Button
if st.button("Predict"):
    st.write(f"Predicting stock price for: {stock}")
    
    # Fetch historical data
    data = yf.download(stock, start="2020-01-01", end="2023-01-01")
    
    if not data.empty:
        # Calculate moving average
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Display the data
        st.line_chart(data[['Close', 'MA50']])
        
        # Simple prediction logic: use the last moving average as the predicted price
        predicted_price = data['MA50'].iloc[-1]
        st.write(f"Predicted stock price: {predicted_price:.2f}")
    else:
        st.write("Invalid stock symbol or no data available.")

