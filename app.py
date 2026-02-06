import streamlit as st
import pandas as pd
import joblib

st.title("Portaventura Attendance Forecast")

# Load model
model = joblib.load("prophet_attendance_model.pkl")

# User input
days = st.slider("Forecast days", 7, 365, 30)

if st.button("Generate Forecast"):

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    st.subheader("Forecast Plot")
    st.pyplot(model.plot(forecast,figsize=(20,10)))

    st.subheader("Trend & Seasonality")
    st.pyplot(model.plot_components(forecast))

    st.subheader("Forecast Table")
    st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days))
