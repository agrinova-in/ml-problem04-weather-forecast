import streamlit as st
import pickle
import pandas as pd

# Load model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
feature_names = model_data["feature_names"]

st.title("Rainfall Prediction App")

# Input fields
pressure = st.number_input("Pressure")
dewpoint = st.number_input("Dew Point")
humidity = st.number_input("Humidity")
cloud = st.number_input("Cloud")
sunshine = st.number_input("Sunshine")
winddirection = st.number_input("Wind Direction")
windspeed = st.number_input("Wind Speed")

if st.button("Predict"):
    input_data = pd.DataFrame([[pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]],
                              columns=feature_names)
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Rainfall Expected")
    else:
        st.error("No Rainfall")