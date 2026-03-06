import streamlit as st
import pandas as pd
import joblib

# Title
st.title("🏠 House Rent Prediction App")

# Load saved model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.write("Enter the house details to predict the price")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1400)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
washrooms = st.number_input("Washrooms", min_value=1, max_value=10, value=2)

# Predict Button
if st.button("Predict Price"):

    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Washrooms': [washrooms]
    })

    # Convert categorical variables
    new_house = pd.get_dummies(new_house)

    # Align columns with training data
    new_house = new_house.reindex(columns=columns, fill_value=0)

    # Scale data
    new_scaled = scaler.transform(new_house)

    # Prediction
    predicted_price = model.predict(new_scaled)

    st.success(f"🏠 Predicted House Price: ₹ {predicted_price[0]:,.2f}")