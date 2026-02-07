import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Retail Sales Prediction",
    layout="centered"
)

st.title("📈 Retail Sales Prediction App")

# Load trained objects (make sure files exist in same folder)
model = joblib.load("gradient_boosting_model.joblib")
scaler = joblib.load("scaler.joblib")
imputer = joblib.load("imputer.joblib")

st.subheader("Enter Business Details")

# User Inputs (MATCH DATASET COLUMNS)
advertising_spend = st.number_input(
    "Advertising Spend", min_value=0, value=5000
)

discount_percentage = st.slider(
    "Discount Percentage (%)", 0, 100, 10
)

footfall = st.number_input(
    "Footfall", min_value=0, value=200
)

season_index = st.slider(
    "Season Index", 0.0, 5.0, 1.0
)

previous_month_sales = st.number_input(
    "Previous Month Sales", min_value=0, value=10000
)

if st.button("Predict Sales"):

    # Create input DataFrame (SAME order + names)
    input_df = pd.DataFrame(
        [[
            advertising_spend,
            discount_percentage,
            footfall,
            season_index,
            previous_month_sales
        ]],
        columns=[
            "Advertising_Spend",
            "Discount_Percentage",
            "Footfall",
            "Season_Index",
            "Previous_Month_Sales"
        ]
    )

    # Preprocessing
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"💰 Predicted Sales: ₹ {prediction[0]:,.2f}")
