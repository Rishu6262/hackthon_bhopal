import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agriculture Price Prediction",
    page_icon="🌾",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("agriculture_price_model_compressed.pkl")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🌾 Agriculture Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict crop modal price using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Enter Details")

state = st.sidebar.number_input("State Code", min_value=0)
district = st.sidebar.number_input("District Code", min_value=0)
market = st.sidebar.number_input("Market Code", min_value=0)
commodity = st.sidebar.number_input("Commodity Code", min_value=0)
variety = st.sidebar.number_input("Variety Code", min_value=0)
grade = st.sidebar.number_input("Grade Code", min_value=0)

year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)

min_price = st.sidebar.number_input("Minimum Price", min_value=0)
max_price = st.sidebar.number_input("Maximum Price", min_value=0)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price"):
    input_data = np.array([[
        state,
        district,
        market,
        commodity,
        variety,
        grade,
        year,
        month,
        day,
        min_price,
        max_price
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Modal Price: ₹ {prediction:.2f}")

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
