import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="🌾 Agriculture Price Prediction",
    page_icon="🌾",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("ui_mapping_data.csv")
    df = df.dropna(subset=[
        "STATE","District Name","Market Name",
        "Commodity","Variety","Grade"
    ])

    cols = ["STATE","District Name","Market Name",
            "Commodity","Variety","Grade"]

    for c in cols:
        df[c] = df[c].astype(str)

    return df

@st.cache_resource
def load_model():
    return joblib.load("agriculture_price_model_compressed.pkl")

df = load_data()
model = load_model()

state_list = sorted(df["STATE"].unique())

district_map = (
    df.groupby("STATE")["District Name"]
      .apply(lambda x: sorted(x.unique()))
      .to_dict()
)

market_map = (
    df.groupby(["STATE","District Name"])["Market Name"]
      .apply(lambda x: sorted(x.unique()))
      .to_dict()
)

commodity_list = sorted(df["Commodity"].unique())

st.title("🌾 Agriculture Price Prediction")
st.markdown(
    "Predict agricultural crop modal prices using a Machine Learning model."
)
st.divider()

st.sidebar.header("📋 Crop Details")

state = st.sidebar.selectbox("State", state_list)

district = st.sidebar.selectbox(
    "District",
    district_map[state]
)

market = st.sidebar.selectbox(
    "Market",
    market_map[(state, district)]
)

commodity = st.sidebar.selectbox(
    "Commodity",
    commodity_list
)

variety = st.sidebar.selectbox(
    "Variety",
    sorted(df[df["Commodity"] == commodity]["Variety"].unique())
)

grade = st.sidebar.selectbox(
    "Grade",
    sorted(df["Grade"].unique())
)

year = st.sidebar.number_input("Year", 2000, 2100, 2025)
month = st.sidebar.number_input("Month", 1, 12, 1)
day = st.sidebar.number_input("Day", 1, 31, 15)

min_price = st.sidebar.number_input("Minimum Price", min_value=0.0)
max_price = st.sidebar.number_input("Maximum Price", min_value=0.0)

left, right = st.columns(2)

with left:
    st.subheader("Selected Details")
    st.write(f"**State:** {state}")
    st.write(f"**District:** {district}")
    st.write(f"**Market:** {market}")
    st.write(f"**Commodity:** {commodity}")
    st.write(f"**Variety:** {variety}")
    st.write(f"**Grade:** {grade}")

with right:
    st.subheader("Input Prices")
    st.metric("Minimum Price", f"₹ {min_price:,.2f}")
    st.metric("Maximum Price", f"₹ {max_price:,.2f}")

st.divider()

# Simple factorization (keep same logic as your original app)
state_code = df[df["STATE"] == state]["STATE"].factorize()[0][0]
district_code = df[df["District Name"] == district]["District Name"].factorize()[0][0]
market_code = df[df["Market Name"] == market]["Market Name"].factorize()[0][0]
commodity_code = df[df["Commodity"] == commodity]["Commodity"].factorize()[0][0]
variety_code = df[df["Variety"] == variety]["Variety"].factorize()[0][0]
grade_code = df[df["Grade"] == grade]["Grade"].factorize()[0][0]

if st.button("🔮 Predict Price", use_container_width=True):

    if min_price > max_price:
        st.error("Minimum Price cannot be greater than Maximum Price.")

    else:
        with st.spinner("Predicting..."):

            X = np.array([[
                state_code,
                district_code,
                market_code,
                commodity_code,
                variety_code,
                grade_code,
                year,
                month,
                day,
                min_price,
                max_price
            ]])

            prediction = model.predict(X)[0]

        st.success("Prediction Completed!")

        st.metric(
            "Predicted Modal Price",
            f"₹ {prediction:,.2f}"
        )

        st.balloons()

st.divider()
st.caption("Built with ❤️ using Python • Streamlit • Machine Learning")




