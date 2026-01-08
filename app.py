import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Agriculture Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# ================= LOAD DATA & MODEL =================
# df = pd.read_csv("Agriculture_price_dataset.csv")
df = pd.read_csv("ui_mapping_data.csv")

model = joblib.load("agriculture_price_model_compressed.pkl")
df = pd.read_csv("ui_mapping_data.csv")

# IMPORTANT: NaN remove + convert to string
df = df.dropna(subset=['STATE', 'District Name', 'Market Name',
                        'Commodity', 'Variety', 'Grade'])

df['STATE'] = df['STATE'].astype(str)
df['District Name'] = df['District Name'].astype(str)
df['Market Name'] = df['Market Name'].astype(str)
df['Commodity'] = df['Commodity'].astype(str)
df['Variety'] = df['Variety'].astype(str)
df['Grade'] = df['Grade'].astype(str)
state_list = sorted(df['STATE'].unique().tolist())

district_map = (
    df.groupby('STATE')['District Name']
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)

market_map = (
    df.groupby(['STATE', 'District Name'])['Market Name']
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)

commodity_list = sorted(df['Commodity'].unique().tolist())

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>🌾 Agriculture Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Predict crop modal price using Machine Learning</p>",
    unsafe_allow_html=True
)
st.divider()

# ================= PREPARE DROPDOWN DATA =================
state_list = sorted(df['STATE'].unique())
district_map = df.groupby('STATE')['District Name'].unique().to_dict()
market_map = df.groupby(['STATE','District Name'])['Market Name'].unique().to_dict()
commodity_list = sorted(df['Commodity'].unique())

# ================= SIDEBAR INPUTS =================
st.sidebar.header("📊 Enter Crop Details")

state = st.sidebar.selectbox("Select State", state_list)

district = st.sidebar.selectbox(
    "Select District",
    district_map[state]
)

market = st.sidebar.selectbox(
    "Select Market",
    market_map[(state, district)]
)

commodity = st.sidebar.selectbox(
    "Select Crop (Commodity)",
    commodity_list
)

variety = st.sidebar.selectbox(
    "Select Variety",
    df[df['Commodity'] == commodity]['Variety'].unique()
)

grade = st.sidebar.selectbox(
    "Select Grade",
    df['Grade'].unique()
)

year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=1)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)

min_price = st.sidebar.number_input("Minimum Price", min_value=0)
max_price = st.sidebar.number_input("Maximum Price", min_value=0)

# ================= ENCODE INPUTS =================
state_code = df[df['STATE'] == state]['STATE'].factorize()[0][0]
district_code = df[df['District Name'] == district]['District Name'].factorize()[0][0]
market_code = df[df['Market Name'] == market]['Market Name'].factorize()[0][0]
commodity_code = df[df['Commodity'] == commodity]['Commodity'].factorize()[0][0]
variety_code = df[df['Variety'] == variety]['Variety'].factorize()[0][0]
grade_code = df[df['Grade'] == grade]['Grade'].factorize()[0][0]

# ================= PREDICTION =================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("🔮 Predict Price", use_container_width=True):

        if min_price > max_price:
            st.error("❌ Minimum Price cannot be greater than Maximum Price")
        else:
            input_data = np.array([[
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

            prediction = model.predict(input_data)[0]

            st.success(f"💰 Predicted Modal Price: ₹ {prediction:.2f}")

# ================= FOOTER =================
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)



