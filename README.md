# 🌾 Agriculture Price Prediction System

## 🔗 Live Demo

**Try the Application Here:**
https://hackthonbhopal-z7pghruiu4puhs63q6fstd.streamlit.app/

---

# 📌 Project Overview

The Agriculture Price Prediction System is a Machine Learning-powered web application developed to predict the **Modal Price** of agricultural commodities using historical market data collected from agricultural markets across India.

The system enables users to select a state, district, market, commodity, variety, and grade while providing minimum and maximum market prices. Based on these inputs, a trained machine learning model predicts the expected modal price of the selected crop.

This project helps farmers, traders, agricultural researchers, and policymakers make better pricing decisions through data-driven insights and predictive analytics.

---

# 🎯 Problem Statement

Agricultural commodity prices change frequently due to factors such as:

* Market demand and supply
* Seasonal variations
* Regional market conditions
* Crop quality and variety
* Economic fluctuations

Because of these uncertainties, farmers often struggle to estimate the right selling price for their crops.

This project addresses that challenge by leveraging machine learning algorithms to predict crop prices based on historical agricultural market data.

---

# ✨ Key Features

✅ Predict Agricultural Commodity Prices

✅ Interactive Streamlit Web Application

✅ State-wise Market Selection

✅ District and Market Filtering

✅ Commodity-wise Price Prediction

✅ Variety and Grade Selection

✅ Real-Time Prediction Results

✅ User-Friendly Interface

✅ Input Validation System

✅ Fast and Lightweight Deployment

---

# 📊 Dataset Information

The project utilizes a large-scale Agricultural Market Dataset containing historical crop prices from multiple markets across India.

### Dataset Features

| Feature       | Description               |
| ------------- | ------------------------- |
| STATE         | State Name                |
| District Name | District Name             |
| Market Name   | Market Name               |
| Commodity     | Crop Name                 |
| Variety       | Crop Variety              |
| Grade         | Crop Quality Grade        |
| Min Price     | Minimum Market Price      |
| Max Price     | Maximum Market Price      |
| Modal Price   | Most Common Selling Price |
| Price Date    | Date of Record            |

### Dataset Statistics

* Total Records: **737,000+**
* Multiple States
* Multiple Markets
* Multiple Commodities
* Historical Agricultural Price Data

---

# 🛠 Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

## Machine Learning

* Regression-Based Prediction Model

## Deployment

* GitHub
* Streamlit Cloud

---

# 📈 Exploratory Data Analysis (EDA)

Before model training, extensive data analysis was performed to understand crop price behavior.

### Analysis Included

* Commodity-wise Price Distribution
* State-wise Price Comparison
* Market Trend Analysis
* Price Variation Analysis
* Outlier Detection
* Data Cleaning and Transformation

### Visualizations Used

* Bar Charts
* Histograms
* Line Charts
* Box Plots
* Correlation Analysis

---

# ⚙️ Machine Learning Workflow

## 1. Data Collection

Collected agricultural market price records from multiple Indian markets.

## 2. Data Cleaning

* Missing Value Handling
* Duplicate Removal
* Data Formatting

## 3. Feature Engineering

Converted categorical features into machine-readable numerical values.

Encoded Features:

* State
* District
* Market
* Commodity
* Variety
* Grade

## 4. Model Training

The machine learning model was trained using historical agricultural pricing data.

## 5. Model Evaluation

Model performance was validated on unseen data before deployment.

## 6. Deployment

The trained model was deployed using Streamlit Cloud for public access.

---

# 🚀 Application Workflow

### Step 1

Select State

### Step 2

Select District

### Step 3

Select Market

### Step 4

Select Commodity

### Step 5

Select Variety

### Step 6

Select Grade

### Step 7

Enter:

* Year
* Month
* Day
* Minimum Price
* Maximum Price

### Step 8

Click **Predict Price**

### Step 9

View Predicted Modal Price

---

# 📥 Model Input Features

| Input Feature |
| ------------- |
| State         |
| District      |
| Market        |
| Commodity     |
| Variety       |
| Grade         |
| Year          |
| Month         |
| Day           |
| Minimum Price |
| Maximum Price |

---

# 📤 Model Output

The model predicts:

### 💰 Modal Price (₹)

The modal price represents the most commonly occurring selling price of a commodity in a market.

---

# 📁 Project Structure

```text
Agriculture-Price-Prediction/
│
├── app.py
├── Agriculture_price_dataset.csv
├── agriculture_price_model_compressed.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
└── assets/
    └── screenshots/
```

---

# 🌟 Application Features

### Dynamic Dropdown Selection

The application automatically updates:

* Districts based on selected State
* Markets based on selected District
* Varieties based on selected Commodity

### Prediction Validation

The system prevents invalid inputs such as:

```python
if min_price > max_price:
    st.error("Minimum Price cannot be greater than Maximum Price")
```

### Real-Time Prediction

Users receive instant crop price predictions without reloading the application.

---

# 🔮 Future Enhancements

### Planned Improvements

* Live Market Data Integration
* Weather-Based Price Prediction
* Crop Recommendation System
* Price Trend Forecasting
* Deep Learning Models
* Mobile Application
* Multi-Language Support
* Farmer Advisory Dashboard
* Government API Integration

---

# 📸 Screenshots

## Home Page

(Add Application Screenshot Here)

## Prediction Page

(Add Prediction Screenshot Here)

## Result Page

(Add Prediction Result Screenshot Here)

---

# 🎯 Project Impact

This system can help:

### Farmers

* Better crop pricing decisions
* Improved profit planning

### Traders

* Market trend analysis
* Purchase planning

### Researchers

* Agricultural data analysis
* Market behavior studies

### Policymakers

* Agricultural market monitoring
* Price trend evaluation

---

# 👨‍💻 Author

## Rishu Gurjar

Aspiring Data Science | Machine Learning & Deep Learning Enthusiast | Python Developer

### Technical Skills

* Python
* SQL
* Machine Learning
* Deep Learning
* Data Analysis
* Streamlit
* Power BI
* Git & GitHub
* postgresql
  

### Connect With Me

LinkedIn: https://www.linkedin.com/in/rishu-gurjar-58072a333

GitHub:   https://github.com/Rishu6262

---

# 📜 License

This project is developed for educational, research, and learning purposes.

You are free to use, modify, and improve the project with proper attribution.

---

# ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐ on GitHub**.

Your support motivates future development and helps others discover the project.
