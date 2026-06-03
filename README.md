# 🌾 Agriculture Price Prediction & Analysis Project

## 📌 Project Overview

The Agriculture Price Prediction & Analysis Project is a data-driven machine learning application designed to analyze agricultural market prices and predict future crop prices based on historical market data. The project helps farmers, traders, researchers, and policymakers make informed decisions by understanding commodity price trends across different states, districts, and markets in India.

The dataset contains agricultural commodity prices collected from various markets, including minimum price, maximum price, and modal price information for different crops.

---

## 📊 Dataset Information

**Dataset Name:** Agriculture_price_dataset.csv

### Features

| Column Name   | Description                       |
| ------------- | --------------------------------- |
| STATE         | State where the market is located |
| District Name | District of the market            |
| Market Name   | Name of the agricultural market   |
| Commodity     | Crop/commodity name               |
| Variety       | Variety of the commodity          |
| Grade         | Quality grade of the commodity    |
| Min_Price     | Minimum price recorded            |
| Max_Price     | Maximum price recorded            |
| Modal_Price   | Most common selling price         |
| Price Date    | Date of price record              |

### Dataset Statistics

* Total Records: **737,392**
* Total Features: **10**
* Data Type: Mixed (Categorical + Numerical)
* Missing Values: Requires preprocessing before model training

---

## 🎯 Project Objectives

* Analyze agricultural commodity price trends.
* Perform Exploratory Data Analysis (EDA).
* Identify price variations across states and markets.
* Build machine learning models for price prediction.
* Visualize market-wise and commodity-wise insights.
* Assist farmers in making better selling decisions.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Joblib/Pickle

---

## 📈 Exploratory Data Analysis

The project performs:

* Commodity-wise price distribution analysis
* State-wise market comparison
* Trend analysis over time
* Correlation analysis
* Outlier detection
* Price fluctuation visualization

### Visualizations

* Bar Charts
* Histograms
* Line Charts
* Box Plots
* Heatmaps
* Commodity Comparison Graphs

---

## 🤖 Machine Learning Models

The following models can be used and compared:

### Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* Gradient Boosting Regressor

### Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Encoding Categorical Features
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Prediction Generation
9. Streamlit Deployment

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/agriculture-price-prediction.git
cd agriculture-price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 💻 Application Features

* Select State
* Select District
* Select Market
* Select Commodity
* Predict Market Price
* View Historical Trends
* Interactive Dashboard
* Real-Time Insights

---

## 📂 Project Structure

```text
Agriculture-Price-Prediction/
│
├── app.py
├── Agriculture_price_dataset.csv
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
└── assets/
    └── images
```

---

## 📊 Expected Outcomes

* Accurate agricultural price prediction.
* Better understanding of crop market trends.
* Improved decision-making for farmers and traders.
* Market intelligence through data analytics.

---

## 🔮 Future Enhancements

* Live Market Price Integration
* Weather Data Integration
* Crop Yield Prediction
* Multi-Language Support
* Mobile Application
* Deep Learning Models
* Price Forecasting for Upcoming Weeks/Months

---

## 👨‍💻 Author

**Rishu Gurjar**

Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

### Skills

* Python
* SQL
* Machine Learning
* Deep Learning
* Data Analysis
* Streamlit
* Power BI

---

## 📜 License

This project is developed for educational and research purposes. Feel free to use and modify it with proper attribution.

⭐ If you found this project useful, consider giving it a star on GitHub.
