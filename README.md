# Retail Demand Forecasting (PySpark & XGBoost)

## 📌 Project Overview
Accurately predicting consumer demand is the cornerstone of retail supply chain optimization. Overstocking leads to markdown losses, while understocking leads to stockouts and lost revenue.

This project implements a highly scalable **Time-Series Demand Forecasting Pipeline**. It utilizes **Apache Spark (PySpark)** to process historical daily sales data and engineer complex temporal features (lags, rolling windows). The engineered dataset is then used to train an **XGBoost Regressor**, predicting future sales volume with high accuracy.

**Business Value:** Enables automated, data-driven inventory allocation for promotional items and baseline assortment, directly maximizing gross margin for retail commerce platforms.

## 🛠️ Tech Stack
* **Big Data Engineering:** Apache Spark (PySpark SQL, Window Functions)
* **Machine Learning:** XGBoost, scikit-learn
* **Data Manipulation:** Pandas, NumPy
* **Language:** Python 3.x

## 🏗️ Architecture & Workflow
1. **Data Ingestion:** Loads historical daily sales records using PySpark.
2. **Temporal Feature Engineering:** Utilizes PySpark `Window` functions to create time-shifted features, including 7-day and 28-day sales lags to capture weekly and monthly retail seasonality.
3. **Temporal Splitting:** Implements a strict chronological train/test split (e.g., train on 2013-2016, validate on 2017) to prevent data leakage.
4. **Gradient Boosting:** Trains an `XGBRegressor` to model complex, non-linear relationships between dates, historical lags, and future demand.
5. **Evaluation:** Measures financial impact accuracy using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## 📂 Project Structure
```text
├── data/                   # Data directory (Dataset excluded via .gitignore)
├── src/                    
│   ├── data_prep.py        # PySpark ingestion and date formatting
│   ├── feature_engineering.py # PySpark Window functions for lag features
│   ├── trainer.py          # XGBoost training and temporal validation
│   └── main.py             # Orchestration script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
