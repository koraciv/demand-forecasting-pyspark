import os
from pyspark.sql import SparkSession
from data_prep import load_and_clean_data
from feature_engineering import engineer_time_series_features
from trainer import train_demand_model

def run_pipeline():
    data_path = "../data/train.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Please place Kaggle demand data at {data_path}")
        return

    spark = SparkSession.builder \
        .appName("RetailDemandForecasting") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # 1. Ingest
        raw_df = load_and_clean_data(spark, data_path)
        
        # 2. Engineer Features
        featured_pandas_df = engineer_time_series_features(raw_df)
        
        # 3. Train & Evaluate
        model = train_demand_model(featured_pandas_df)
        
        print("\nDemand Forecasting Pipeline executed successfully.")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    run_pipeline()
