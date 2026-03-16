from pyspark.sql.window import Window
from pyspark.sql.functions import col, dayofweek, month, year, lag, avg

def engineer_time_series_features(df):
    print("Extracting temporal features (Day, Month, Year)...")
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))
    df = df.withColumn("year", year(col("date")))
    
    print("Calculating rolling window features (Lags & Moving Averages)...")
    # Define a time-series window ordered by date
    window_spec = Window.partitionBy("store", "item").orderBy("date")
    
    # Feature: Sales from exactly 7 days ago (captures weekly seasonality)
    df = df.withColumn("sales_lag_7", lag("sales", 7).over(window_spec))
    
    # Feature: Sales from exactly 28 days ago (captures monthly seasonality)
    df = df.withColumn("sales_lag_28", lag("sales", 28).over(window_spec))
    
    # Drop rows with nulls caused by the lag shift
    clean_df = df.dropna()
    
    print("Converting engineered PySpark DataFrame to Pandas...")
    return clean_df.toPandas()
