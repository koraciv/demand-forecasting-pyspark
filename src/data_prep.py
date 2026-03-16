from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

def load_and_clean_data(spark, file_path):
    print("Loading historical sales data into PySpark...")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Ensure the date column is an actual DateType, not a string
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    
    # For this prototype, we will forecast for a single store and item to demonstrate the logic
    # In production, this would be partitioned and run across a cluster for all items
    print("Filtering data for Store 1, Item 1 (Prototype Scope)...")
    prototype_df = df.filter((col("store") == 1) & (col("item") == 1))
    
    return prototype_df
