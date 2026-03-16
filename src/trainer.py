import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_demand_model(pandas_df):
    print("Executing temporal train/test split (Train: 2013-2016, Test: 2017)...")
    
    # Sort by date just to be safe
    pandas_df = pandas_df.sort_values("date")
    
    train = pandas_df[pandas_df["year"] < 2017]
    test = pandas_df[pandas_df["year"] == 2017]
    
    # Drop columns the model shouldn't train on
    drop_cols = ["date", "sales", "store", "item"]
    X_train = train.drop(columns=drop_cols)
    y_train = train["sales"]
    X_test = test.drop(columns=drop_cols)
    y_test = test["sales"]
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating forecast accuracy...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n================ FORECAST RESULTS ================")
    print(f"Mean Absolute Error (MAE): {mae:.2f} units")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} units")
    print("==================================================")
    
    return model
