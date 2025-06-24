import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import argparse
import numpy as np

def create_time_series_features(df, target_col, num_lags, window_size):
    """
    (Optimized to avoid fragmentation)
    Create time-series features from the input data.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'timestamp' column.
        target_col (str): The name of the column to forecast.
        num_lags (int): The number of past time steps to use as features.
        window_size (int): The size of the rolling window for statistical features.

    Returns:
        pd.DataFrame: A DataFrame with the new time-series features.
    """
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Create a list to hold all new feature columns (as Series)
    features_to_add = []

    # --- Lag Features ---
    print(f"  -> Creating {num_lags} lag features...")
    for lag in range(1, num_lags + 1):
        lag_series = df_sorted[target_col].shift(lag)
        lag_series.name = f'{target_col}_lag_{lag}'
        features_to_add.append(lag_series)

    # --- Rolling Window Features ---
    print(f"  -> Creating rolling window features with size {window_size}...")
    roll_mean = df_sorted[target_col].rolling(window=window_size, min_periods=1).mean()
    roll_mean.name = f'{target_col}_roll_mean_{window_size}'
    features_to_add.append(roll_mean)

    roll_std = df_sorted[target_col].rolling(window=window_size, min_periods=1).std()
    roll_std.name = f'{target_col}_roll_std_{window_size}'
    features_to_add.append(roll_std)

    # --- Time-based Features ---
    print("  -> Creating time-based features (hour, day_of_week)...")
    dt_index = pd.to_datetime(df_sorted['timestamp'], unit='s')

    hour_series = dt_index.dt.hour
    hour_series.name = 'hour'
    features_to_add.append(hour_series)

    day_of_week_series = dt_index.dt.dayofweek
    day_of_week_series.name = 'day_of_week'
    features_to_add.append(day_of_week_series)

    # Concatenate all new feature columns to the original DataFrame at once
    df_featured = pd.concat([df_sorted] + features_to_add, axis=1)

    # Fill any initial NaN values created by shifts/rolling windows
    df_featured.fillna(0, inplace=True)

    return df_featured

def main(args):
    """
    Main function to load data, create features, train a model, and save it.
    """
    print(f"Loading data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found in the CSV file.")
        return

    print("Starting feature engineering...")
    df_featured = create_time_series_features(df, args.target_col, args.lags, args.window_size)

    print("Feature engineering complete. Final features:")
    features = [col for col in df_featured.columns if col not in ['timestamp', 'label', args.target_col]]
    print(f"  - Using features: {features}")

    X = df_featured[features]
    y = df_featured[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"\nTraining RandomForestRegressor model to predict '{args.target_col}'...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=10)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model evaluation (MSE on test set): {mse:.2f}")

    joblib.dump(model, args.output_model)
    print(f"\n✅ Model trained and saved to {args.output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a time-series forecasting model for network traffic.")
    parser.add_argument("--input-csv", required=True, help="Input CSV file with generated metrics.")
    parser.add_argument("--output-model", required=True, help="Path to save the output model file (e.g., model.pkl).")
    parser.add_argument("--target-col", type=str, default="rx_packets", help="The column name to forecast (e.g., 'rx_packets', 'latency').")
    parser.add_argument("--lags", type=int, default=5, help="Number of lag features to create.")
    parser.add_argument("--window-size", type=int, default=10, help="Size of the rolling window for statistical features.")

    args = parser.parse_args()
    main(args)

