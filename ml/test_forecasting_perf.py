import pandas as pd
import joblib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_time_series_features(df, target_col, num_lags, window_size):
    """
    Created time-series features from the input data. This function must be
    identical to the one used during training to ensure the model receives
    the features it expects.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'timestamp' column.
        target_col (str): The name of the column to forecast.
        num_lags (int): The number of past time steps to use as features.
        window_size (int): The size of the rolling window for statistical features.

    Returns:
        pd.DataFrame: A DataFrame with the new time-series features.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values('timestamp').reset_index(drop=True)

    # --- Lag Features ---
    for lag in range(1, num_lags + 1):
        df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)

    # --- Rolling Window Features ---
    df_copy[f'{target_col}_roll_mean_{window_size}'] = df_copy[target_col].rolling(window=window_size, min_periods=1).mean()
    df_copy[f'{target_col}_roll_std_{window_size}'] = df_copy[target_col].rolling(window=window_size, min_periods=1).std()

    # --- Time-based Features ---
    df_copy['hour'] = pd.to_datetime(df_copy['timestamp'], unit='s').dt.hour
    df_copy['day_of_week'] = pd.to_datetime(df_copy['timestamp'], unit='s').dt.dayofweek

    # Fill any initial NaN values created by shifts/rolling windows
    df_copy.fillna(0, inplace=True)

    return df_copy

def main(args):
    """
    Main function to load a trained model and test its forecasting performance.
    """
    print(f"Loading model from {args.model_path}...")
    try:
        model = joblib.load(args.model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return

    print(f"Loading test data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    print("Creating time-series features for the test data...")
    df_featured = create_time_series_features(df, args.target_col, args.lags, args.window_size)

    # Define the features (X) and the actual values (y)
    features = [col for col in df_featured.columns if col not in ['timestamp', 'label', args.target_col]]
    X_test = df_featured[features]
    y_test = df_featured[args.target_col]

    print("Generating predictions...")
    predictions = model.predict(X_test)

    # --- Performance Evaluation ---
    print("\n--- Model Performance Metrics ---")
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    print(f"  Mean Squared Error (MSE):      {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE):  {rmse:.2f}")
    print(f"  Mean Absolute Error (MAE):     {mae:.2f}")
    print("---------------------------------\n")

    # --- Visualization ---
    print("Generating plot to compare actual vs. predicted values...")

    # To make the plot readable, we'll only show the last N data points
    plot_points = 1000

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(y_test.index[-plot_points:], y_test.values[-plot_points:], label='Actual Traffic', color='blue', linewidth=2)
    ax.plot(y_test.index[-plot_points:], predictions[-plot_points:], label='Predicted Traffic', color='orange', linestyle='--')

    ax.set_title(f'Traffic Forecasting: Actual vs. Predicted (Last {plot_points} points)', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel(f'{args.target_col}', fontsize=12)
    ax.legend(fontsize=12)

    if args.output_plot:
        plt.savefig(args.output_plot)
        print(f"✅ Plot saved to {args.output_plot}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained time-series forecasting model.")
    parser.add_argument("--input-csv", required=True, help="Input CSV file to use for testing.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file (.pkl).")
    parser.add_argument("--target-col", type=str, default="rx_packets", help="The column the model was trained to forecast.")
    parser.add_argument("--lags", type=int, default=5, help="Number of lag features used during training.")
    parser.add_argument("--window-size", type=int, default=10, help="Size of the rolling window used during training.")
    parser.add_argument("--output-plot", type=str, help="Optional path to save the output plot (e.g., prediction_plot.png).")

    args = parser.parse_args()
    main(args)

