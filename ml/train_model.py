import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse

def main(input_csv, output_model):
    df = pd.read_csv(input_csv)
    X = df[["rx_packets", "tx_packets", "packet_drops", "latency"]]
    y = df["label"]
    # For demo, convert labels to binary
    y = (y != "normal").astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, output_model)
    print(f"Model trained and saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file with metrics")
    parser.add_argument("--output", required=True, help="Output model file (.pkl)")
    args = parser.parse_args()
    main(args.input, args.output)

