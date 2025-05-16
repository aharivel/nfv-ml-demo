import pandas as pd

def preprocess(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Example: normalize columns
    for col in ["rx_packets", "tx_packets", "packet_drops", "latency"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess("../data/metrics.csv", "../data/metrics_norm.csv")

