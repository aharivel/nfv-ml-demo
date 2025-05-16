import requests
import json
import argparse

def main(api_url, metrics_json):
    with open(metrics_json, "r") as f:
        metrics = json.load(f)
    response = requests.post(api_url, json=metrics)
    print("Prediction response:", response.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True, help="Inference API URL")
    parser.add_argument("--metrics", required=True, help="JSON file with metrics")
    args = parser.parse_args()
    main(args.api_url, args.metrics)

