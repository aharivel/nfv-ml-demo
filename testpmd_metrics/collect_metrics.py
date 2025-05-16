import time
import csv
import subprocess

CSV_FILE = "../data/metrics.csv"

def get_testpmd_stats():
    # Replace this with actual logic to fetch testpmd stats, e.g., via SSH or telemetry
    # Here we simulate some dummy stats
    import random
    return {
        "timestamp": int(time.time()),
        "rx_packets": random.randint(100000, 200000),
        "tx_packets": random.randint(100000, 200000),
        "packet_drops": random.randint(0, 100),
        "latency": random.uniform(0.05, 1.5),
        "label": "normal"
    }

def main():
    with open(CSV_FILE, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "rx_packets", "tx_packets", "packet_drops", "latency", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(300):  # collect for 5 minutes (assuming 1s interval)
            stats = get_testpmd_stats()
            writer.writerow(stats)
            print(f"Collected: {stats}")
            time.sleep(1)

if __name__ == "__main__":
    main()

