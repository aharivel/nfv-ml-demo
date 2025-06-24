import time
import csv
import random
from datetime import datetime, timedelta

# Define the output file path for the larger dataset
CSV_FILE = "metrics_1_week.csv"

def get_testpmd_stats(current_hour, timestamp, has_drop):
    """
    Generates simulated testpmd statistics based on a given time.

    This function simulates a high-traffic "day" profile or a low-traffic
    "night" profile based on the hour provided. Packet drops are determined
    by the 'has_drop' boolean.

    Args:
        current_hour (int): The hour of the day (0-23) to simulate.
        timestamp (int): The Unix timestamp for the data point.
        has_drop (bool): True if a packet drop should be recorded, False otherwise.
    """
    # --- Define Day and Night Periods ---
    # Day is considered from 7 AM (7) to 10 PM (22).
    if 7 <= current_hour < 22:
        # --- Day Traffic Profile (High Traffic) ---
        # Higher packet counts, more potential for drops and higher latency.
        rx_packets = random.randint(150000, 350000)
        tx_packets = random.randint(150000, 350000)
        latency = random.uniform(0.15, 2.2)
        label = "normal_day"
    else:
        # --- Night Traffic Profile (Low Traffic) ---
        # Lower packet counts, minimal drops, and low latency.
        rx_packets = random.randint(5000, 60000)
        tx_packets = random.randint(5000, 60000)
        latency = random.uniform(0.05, 0.4)
        label = "normal_night"

    # A single packet is dropped if the simulation logic for the day decides it.
    packet_drops = 1 if has_drop else 0

    # Return the generated stats as a dictionary
    return {
        "timestamp": timestamp,
        "rx_packets": rx_packets,
        "tx_packets": tx_packets,
        "packet_drops": packet_drops,
        "latency": latency,
        "label": label
    }

def main():
    """
    Main function to generate a week's worth of simulated data and write it to a CSV file.
    It simulates time passing instead of waiting in real-time.
    """
    days_to_simulate = 7
    seconds_in_a_day = 24 * 60 * 60

    # We can use the current time as a starting point for our simulation
    # or a fixed one for reproducibility.
    start_time = datetime.now()

    try:
        with open(CSV_FILE, "w", newline="") as csvfile:
            # Define the column headers for the CSV file
            fieldnames = ["timestamp", "rx_packets", "tx_packets", "packet_drops", "latency", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            print(f"Starting data generation for {days_to_simulate} days...")

            # The main loop iterates through each day to be simulated
            for day_offset in range(days_to_simulate):
                print(f"  - Generating data for Day {day_offset + 1}...")

                # --- Packet Drop Simulation for the Day ---
                # Decide how many total packets will be dropped this day (1 to 10).
                total_daily_drops = random.randint(1, 10)
                # Create a set of specific seconds within the day when these drops will occur.
                drop_seconds_for_day = set(random.sample(range(seconds_in_a_day), total_daily_drops))

                # Inner loop iterates through every second of the current simulated day
                for second_offset in range(seconds_in_a_day):
                    # Calculate the exact time for this data point
                    current_sim_time = start_time + timedelta(days=day_offset, seconds=second_offset)
                    current_hour = current_sim_time.hour
                    current_timestamp = int(current_sim_time.timestamp())

                    # Check if this specific second is one of the chosen drop times
                    should_drop_packet = second_offset in drop_seconds_for_day

                    # Fetch the simulated stats for this exact moment
                    stats = get_testpmd_stats(current_hour, current_timestamp, should_drop_packet)

                    # Write the stats to the CSV file
                    writer.writerow(stats)

            print(f"\nData generation finished. {days_to_simulate} days of data saved to '{CSV_FILE}'")

    except IOError as e:
        print(f"Error writing to file {CSV_FILE}: {e}")

if __name__ == "__main__":
    main()

