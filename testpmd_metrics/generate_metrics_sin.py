import csv
import random
import numpy as np # We need numpy for the sine wave functions
from datetime import datetime, timedelta

# Define the output file path for the new dataset
CSV_FILE = "metrics_1_week_sinusoidal.csv"

def generate_sinusoidal_traffic(simulated_time):
    """
    Generates a smooth, wave-like traffic pattern based on the time of day.

    Args:
        simulated_time (datetime): The exact time to generate data for.

    Returns:
        tuple: A tuple containing the rx_packets and tx_packets.
    """
    # --- Define the base traffic levels ---
    MIN_TRAFFIC = 20000  # The lowest traffic level during the night
    MAX_TRAFFIC = 300000 # The peak traffic level during the day

    # Calculate the number of seconds that have passed since the start of the day
    seconds_in_day = 24 * 60 * 60
    seconds_from_midnight = (simulated_time - simulated_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

    # --- Create the Sine Wave ---
    # Convert the second of the day into an angle for the sine function.
    # We add a phase shift to make the lowest point occur around 2 AM.
    # A phase shift of -8 hours (in seconds) makes the trough at 2 AM and peak at 2 PM.
    phase_shift_seconds = -8 * 3600
    angle = ((seconds_from_midnight + phase_shift_seconds) / seconds_in_day) * 2 * np.pi

    # The sine function gives a value from -1 (trough) to +1 (peak)
    sin_value = np.sin(angle)

    # --- Scale the Sine Wave to our Traffic Range ---
    # First, map the [-1, 1] range to [0, 1]
    scaled_value = (sin_value + 1) / 2

    # Now, map the [0, 1] range to our [MIN_TRAFFIC, MAX_TRAFFIC] range
    base_traffic = MIN_TRAFFIC + scaled_value * (MAX_TRAFFIC - MIN_TRAFFIC)

    # --- Add Realistic Jitter/Noise ---
    # Add a small amount of random noise to make it less perfect.
    # This noise is much smaller than the main signal.
    noise = random.uniform(-5000, 5000) # Fluctuate by at most 5000 packets
    final_traffic_value = int(base_traffic + noise)

    # Ensure traffic never goes below zero
    if final_traffic_value < 0:
        final_traffic_value = 0

    # For simplicity, we'll make rx and tx packets follow the same pattern but be slightly different
    rx = final_traffic_value
    tx = int(final_traffic_value * random.uniform(0.95, 1.05))

    return rx, tx

def get_testpmd_stats(simulated_time, has_drop):
    """
    Generates a full data point using the sinusoidal traffic pattern.

    Args:
        simulated_time (datetime): The exact time to generate data for.
        has_drop (bool): True if a packet drop should be recorded.

    Returns:
        dict: A dictionary representing one row of data.
    """
    timestamp = int(simulated_time.timestamp())
    rx_packets, tx_packets = generate_sinusoidal_traffic(simulated_time)

    # Latency can also follow a similar, but less pronounced, daily pattern
    latency = 0.1 + ((rx_packets / 300000) * 1.5) + random.uniform(-0.1, 0.1)
    label = "normal_day" if 7 <= simulated_time.hour < 22 else "normal_night"
    packet_drops = 1 if has_drop else 0

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
    """
    days_to_simulate = 7
    seconds_in_a_day = 24 * 60 * 60
    start_time = datetime.now()

    try:
        with open(CSV_FILE, "w", newline="") as csvfile:
            fieldnames = ["timestamp", "rx_packets", "tx_packets", "packet_drops", "latency", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            print(f"Starting sinusoidal data generation for {days_to_simulate} days...")

            for day_offset in range(days_to_simulate):
                print(f"  - Generating data for Day {day_offset + 1}...")

                total_daily_drops = random.randint(1, 10)
                drop_seconds_for_day = set(random.sample(range(seconds_in_a_day), total_daily_drops))

                for second_offset in range(seconds_in_a_day):
                    current_sim_time = start_time + timedelta(days=day_offset, seconds=second_offset)
                    should_drop_packet = second_offset in drop_seconds_for_day

                    stats = get_testpmd_stats(current_sim_time, should_drop_packet)
                    writer.writerow(stats)

            print(f"\nData generation finished. Data saved to '{CSV_FILE}'")

    except IOError as e:
        print(f"Error writing to file {CSV_FILE}: {e}")

if __name__ == "__main__":
    main()

