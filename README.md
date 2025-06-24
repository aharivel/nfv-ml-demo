# NFV Telco ML Demo: Traffic Forecasting & Failure Prediction

This project demonstrates how to use machine learning for traffic forecasting and failure prediction in a Network Functions Virtualization (NFV) telco environment using DPDK-based VNFs. The demo uses TRex as a traffic generator, testPMD for metrics collection, and a Python-based ML pipeline for prediction and automation.

---

## Project Structure


- **data/**: Directory for storing raw and processed metrics data.

- **trex_profiles/**: Contains TRex traffic generation scripts.

- **testpmd_metrics/**: Includes scripts to collect and export testPMD metrics.

- **ml/**: Contains machine learning related files:
  - `train_model.py`: Script to train the machine learning model.
  - `model.pkl`: Serialized trained model.
  - `inference_server.py`: Script to serve the model for inference.

- **deployment/**: Contains scripts and configurations for deploying the ML model as an API.

- **requirements.txt**: Lists all Python dependencies required for the project.

- **utils/**: Contains helper scripts, such as those for data preprocessing.

---

## Simulation test

The goal of the simulation is to try to to tune and experiment with different algorithm
to find the most suited one for what we want to achieve.

### 1. **Generating Traffic**

- Generate 1 week of traffic in a CSV file:

```bash
cd testpmd_metrics/

python3 generate_metrics.py 
```

### 2. **Train Model**

- The traffic trainer script trains the model with the generated traffic
- At the moment, only Random Forest Regressor is available.
- 2 parameters to tune the model:
  - **lags** : default 5. To predict the traffic for right now, look at the traffic from 1 second ago, 2 seconds ago, 3 seconds ago, 4 seconds and 5 seconds ago.
  - **window-size** : default 10. To predict the traffic for right now, calculate the average and standard deviation of the traffic over the last 10 seconds.

```bash
cd ml/

python3 ts_traffic_trainer_model.py --input-csv ../data/metrics_1_week_sinusoidal.csv --output-model rx_forecaster_sin.pkl --lags 300 --window-size 300
```

### 3. **Perf test of the model accuracy**

- This script is used to evaluate the performance of a trained time-series forecasting model. Its main purpose is to load a pre-trained model and a test dataset, 
run predictions, and then compare those predictions against the actual ground truth values to determine the model's accuracy.

- The script performs the following steps:
  - Loads the Model: It loads the serialized model saved by the training script (e.g., rx_forecaster.pkl).
  - Loads Test Data: It reads a CSV file containing time-series data that the model has not seen before.
  - Feature Engineering: It creates the exact same set of features (lags, rolling windows, time-based features) that the model was originally trained on. This step is mandatory for the model to understand the input data.
  - Prediction: It uses the loaded model to predict the target variable (e.g., rx_packets) for every time step in the test data.
  - Performance Metrics: It calculates and prints key regression metrics to quantify the model's accuracy:
    - Mean Absolute Error (MAE): The average absolute difference between the predicted and actual values. A lower value is better.
    - Root Mean Squared Error (RMSE): Similar to MAE, but gives a higher weight to large errors. A lower value is better.
  - Visualization: It generates and displays a plot that visually overlays the model's predictions on top of the actual data, making it easy to see how well the model tracks the traffic patterns. This plot can optionally be saved to a file.

```bash
python test_forecasting_perf.py --input-csv ../data/metrics_2_week_sinusoidal.csv --model-path rx_forecaster_sin.pkl --target-col rx_packets --lags 300 --window-size 300 --output-plot prediction_vs_actual.png
```

---

## Hardware test (not yet functional / WIP)

### 1. **Traffic Generation with TRex**

- Use TRex to generate synthetic or realistic network traffic from Machine A to Machine B.
- Custom traffic profiles can be created to simulate various patterns or anomalies.

### 2. **Metrics Collection with testPMD**

- Run testPMD on Machine B to process incoming traffic and collect DPDK metrics.
- Key metrics: packets/bytes sent/received, error counts, latency, and core utilization.
- Metrics are exported periodically (e.g., every second) to CSV files in the `data/` directory.

### 3. **Machine Learning Pipeline**

- Use collected metrics to train ML models for:
  - **Traffic forecasting** (predict network load)
  - **Failure prediction** (detect/prevent faults)
- Models are trained and saved in the `ml/` directory.

### 4. **Model Deployment & Real-Time Inference**

- Deploy the trained model as a REST API using frameworks like BentoML or FastAPI.
- Real-time metrics from testPMD are sent to the API for prediction.
- Based on predictions, automated actions (e.g., alerts, scaling) can be triggered.

---

## How to Use This Project

### 1. **Set Up Environment**

- Install dependencies:

pip install -r requirements.txt

- Ensure you have access to two machines:
- **Machine A:** TRex traffic generator
- **Machine B:** testPMD (DPDK) for metrics collection

### 2. **Generate Traffic**

- Place your TRex traffic profile scripts in `trex_profiles/`.
- Start TRex on Machine A using the desired profile.

### 3. **Collect Metrics**

- Run the metrics collection scripts from `testpmd_metrics/` on Machine B.
- Metrics will be saved in the `data/` directory.

### 4. **Train ML Model**

- Use `ml/train_model.py` to process data and train your model.
- The trained model will be saved as `ml/model.pkl`.

### 5. **Deploy Model for Inference**

- Start the inference server using `ml/inference_server.py`.
- The server exposes a REST API for real-time predictions.

### 6. **Trigger Actions**

- Integrate your monitoring/automation scripts to call the inference API and act based on predictions (e.g., send alerts, scale resources).

---

## Example Workflow

1. **Generate traffic** with TRex:

./trex-console -f trex_profiles/my_profile.py -d 300

2. **Collect metrics** with testPMD:

python testpmd_metrics/collect_metrics.py

3. **Train your ML model:**

python ml/train_model.py --input data/metrics.csv --output ml/model.pkl

4. **Deploy the model as an API:**

python ml/inference_server.py --model ml/model.pkl

5. **Send real-time metrics for prediction:**

python utils/send_metrics.py --api-url http://localhost:8000/predict --metrics data/latest_metrics.json


---

## Key Technologies

- **TRex**: High-performance traffic generator
- **DPDK/testPMD**: Fast packet processing and metrics collection
- **Python (scikit-learn, pandas, FastAPI/BentoML)**: ML pipeline and API serving
- **CSV/JSON**: Data storage and interchange formats

---

## Extending the Demo

- Add advanced traffic patterns or failure scenarios in TRex profiles.
- Implement more sophisticated ML models (e.g., LSTM for time series).
- Integrate with Prometheus/Grafana for visualization.
- Use Kafka or another streaming platform for real-time data pipelines.

---

## License

MIT License

---

## Contributors

- Anthony Harivel

---

## References

- [DPDK Documentation](https://doc.dpdk.org/)
- [TRex Documentation](https://trex-tgn.cisco.com/)
- [scikit-learn](https://scikit-learn.org/)
- [BentoML](https://bentoml.com/)
- [FastAPI](https://fastapi.tiangolo.com/)






