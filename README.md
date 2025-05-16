# NFV Telco ML Demo: Traffic Forecasting & Failure Prediction

This project demonstrates how to use machine learning for traffic forecasting and failure prediction in a Network Functions Virtualization (NFV) telco environment using DPDK-based VNFs. The demo uses TRex as a traffic generator, testPMD for metrics collection, and a Python-based ML pipeline for prediction and automation.

---

## Project Structure

nfv-ml-demo/
│
├── README.md
├── data/
│   └── (raw and processed metrics)
├── trex_profiles/
│   └── (TRex traffic generation scripts)
├── testpmd_metrics/
│   └── (Scripts to collect and export testPMD metrics)
├── ml/
│   ├── train_model.py
│   ├── model.pkl
│   └── inference_server.py
├── deployment/
│   └── (Deployment scripts for ML model API)
├── requirements.txt
└── utils/
    └── (Helper scripts, e.g., for data preprocessing)

---

## Demo Overview

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






