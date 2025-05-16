from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class Metrics(BaseModel):
    rx_packets: int
    tx_packets: int
    packet_drops: int
    latency: float

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.pkl")

@app.post("/predict")
def predict(metrics: Metrics):
    X = [[metrics.rx_packets, metrics.tx_packets, metrics.packet_drops, metrics.latency]]
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "meaning": "failure" if pred else "normal"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_server:app", host="0.0.0.0", port=8000, reload=True)

