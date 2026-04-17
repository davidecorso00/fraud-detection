from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

app = FastAPI(title="Fraud Detection API", version="1.1.0")

# Alias stabile: la versione puntata da @champion può cambiare nel registry
# senza toccare il codice (vedi src/models/promote.py).
MODEL_URI = "models:/fraud-detector@champion"
model = mlflow.xgboost.load_model(MODEL_URI)


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: int
    card1: float
    card2: float
    card3: float
    card4: int
    card5: float
    card6: int
    addr1: float
    addr2: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in data.columns:
            data[col] = 0
    data = data[model_features]

    prob = model.predict_proba(data)[0][1]
    is_fraud = bool(prob > 0.5)

    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(float(prob), 4),
    }
