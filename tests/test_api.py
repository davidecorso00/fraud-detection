"""Test degli endpoint FastAPI /health e /predict con modello MLflow mockato."""
import numpy as np


VALID_PAYLOAD = {
    "TransactionAmt": 100.0,
    "ProductCD": 1,
    "card1": 1234.0,
    "card2": 111.0,
    "card3": 150.0,
    "card4": 2,
    "card5": 226.0,
    "card6": 1,
    "addr1": 315.0,
    "addr2": 87.0,
}


def test_health_returns_ok(api_client):
    r = api_client.get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_expected_schema(api_client):
    r = api_client.post("/predict", json=VALID_PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"is_fraud", "fraud_probability"}
    assert isinstance(body["is_fraud"], bool)
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_threshold_logic(api_client, mock_model):
    # Con predict_proba di default (0.8) -> is_fraud True, prob 0.8
    r = api_client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] == 0.8

    # Sotto soglia 0.5 -> is_fraud False
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    r = api_client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["is_fraud"] is False
    assert body["fraud_probability"] == 0.3


def test_predict_aligns_columns_to_model_features(api_client, mock_model):
    """L'API deve riempire con 0 le feature che il modello ha ma il payload no,
    e passare al modello un DataFrame con esattamente feature_names_in_."""
    api_client.post("/predict", json=VALID_PAYLOAD)

    call_args = mock_model.predict_proba.call_args
    passed_df = call_args[0][0]
    assert list(passed_df.columns) == list(mock_model.feature_names_in_)
    # Le feature non presenti nel payload sono riempite con 0
    for extra in ["C1", "C2", "D1"]:
        assert passed_df[extra].iloc[0] == 0


def test_predict_rejects_missing_fields(api_client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "card1"}
    r = api_client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_predict_rejects_wrong_types(api_client):
    bad = dict(VALID_PAYLOAD, TransactionAmt="not-a-number")
    r = api_client.post("/predict", json=bad)
    assert r.status_code == 422
