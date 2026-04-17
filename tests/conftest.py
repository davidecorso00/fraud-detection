"""Fixtures condivise per i test."""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


# Le 10 feature esposte dall'API + un paio di feature extra per simulare
# l'allineamento che fa l'API (feature nel modello ma non nel payload).
API_FEATURES = [
    "TransactionAmt", "ProductCD", "card1", "card2", "card3",
    "card4", "card5", "card6", "addr1", "addr2",
]
EXTRA_MODEL_FEATURES = ["C1", "C2", "D1"]


@pytest.fixture
def mock_model():
    """Stub del modello MLflow: predict_proba deterministico + feature_names_in_."""
    m = MagicMock()
    m.feature_names_in_ = np.array(API_FEATURES + EXTRA_MODEL_FEATURES)
    m.predict_proba.return_value = np.array([[0.2, 0.8]])
    return m


@pytest.fixture
def api_client(mock_model, monkeypatch):
    """TestClient FastAPI con il modello MLflow sostituito da un MagicMock.

    Il patch avviene prima dell'import di src.serving.api perche' l'API
    carica il modello a import-time (module-level).
    """
    import mlflow.xgboost

    monkeypatch.setattr(mlflow.xgboost, "load_model", lambda uri: mock_model)

    # Rimuovi eventuale import cachato da test precedenti per forzare
    # la riesecuzione del module-level load con il mock attivo.
    sys.modules.pop("src.serving.api", None)

    from fastapi.testclient import TestClient
    from src.serving.api import app

    return TestClient(app)
