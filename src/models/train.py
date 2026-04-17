import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn

# Path
PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train_processed.csv"
MODEL_DIR = Path("models")

# Parametri
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET = "isFraud"


def load_data():
    print("Caricamento dati processati...")
    df = pd.read_csv(TRAIN_FILE)
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    print(f"Shape X: {X.shape}")
    print(f"Frodi nel dataset: {y.sum()} ({y.mean()*100:.2f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} righe")
    print(f"Test set: {X_test.shape[0]} righe")
    
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    
    params = {
        "n_estimators": 100,
        "max_depth": 20,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    
    with mlflow.start_run():
        
        # Logga parametri
        mlflow.log_params(params)
        
        # Training
        print("\nTraining in corso...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predizioni
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metriche
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("precision_fraud", report["1"]["precision"])
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("f1_fraud", report["1"]["f1-score"])
        
        # Salva modello
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nAUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
    return model


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment("fraud-detection")
    
    X_train, X_test, y_train, y_test = load_data()
    model = train(X_train, X_test, y_train, y_test)