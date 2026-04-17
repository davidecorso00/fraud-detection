import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

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
        stratify=y,
    )

    print(f"Training set: {X_train.shape[0]} righe")
    print(f"Test set: {X_test.shape[0]} righe")

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    # scale_pos_weight bilancia il target sbilanciato (equivalente a class_weight di sklearn)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(params)
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("stage", "baseline")

        print("\nTraining XGBoost in corso...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("precision_fraud", report["1"]["precision"])
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("f1_fraud", report["1"]["f1-score"])

        mlflow.xgboost.log_model(model, "model")

        print(f"\nAUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("fraud-detection")

    X_train, X_test, y_train, y_test = load_data()
    model = train(X_train, X_test, y_train, y_test)
