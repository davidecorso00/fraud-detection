import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.xgboost

# Path
PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train_processed.csv"

# Parametri
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
TARGET = "isFraud"
N_TRIALS = 25
REGISTERED_MODEL_NAME = "fraud-detector"


def load_splits():
    print("Caricamento dati processati...")
    df = pd.read_csv(TRAIN_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Split iniziale: train+val vs test (held-out per la valutazione finale)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split interno: train vs val (val usato per early stopping e per Optuna)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    return (X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight)


def make_objective(X_train, X_val, y_train, y_val, scale_pos_weight):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        fixed = {
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.set_tag("model_family", "xgboost")
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("trial_number", trial.number)

            model = XGBClassifier(**params, **fixed)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_val_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_prob)

            mlflow.log_metric("val_auc_roc", auc)
            mlflow.log_metric("best_iteration", model.best_iteration)

        return auc

    return objective


def retrain_and_register(best_params, X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight):
    # Refit su train+val con i best params, valuto su test held-out
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    fixed = {
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-best") as run:
        mlflow.log_params(best_params)
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("stage", "final")

        print("\nRefit sul full train con i best params...")
        model = XGBClassifier(**best_params, **fixed)
        model.fit(X_full, y_full, verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("precision_fraud", report["1"]["precision"])
        mlflow.log_metric("recall_fraud", report["1"]["recall"])
        mlflow.log_metric("f1_fraud", report["1"]["f1-score"])

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"\nTest AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        return run.info.run_id, auc


if __name__ == "__main__":
    mlflow.set_experiment("fraud-detection")

    (X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight) = load_splits()

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgboost-tuning")

    with mlflow.start_run(run_name="xgboost-optuna-parent"):
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("stage", "tuning-parent")
        mlflow.log_param("n_trials", N_TRIALS)

        objective = make_objective(X_train, X_val, y_train, y_val, scale_pos_weight)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        print(f"\nBest val AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_auc", study.best_value)

    run_id, test_auc = retrain_and_register(
        study.best_params, X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight,
    )
    print(f"\nModello registrato come '{REGISTERED_MODEL_NAME}' (run {run_id}), test AUC {test_auc:.4f}")
