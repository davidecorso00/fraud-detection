"""Promuove una versione del modello registrato a un alias (default: champion).

Uso:
    python -m src.models.promote                      # champion -> v3
    python -m src.models.promote --version 4          # champion -> v4
    python -m src.models.promote --alias challenger --version 2
"""
import argparse
from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-detector"
DEFAULT_VERSION = 3
DEFAULT_ALIAS = "champion"


def promote(version: int, alias: str, model_name: str = MODEL_NAME) -> None:
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, str(version))
    print(f"Alias '{alias}' -> {model_name} v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promuovi un modello a un alias MLflow.")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--alias", type=str, default=DEFAULT_ALIAS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    promote(version=args.version, alias=args.alias, model_name=args.model_name)


if __name__ == "__main__":
    main()
