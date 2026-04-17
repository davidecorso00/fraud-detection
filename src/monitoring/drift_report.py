"""Genera un report Evidently di data drift sul dataset processato.

Confronta una porzione 'reference' (primi 80% delle righe) con una 'current'
(ultimi 20%) in cui viene introdotto uno shift artificiale su TransactionAmt
per dimostrare il rilevamento del drift.

Uso:
    python -m src.monitoring.drift_report
"""
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

PROCESSED_FILE = Path("data/processed/train_processed.csv")
REPORTS_DIR = Path("reports")
REPORT_FILE = REPORTS_DIR / "drift_report.html"

REFERENCE_FRACTION = 0.8
SHIFT_COLUMN = "TransactionAmt"
SHIFT_FACTOR = 1.5


def load_reference_and_current(path: Path = PROCESSED_FILE):
    df = pd.read_csv(path)
    cut = int(len(df) * REFERENCE_FRACTION)
    reference = df.iloc[:cut].copy()
    current = df.iloc[cut:].copy()

    # Shift artificiale per simulare drift in produzione
    if SHIFT_COLUMN in current.columns:
        current[SHIFT_COLUMN] = current[SHIFT_COLUMN] * SHIFT_FACTOR

    print(f"Reference: {reference.shape} | Current: {current.shape}")
    print(f"Shift simulato: {SHIFT_COLUMN} *= {SHIFT_FACTOR}")
    return reference, current


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # PROCESSED_FILE passato esplicito per rispettare i monkeypatch nei test
    # (i default delle funzioni sono valutati a import-time, non a runtime).
    reference, current = load_reference_and_current(PROCESSED_FILE)
    report = build_report(reference, current)
    report.save_html(str(REPORT_FILE))

    print(f"Report salvato in {REPORT_FILE}")


if __name__ == "__main__":
    main()
