"""Test del modulo src.monitoring.drift_report."""
from pathlib import Path

import pandas as pd
import pytest

from src.monitoring import drift_report


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "TransactionAmt": [float(i) for i in range(100)],
        "feature_a": list(range(100)),
        "isFraud": [0, 1] * 50,
    })
    p = tmp_path / "train_processed.csv"
    df.to_csv(p, index=False)
    return p


def test_load_splits_80_20(sample_csv):
    reference, current = drift_report.load_reference_and_current(sample_csv)

    assert len(reference) == 80
    assert len(current) == 20


def test_load_applies_shift_only_on_current(sample_csv):
    reference, current = drift_report.load_reference_and_current(sample_csv)

    expected_current = [v * drift_report.SHIFT_FACTOR for v in range(80, 100)]
    assert list(current["TransactionAmt"]) == expected_current

    # La reference non deve essere toccata
    assert list(reference["TransactionAmt"]) == [float(i) for i in range(80)]


def test_build_report_produces_metrics(sample_csv):
    reference, current = drift_report.load_reference_and_current(sample_csv)
    report = drift_report.build_report(reference, current)

    result = report.as_dict()
    assert "metrics" in result
    assert len(result["metrics"]) > 0


def test_main_writes_html_report(monkeypatch, tmp_path, sample_csv):
    out_dir = tmp_path / "reports"
    out_file = out_dir / "drift_report.html"

    monkeypatch.setattr(drift_report, "PROCESSED_FILE", sample_csv)
    monkeypatch.setattr(drift_report, "REPORTS_DIR", out_dir)
    monkeypatch.setattr(drift_report, "REPORT_FILE", out_file)

    drift_report.main()

    assert out_file.exists()
    assert out_file.stat().st_size > 0
