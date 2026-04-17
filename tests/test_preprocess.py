"""Test delle funzioni di preprocessing: clean, impute, encode."""
import numpy as np
import pandas as pd

from src.data.preprocess import clean, encode, impute


def test_clean_drops_high_nan_columns():
    # 'high_nan' ha 95% NaN, deve essere rimossa (threshold 90%)
    df = pd.DataFrame({
        "ok": range(20),
        "high_nan": [np.nan] * 19 + [1.0],
        "isFraud": [0, 1] * 10,
    })

    result = clean(df)

    assert "ok" in result.columns
    assert "high_nan" not in result.columns
    assert "isFraud" in result.columns


def test_clean_drops_single_value_columns():
    # 'constant' ha un solo valore unico, deve essere rimossa
    df = pd.DataFrame({
        "ok": range(10),
        "constant": [5] * 10,
        "isFraud": [0, 1] * 5,
    })

    result = clean(df)

    assert "constant" not in result.columns
    assert "ok" in result.columns


def test_impute_fills_numeric_with_median():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "isFraud": [0, 1, 0, 1, 0],
    })

    result = impute(df)

    assert result["num"].isnull().sum() == 0
    # Mediana di [1, 2, 4, 5] = 3.0
    assert result.loc[2, "num"] == 3.0


def test_impute_fills_categorical_with_missing_literal():
    df = pd.DataFrame({
        "cat": ["a", None, "b", "a"],
        "isFraud": [0, 1, 0, 1],
    })

    result = impute(df)

    assert result["cat"].isnull().sum() == 0
    assert result.loc[1, "cat"] == "missing"


def test_impute_does_not_touch_target_column():
    # La target non deve essere toccata anche se avesse NaN (non dovrebbe averli
    # in produzione, ma serve a documentare il contratto)
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0],
        "isFraud": [0, 1, 0],
    })

    result = impute(df)

    assert list(result["isFraud"]) == [0, 1, 0]


def test_encode_converts_categoricals_to_integers():
    df = pd.DataFrame({
        "cat": ["a", "b", "a", "c", "b"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    result = encode(df)

    assert pd.api.types.is_integer_dtype(result["cat"])
    # Le numeriche restano invariate
    assert list(result["num"]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    # Stessi valori originali -> stessi codici
    assert result.loc[0, "cat"] == result.loc[2, "cat"]
    assert result.loc[1, "cat"] == result.loc[4, "cat"]


def test_pipeline_end_to_end_produces_numeric_only_frame():
    df = pd.DataFrame({
        "num_with_nan": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat_with_nan": ["x", "y", None, "x", "y", "x", "y", "x", "y", "x"],
        "all_nan": [np.nan] * 10,
        "isFraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    cleaned = clean(df)
    imputed = impute(cleaned)
    encoded = encode(imputed)

    assert "all_nan" not in encoded.columns
    assert encoded.isnull().sum().sum() == 0
    non_target = encoded.drop(columns=["isFraud"])
    assert all(pd.api.types.is_numeric_dtype(non_target[c]) for c in non_target.columns)
