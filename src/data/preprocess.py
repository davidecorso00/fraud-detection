import pandas as pd
import numpy as np
from pathlib import Path

#Path
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# File
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
OUTPUT_FILE = PROCESSED_DIR / "train_processed.csv"


def load_data() -> pd.DataFrame:
    print("Caricamento train_transaction...")
    transaction = pd.read_csv(TRANSACTION_FILE)
    
    print("Caricamento train_identity...")
    identity = pd.read_csv(IDENTITY_FILE)
    
    print("Join dei due dataset...")
    df = transaction.merge(identity, on="TransactionID", how="left")
    
    print(f"Shape finale: {df.shape}")
    return df


def explore(df: pd.DataFrame) -> None:
    print("\n=== INFO GENERALI ===")
    print(f"Righe: {df.shape[0]}, Colonne: {df.shape[1]}")
    
    print("\n=== TARGET (isFraud) ===")
    counts = df["isFraud"].value_counts()
    pct = df["isFraud"].value_counts(normalize=True) * 100
    print(f"Non frode (0): {counts[0]} ({pct[0]:.2f}%)")
    print(f"Frode    (1): {counts[1]} ({pct[1]:.2f}%)")
    
    print("\n=== VALORI MANCANTI ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "missing": missing,
        "pct": missing_pct
    }).query("missing > 0").sort_values("pct", ascending=False)
    print(f"Colonne con NaN: {len(missing_df)} su {df.shape[1]}")
    print(missing_df.head(20).to_string())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== PULIZIA ===")
    
    # Droppa colonne con più del 90% di NaN
    threshold = 0.90
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Colonne droppate (>{threshold*100:.0f}% NaN): {len(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)
    
    # Droppa colonne con un solo valore unico (non informative)
    single_value_cols = [c for c in df.columns if df[c].nunique() <= 1]
    print(f"Colonne droppate (valore unico): {len(single_value_cols)}")
    df = df.drop(columns=single_value_cols)
    
    print(f"Shape dopo pulizia: {df.shape}")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== IMPUTAZIONE NaN ===")
    
    # Separa colonne numeriche e categoriche
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Rimuovi la target dai numerici
    num_cols = [c for c in num_cols if c != "isFraud"]
    
    print(f"Colonne numeriche: {len(num_cols)}")
    print(f"Colonne categoriche: {len(cat_cols)}")
    
    # Imputa numeriche con mediana
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Imputa categoriche con "missing"
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("missing")
    
    remaining = df.isnull().sum().sum()
    print(f"NaN rimanenti: {remaining}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== ENCODING CATEGORICHE ===")
    
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Colonne da encodare: {len(cat_cols)}")
    
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    
    print(f"Shape finale: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_data()
    explore(df)
    df = clean(df)
    df = impute(df)
    df = encode(df)

    print("\n=== SALVATAGGIO ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Salvato in {OUTPUT_FILE}")