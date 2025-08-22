import os
import io
import requests
import pandas as pd
import numpy as np

def ensure_data_local(url: str, dest_path: str, chunk_size: int = 1 << 20) -> str:
    """
    Descarga el dataset desde `url` si `dest_path` no existe.
    Devuelve la ruta local al archivo.
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest_path

def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga .pkl/.pkl.gz/.parquet/.csv en un DataFrame.
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # Por defecto pickle (incluye .pkl.gz)
    return pd.read_pickle(path)

def get_feature_target(df: pd.DataFrame, target_col: str):
    """
    Separa X/y y devuelve además nombres de columnas numéricas y categóricas.
    """
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return X, y, numeric_cols, categorical_cols
