from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

def build_preprocessor(X: pd.DataFrame):
    num_selector = selector(dtype_include=np.number)
    cat_selector = selector(dtype_exclude=np.number)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # compatible con sparse
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # devuelve sparse
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_selector),
            ("cat", categorical_transformer, cat_selector),
        ]
    )
    return preprocessor

def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    n_train: int,
    positive_class,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entrena un modelo simple (LogisticRegression) con N muestras y calcula métricas.
    Asume clasificación binaria para ROC/PR (elige 'positive_class').
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Limitar tamaño de entrenamiento
    n_train = int(min(n_train, len(X_train)))
    X_train_small = X_train.iloc[:n_train]
    y_train_small = y_train.iloc[:n_train]

    # Pipeline
    pre = build_preprocessor(X)
    clf = LogisticRegression(max_iter=200, solver="liblinear")

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train_small, y_train_small)

    # Predicciones
    y_pred = pipe.predict(X_test)

    # Probabilidades para ROC/PR (columna de la clase positiva)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        # Obtener índice de la clase positiva según classes_
        classes_ = pipe.named_steps["clf"].classes_
        if positive_class not in classes_:
            raise ValueError(f"La clase positiva '{positive_class}' no está en las clases del modelo: {classes_}")
        pos_idx = int(np.where(classes_ == positive_class)[0][0])
        y_score = proba[:, pos_idx]
    else:
        # Fallback a decision_function si existe
        y_score = pipe.decision_function(X_test)
        # Escalar a 0-1 por comodidad
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=positive_class)
    # Para ROC/PR se usa binarización booleana de la clase positiva
    y_true_pos = (y_test == positive_class).astype(int)
    roc_auc = roc_auc_score(y_true_pos, y_score)
    precision, recall, _ = precision_recall_curve(y_true_pos, y_score)
    avg_precision = average_precision_score(y_true_pos, y_score)

    return {
        "metrics": {
            "accuracy": float(acc),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "avg_precision": float(avg_precision),
        },
        "y_true": y_test,
        "y_pred": y_pred,
        "y_true_pos": y_true_pos,
        "y_score": y_score,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classes_": list(np.unique(y)),
    }
