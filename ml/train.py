from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
    Entrena un modelo (LogisticRegression) con N muestras y calcula métricas.
    Incluye validación cruzada para dar valores más realistas y menos inflados.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 1. Eliminar columnas que copian al target
    if target_col in X.columns:
        X = X.drop(columns=[target_col])

    # 2. Quitar columnas con cardinalidad igual al target (posible fuga de info)
    for col in X.columns:
        if X[col].nunique() == y.nunique() and set(X[col].unique()) == set(y.unique()):
            X = X.drop(columns=[col])

    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Ajustar n_train al tamaño disponible
    n_train = min(n_train, len(X_train))
    if n_train < len(X_train):
        train_idx = np.random.choice(len(X_train), size=n_train, replace=False)
        X_train = X_train.iloc[train_idx]
        y_train = y_train.iloc[train_idx]

    # 5. Pipeline completo (preprocesador + modelo)
    pre = build_preprocessor(X)
    clf = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # 6. Validación cruzada en el set de entrenamiento
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

    # 7. Entrenamiento final con los datos de train
    pipe.fit(X_train, y_train)

    # 8. Predicciones
    y_pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        classes_ = pipe.named_steps["clf"].classes_
        if positive_class not in classes_:
            raise ValueError(f"La clase positiva '{positive_class}' no está en las clases del modelo: {classes_}")
        pos_idx = int(np.where(classes_ == positive_class)[0][0])
        y_score = proba[:, pos_idx]
    else:
        y_score = pipe.decision_function(X_test)
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    # 9. Métricas
    y_true_pos = (y_test == positive_class).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, pos_label=positive_class)),
        "roc_auc": float(roc_auc_score(y_true_pos, y_score)),
        "avg_precision": float(average_precision_score(y_true_pos, y_score)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
    }

    return {
        "metrics": metrics,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_true_pos": y_true_pos,
        "y_score": y_score,
        "precision": precision_recall_curve(y_true_pos, y_score)[0],
        "recall": precision_recall_curve(y_true_pos, y_score)[1],
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classes_": list(np.unique(y)),
    }

