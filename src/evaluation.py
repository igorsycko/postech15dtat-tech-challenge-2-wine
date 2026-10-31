"""Métricas de classificação, escolha de limiar e análise de erros."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from .modeling import cv_splitter


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Calcula as métricas principais a partir das probabilidades e de um limiar."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "limiar": threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "especificidade": tn / (tn + fp) if (tn + fp) else 0.0,
        "acuracia": accuracy_score(y_true, y_pred),
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
    }


def oof_probabilities(estimator, X_train, y_train) -> np.ndarray:
    """Probabilidades 'fora da dobra' no treino (cada linha prevista por um modelo que não a viu)."""
    return cross_val_predict(estimator, X_train, y_train, cv=cv_splitter(),
                             method="predict_proba", n_jobs=-1)[:, 1]


def best_f1_threshold(y_true, y_proba) -> tuple[float, float]:
    """Limiar que maximiza o F1 — escolhido no TREINO (probabilidades fora da dobra), nunca no teste."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thresholds[i]), float(f1[i])


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Tabela de trade-off precisão x recall para vários limiares (apoio à decisão de negócio)."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = [classification_metrics(y_true, y_proba, t) for t in thresholds]
    cols = ["limiar", "precision", "recall", "f1", "especificidade", "VP", "FP", "FN", "VN"]
    return pd.DataFrame(rows)[cols]


def format_metrics(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Arredonda colunas numéricas para exibição."""
    out = df.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    return out
