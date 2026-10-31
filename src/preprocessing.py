"""Pré-processamento e feature engineering, sempre dentro de um Pipeline (sem vazamento)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Variáveis com forte assimetria à direita (ver EDA): recebem log1p
SKEWED_FEATURES = [
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "sulphates",
]


class WineFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria variáveis derivadas com significado enológico.

    - total_acidity: acidez fixa + volátil (acidez percebida no paladar)
    - free_so2_ratio: SO₂ livre / SO₂ total (quanto do conservante está ativo)
    - bound_sulfur_dioxide: SO₂ total - livre (parte já combinada no vinho)
    - alcohol_x_sulphates: interação álcool × sulfatos (os dois maiores sinais positivos da EDA)

    É uma transformação linha a linha (não aprende nada dos dados), por isso
    não há risco de vazamento entre treino e teste.
    """

    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        if self.add_features:
            X["total_acidity"] = X["fixed_acidity"] + X["volatile_acidity"]
            X["free_so2_ratio"] = X["free_sulfur_dioxide"] / X["total_sulfur_dioxide"]
            X["bound_sulfur_dioxide"] = X["total_sulfur_dioxide"] - X["free_sulfur_dioxide"]
            X["alcohol_x_sulphates"] = X["alcohol"] * X["sulphates"]
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.add_features:
            names += ["total_acidity", "free_so2_ratio", "bound_sulfur_dioxide", "alcohol_x_sulphates"]
        return np.asarray(names, dtype=object)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Aplica log1p nas colunas assimétricas indicadas (as demais passam inalteradas)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        cols = SKEWED_FEATURES if self.columns is None else self.columns
        for c in cols:
            if c in X.columns:
                X[c] = np.log1p(X[c])
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def iqr_outlier_table(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Conta outliers pela regra de Tukey (1,5 × IQR) para cada variável."""
    rows = []
    for c in columns:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_low = int((df[c] < low).sum())
        n_high = int((df[c] > high).sum())
        rows.append({
            "variavel": c,
            "limite_inferior": round(low, 4),
            "limite_superior": round(high, 4),
            "outliers_abaixo": n_low,
            "outliers_acima": n_high,
            "total_outliers": n_low + n_high,
            "pct_outliers": round(100 * (n_low + n_high) / len(df), 1),
        })
    return pd.DataFrame(rows).sort_values("total_outliers", ascending=False).reset_index(drop=True)


def make_scaler():
    """Padronização z-score (StandardScaler), ajustada apenas no treino via Pipeline."""
    return StandardScaler()


def identity():
    """Transformador neutro (útil para desligar etapas no GridSearch)."""
    return FunctionTransformer(feature_names_out="one-to-one")
