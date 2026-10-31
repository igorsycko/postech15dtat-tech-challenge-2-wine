"""Definição dos modelos, grades de hiperparâmetros e busca com validação cruzada estratificada."""

from __future__ import annotations

import time

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_STATE
from .preprocessing import LogTransformer, WineFeatureEngineer, make_scaler

# Métricas avaliadas na validação cruzada; a escolha dos hiperparâmetros usa o F1 da classe "alta"
SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
}

MODEL_NAMES = {
    "logreg": "Regressão Logística",
    "knn": "KNN",
    "svm": "SVM (kernel RBF)",
    "tree": "Árvore de Decisão",
    "rf": "Random Forest",
}


def cv_splitter(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """K-Fold estratificado e embaralhado: mantém ~13% de vinhos de alta qualidade em cada dobra."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def build_models() -> dict:
    """Retorna {nome: (pipeline, grade_de_hiperparâmetros)}.

    Todo o pré-processamento fica DENTRO do pipeline, então o scaler e o
    oversampling são ajustados apenas nas dobras de treino (sem vazamento).
    Desbalanceamento: class_weight='balanced' nos modelos que aceitam; no KNN,
    que não aceita pesos, usamos oversampling aplicado só no treino.
    """
    fe = ("features", WineFeatureEngineer())
    log = ("log", LogTransformer())
    scale = ("scaler", make_scaler())

    models = {
        "logreg": (
            Pipeline([fe, log, scale,
                      ("model", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {
                "features__add_features": [False, True],
                "model__C": [0.01, 0.1, 1, 10],
            },
        ),
        "knn": (
            Pipeline([fe, log, scale,
                      ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
                      ("model", KNeighborsClassifier())]),
            {
                "model__n_neighbors": [5, 9, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        ),
        "svm": (
            Pipeline([fe, log, scale,
                      ("model", SVC(kernel="rbf", class_weight="balanced",
                                    probability=True, random_state=RANDOM_STATE))]),
            {
                "model__C": [0.3, 1, 3, 10],
                "model__gamma": ["scale", 0.03, 0.1],
            },
        ),
        "tree": (
            Pipeline([fe,
                      ("model", DecisionTreeClassifier(class_weight="balanced",
                                                       random_state=RANDOM_STATE))]),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
        "rf": (
            Pipeline([fe,
                      ("model", RandomForestClassifier(n_estimators=400,
                                                       class_weight="balanced_subsample",
                                                       random_state=RANDOM_STATE, n_jobs=-1))]),
            {
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
    }
    return models


def tune_models(X_train, y_train, models: dict | None = None, verbose: bool = True):
    """Roda GridSearchCV (5 dobras estratificadas, refit pelo F1) para cada modelo.

    Retorna (buscas_ajustadas, tabela_resumo_cv).
    """
    models = models or build_models()
    searches, rows = {}, []
    for name, (pipe, grid) in models.items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="f1",
                          cv=cv_splitter(), n_jobs=-1, return_train_score=True)
        gs.fit(X_train, y_train)
        searches[name] = gs
        i = gs.best_index_
        res = gs.cv_results_
        row = {"modelo": MODEL_NAMES[name]}
        for m in SCORING:
            row[f"cv_{m}"] = res[f"mean_test_{m}"][i]
            row[f"cv_{m}_dp"] = res[f"std_test_{m}"][i]
        row["treino_f1"] = res["mean_train_f1"][i]
        row["melhores_parametros"] = gs.best_params_
        row["tempo_s"] = round(time.time() - t0, 1)
        rows.append(row)
        if verbose:
            print(f"{MODEL_NAMES[name]:<22} F1 CV = {row['cv_f1']:.3f} ± {row['cv_f1_dp']:.3f} "
                  f"| ROC-AUC = {row['cv_roc_auc']:.3f} | {row['tempo_s']}s")
    summary = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    return searches, summary
