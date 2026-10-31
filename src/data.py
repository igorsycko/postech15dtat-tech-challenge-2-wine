"""Carga, padronização e preparação inicial da base Wine Quality."""

from __future__ import annotations

import pandas as pd

from .config import DATA_PATH, FEATURES, QUALITY_THRESHOLD, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Lê o CSV original do Kaggle (WineQT.csv) sem nenhuma alteração."""
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os nomes das colunas: minúsculas, sem espaços (boa prática das aulas)."""
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def create_target(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 = alta qualidade (nota >= threshold), 0 = baixa/média."""
    out = df.copy()
    out[TARGET] = (out["quality"] >= threshold).astype(int)
    return out


def duplicate_report(df: pd.DataFrame) -> dict:
    """Resume as linhas duplicadas considerando as 11 variáveis químicas + a nota."""
    cols = FEATURES + ["quality"]
    dup_mask = df.duplicated(subset=cols, keep="first")
    groups = df[df.duplicated(subset=cols, keep=False)].groupby(cols).size()
    conflicting = df.groupby(FEATURES)["quality"].nunique().gt(1).sum()
    return {
        "linhas_total": len(df),
        "linhas_duplicadas_removiveis": int(dup_mask.sum()),
        "percentual_duplicado": round(100 * dup_mask.mean(), 1),
        "grupos_com_repeticao": int(len(groups)),
        "maior_repeticao": int(groups.max()) if len(groups) else 0,
        "mesma_quimica_com_notas_diferentes": int(conflicting),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove amostras repetidas (mesma química e mesma nota), mantendo a primeira ocorrência.

    Motivo: linhas idênticas podem cair ao mesmo tempo no treino e no teste,
    o que faz o modelo ser avaliado em exemplos que ele já viu (vazamento).
    """
    return df.drop_duplicates(subset=FEATURES + ["quality"], keep="first").reset_index(drop=True)


def load_dataset(drop_duplicates: bool = True) -> pd.DataFrame:
    """Pipeline completo de carga: lê, padroniza nomes, remove Id, cria alvo e (opcional) deduplica."""
    df = standardize_columns(load_raw())
    df = df.drop(columns=["id"], errors="ignore")
    df = create_target(df)
    if drop_duplicates:
        df = remove_duplicates(df)
    return df
