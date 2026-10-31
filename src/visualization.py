"""Padrão visual e gráficos reutilizáveis (Matplotlib + Seaborn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES_DIR, TARGET, label

# Paleta validada para daltonismo: azul = alta qualidade, laranja = baixa/média
COLOR_HIGH = "#2a78d6"
COLOR_LOW = "#eb6834"
CLASS_PALETTE = {0: COLOR_LOW, 1: COLOR_HIGH}
CLASS_NAMES = {0: "Baixa/média (< 7)", 1: "Alta (≥ 7)"}
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # ordem fixa da paleta categórica
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Divergente vermelho (negativo) <-> azul (positivo) com meio cinza — mesma convenção das barras
DIVERGING = LinearSegmentedColormap.from_list(
    "vermelho_cinza_azul", ["#8c1f1f", "#e34948", "#f2a3a2", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])


def set_style():
    """Estilo único para todos os gráficos: fundo claro, grade discreta, sem bordas pesadas."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save_fig(fig, name: str):
    """Salva a figura em results/figures/<name>.png."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def class_legend(ax, loc="upper right"):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_PALETTE[k]) for k in (1, 0)]
    ax.legend(handles, [CLASS_NAMES[1], CLASS_NAMES[0]], loc=loc, fontsize=9)


def plot_hist_grid(df: pd.DataFrame, cols, ncols=4, bins=30):
    """Histogramas de todas as variáveis com a assimetria (skew) no título."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    for ax, c in zip(axes.flat, cols):
        ax.hist(df[c], bins=bins, color=COLOR_HIGH, alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(df[c].median(), color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(f"{label(c)}  (assimetria {df[c].skew():.1f})", fontsize=10)
        ax.set_ylabel("")
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distribuição das variáveis físico-químicas (linha tracejada = mediana)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_box_by_class(df: pd.DataFrame, cols, ncols=4):
    """Boxplot de cada variável separado por classe (alta x baixa/média)."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    order = [0, 1]
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(data=df, x=TARGET, y=c, hue=TARGET, order=order, palette=CLASS_PALETTE,
                    legend=False, ax=ax, width=0.55, linewidth=1, fliersize=2)
        ax.set_title(label(c), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baixa/média", "Alta"])
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Como cada variável se comporta nos vinhos de alta x baixa/média qualidade",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_kde_by_class(df: pd.DataFrame, cols, ncols=3):
    """Densidades sobrepostas por classe (revelam diferenças que o histograma esconde)."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3 * nrows))
    for ax, c in zip(np.atleast_1d(axes).flat, cols):
        for k in (0, 1):
            sns.kdeplot(df.loc[df[TARGET] == k, c], ax=ax, fill=True, alpha=0.25,
                        color=CLASS_PALETTE[k], linewidth=2, common_norm=False, label=CLASS_NAMES[k])
        ax.set_title(label(c), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks([])
    class_legend(np.atleast_1d(axes).flat[0], loc="upper right")
    for ax in list(np.atleast_1d(axes).flat)[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_corr_heatmap(corr: pd.DataFrame, title: str):
    """Mapa de calor de correlação com escala divergente centrada em zero."""
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8}, linewidths=1.5, linecolor=SURFACE, square=True,
                cbar_kws={"shrink": 0.7, "label": "Correlação"}, ax=ax,
                xticklabels=[label(c) for c in corr.columns], yticklabels=[label(c) for c in corr.index])
    ax.grid(False)
    ax.set_title(title, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def plot_bar_signed(values: pd.Series, title: str, xlabel: str, annotate=True):
    """Barras horizontais com sinal (azul = positivo, vermelho = negativo), ordenadas."""
    values = values.sort_values()
    colors = [COLOR_HIGH if v >= 0 else "#e34948" for v in values]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(values) + 1.2))
    ax.barh([label(i) for i in values.index], values.values, color=colors, height=0.6)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    if annotate:
        for i, v in enumerate(values.values):
            ax.text(v + (0.01 if v >= 0 else -0.01) * (abs(values).max() * 5), i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK_2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    lim = abs(values).max() * 1.3
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    return fig
