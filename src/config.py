"""Configurações centrais do projeto: caminhos, constantes e rótulos em português."""

from pathlib import Path

# Raiz do repositório (a pasta acima de src/)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "WineQT.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Reprodutibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Definição do alvo (enunciado do Tech Challenge)
QUALITY_THRESHOLD = 7          # nota >= 7 -> alta qualidade
TARGET = "alta_qualidade"
CLASS_LABELS = {0: "Baixa/média", 1: "Alta"}

# Variáveis físico-químicas originais (após padronização dos nomes)
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

# Rótulos em português para gráficos, tabelas e apresentação
LABELS_PT = {
    "fixed_acidity": "Acidez fixa",
    "volatile_acidity": "Acidez volátil",
    "citric_acid": "Ácido cítrico",
    "residual_sugar": "Açúcar residual",
    "chlorides": "Cloretos",
    "free_sulfur_dioxide": "SO₂ livre",
    "total_sulfur_dioxide": "SO₂ total",
    "density": "Densidade",
    "ph": "pH",
    "sulphates": "Sulfatos",
    "alcohol": "Teor alcoólico",
    "quality": "Nota de qualidade",
    "alta_qualidade": "Alta qualidade",
    # variáveis criadas no feature engineering
    "total_acidity": "Acidez total",
    "free_so2_ratio": "Proporção SO₂ livre/total",
    "bound_sulfur_dioxide": "SO₂ combinado",
    "alcohol_x_sulphates": "Álcool × sulfatos",
}

# Unidades (dicionário de dados do dataset Wine Quality / UCI)
UNITS = {
    "fixed_acidity": "g/dm³ ác. tartárico",
    "volatile_acidity": "g/dm³ ác. acético",
    "citric_acid": "g/dm³",
    "residual_sugar": "g/dm³",
    "chlorides": "g/dm³ NaCl",
    "free_sulfur_dioxide": "mg/dm³",
    "total_sulfur_dioxide": "mg/dm³",
    "density": "g/cm³",
    "ph": "",
    "sulphates": "g/dm³ sulfato de potássio",
    "alcohol": "% vol.",
}


def label(col: str) -> str:
    """Retorna o rótulo em português de uma coluna (ou o próprio nome)."""
    return LABELS_PT.get(col, col)
