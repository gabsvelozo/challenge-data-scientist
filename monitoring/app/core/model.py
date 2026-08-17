"""Carregamento e scoring com o modelo de crédito pré-treinado (model.pkl)"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parents[2] / "model.pkl"


@dataclass(frozen=True)
class FeatureMetadata:
    # Basicamente tudo que a API precisa saber sobre os inputs do pipeline fitado
    feature_names: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    known_categories: Dict[str, set]


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    # Carrega o model.pkl uma vez por processo
    import pickle

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_feature_metadata() -> FeatureMetadata:
    """Introspecciona o ColumnTransformer fitado pra saber as colunas esperadas.
    simple_preprocessing é um ColumnTransformer com dois ramos:
    pipeline-1 (imputação por mediana) para colunas numéricas e pipeline-2 (imputação por moda + OneHotEncoder) para categóricas
    """
    model = load_model()
    column_transformer = model.named_steps["simple_preprocessing"]

    numeric_columns = list(column_transformer.transformers_[0][2])
    categorical_columns = list(column_transformer.transformers_[1][2])

    encoder = column_transformer.named_transformers_["pipeline-2"].named_steps["encoder"]
    known_categories = {
        col: set(cats) for col, cats in zip(categorical_columns, encoder.categories_)
    }

    return FeatureMetadata(
        feature_names=list(model.feature_names_in_),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        known_categories=known_categories,
    )


def sanitize_unknown_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # Substitui categorias fora do vocabulário de treino por NaN antes de escorar, o OneHotEncoder fitado tem handle_unknown="error"
    metadata = get_feature_metadata()
    df = df.copy()
    unknown_counts: Dict[str, int] = {}

    for col, known in metadata.known_categories.items():
        mask = df[col].notna() & ~df[col].isin(known)
        n_unknown = int(mask.sum())
        if n_unknown:
            unknown_counts[col] = n_unknown
            df.loc[mask, col] = np.nan

    return df, unknown_counts


def score(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    # Escora um DataFrame com o modelo pré-treinado
    metadata = get_feature_metadata()
    model = load_model()

    X = df[metadata.feature_names]
    X = X.where(pd.notnull(X), np.nan)
    X, unknown_counts = sanitize_unknown_categories(X)

    proba = model.predict_proba(X)[:, 1]
    return proba, unknown_counts
