"""Leitura dos datasets locais de credit_01 usados pelo /v1/aderencia"""
from pathlib import Path

import numpy as np
import pandas as pd

MONITORING_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = MONITORING_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"

# Base de referência fixa para aderência: o split de Teste original da modelagem
REFERENCE_DATASET_PATH = DATASETS_DIR / "credit_01" / "test.gz"


def resolve_dataset_path(raw_path: str) -> Path:
    candidate = (MONITORING_DIR / raw_path).resolve()

    if REPO_ROOT != candidate and REPO_ROOT not in candidate.parents:
        raise ValueError(f"'{raw_path}' resolve para fora do repositório")

    if not candidate.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: '{raw_path}' (resolvido para {candidate})")

    return candidate


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    return df.where(pd.notnull(df), np.nan)
