"""Modelos Pydantic dos endpoints /v1/performance e /v1/aderencia."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, conint, create_model
from ..core.model import get_feature_metadata

_metadata = get_feature_metadata()

_dynamic_fields = {
    **{col: (Optional[float], None) for col in _metadata.numeric_columns},
    **{col: (Optional[str], None) for col in _metadata.categorical_columns},
}

PerformanceRecord = create_model(
    "PerformanceRecord",
    REF_DATE=(datetime, Field(..., description="Data de referência do registro")),
    TARGET=(conint(ge=0, le=1), Field(..., description="Alvo observado (0 ou 1)")),
    **_dynamic_fields,
)
PerformanceRecord.__doc__ = (
    "Um registro a ser escorado: as 118 features do modelo (todas opcionais, "
    "já que os imputadores do pipeline tratam valores ausentes) mais os dois "
    "campos de metadado REF_DATE e TARGET, que não são features do modelo."
)


class PerformanceResponse(BaseModel):
    n_records: int = Field(..., description="Total de registros recebidos")
    volumetria: Dict[str, int] = Field(
        ..., description="Quantidade de registros por mês (YYYY-MM), a partir de REF_DATE"
    )
    auc_roc: float = Field(
        ..., description="AUC-ROC das probabilidades preditas pelo modelo vs TARGET"
    )


class AderenciaRequest(BaseModel):
    dataset_path: str = Field(
        ...,
        description="Caminho de um dataset .gz local, relativo a monitoring/ "
        '(ex: "../datasets/credit_01/oot.gz")',
    )


class AderenciaResponse(BaseModel):
    dataset_path: str
    reference_dataset_path: str
    n_records: int
    n_records_reference: int
    ks_statistic: float = Field(..., description="Distância KS entre as duas distribuições de score")
    ks_pvalue: float = Field(
        ..., description="p-valor da hipótese nula de que os dois scores vêm da mesma distribuição"
    )
    score_mean: float
    score_mean_reference: float
    unknown_categories: Dict[str, int] = Field(
        default_factory=dict,
        description="Valores categóricos fora do vocabulário de treino do modelo, "
        "por coluna (evidência de deriva categórica, substituídos por NaN antes de escorar)",
    )
