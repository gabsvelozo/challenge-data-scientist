"""Endpoint para cálculo de Performance."""
from typing import List

import pandas as pd
from fastapi import APIRouter
from sklearn.metrics import roc_auc_score

from ...core.model import score
from ..schemas import PerformanceRecord, PerformanceResponse

router = APIRouter(prefix="/performance")


@router.post("", response_model=PerformanceResponse)
def compute_performance(records: List[PerformanceRecord]) -> PerformanceResponse:
    """Volumetria mensal + AUC-ROC do model.pkl sobre os registros recebidos.

    Volumetria e performance são calculadas juntas porque respondem a uma
    mesma pergunta de monitoramento, "o que aconteceu neste lote?", e um
    consumidor (dashboard, alerta) normalmente quer as duas lado a lado: a
    volumetria contextualiza se a AUC calculada é confiável (poucos
    registros e/ou poucos positivos tornam a AUC instável).
    """
    df = pd.DataFrame([record.dict() for record in records])

    ref_month = df["REF_DATE"].dt.to_period("M").astype(str)
    volumetria = ref_month.value_counts().sort_index().to_dict()

    proba, _unknown_categories = score(df)
    auc_roc = roc_auc_score(y_true=df["TARGET"], y_score=proba)

    return PerformanceResponse(
        n_records=len(df),
        volumetria=volumetria,
        auc_roc=auc_roc,
    )
