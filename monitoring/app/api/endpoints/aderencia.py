"""Endpoint para cálculo de aderência."""
from fastapi import APIRouter, HTTPException
from scipy.stats import ks_2samp

from ...core.datasets import REFERENCE_DATASET_PATH, read_dataset, resolve_dataset_path
from ...core.model import score
from ..schemas import AderenciaRequest, AderenciaResponse

router = APIRouter(prefix="/aderencia")


@router.post("", response_model=AderenciaResponse)
def compute_aderencia(request: AderenciaRequest) -> AderenciaResponse:
    """KS entre o score do dataset informado e o score da base de Teste.

    KS (Kolmogorov-Smirnov) foi escolhido porque é a métrica padrão de
    mercado para comparar distribuições de score de crédito: não assume
    forma da distribuição, funciona bem com scores concentrados numa faixa
    (como os deste modelo, média ~0.79) e sua escala (0 a 1, distância
    máxima entre as CDFs) é a mesma usada para avaliar separação de score
    entre bons/maus pagadores, então times de risco já sabem interpretá-la.
    """
    try:
        input_path = resolve_dataset_path(request.dataset_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    df_input = read_dataset(input_path)
    df_reference = read_dataset(REFERENCE_DATASET_PATH)

    proba_input, unknown_input = score(df_input)
    proba_reference, unknown_reference = score(df_reference)

    ks_result = ks_2samp(proba_input, proba_reference)

    unknown_categories = dict(unknown_input)
    for col, count in unknown_reference.items():
        unknown_categories[col] = unknown_categories.get(col, 0) + count

    return AderenciaResponse(
        dataset_path=request.dataset_path,
        reference_dataset_path=str(REFERENCE_DATASET_PATH),
        n_records=len(df_input),
        n_records_reference=len(df_reference),
        ks_statistic=ks_result.statistic,
        ks_pvalue=ks_result.pvalue,
        score_mean=proba_input.mean(),
        score_mean_reference=proba_reference.mean(),
        unknown_categories=unknown_categories,
    )
