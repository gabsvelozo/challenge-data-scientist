# Monitoramento de Modelos

Solução do desafio técnico de Ciência de Dados: API para avaliação de **performance** e **aderência** de um modelo de crédito pré-treinado, construída com **FastAPI + Uvicorn**.

## Estrutura

```
monitoring/
├── app/
│   ├── main.py                       # cria a FastAPI, inclui os routers com prefix="/v1"
│   ├── api/
│   │   ├── routers.py                # agrega os routers de performance e aderência
│   │   ├── schemas.py                # modelos Pydantic (PerformanceRecord derivado do próprio model.pkl)
│   │   └── endpoints/
│   │       ├── performance.py        # POST /v1/performance
│   │       └── aderencia.py          # POST /v1/aderencia
│   ├── core/
│   │   ├── model.py                  # carregamento e scoring do model.pkl
│   │   └── datasets.py               # leitura segura dos datasets locais
│   └── requirements.txt
├── model.pkl                         # modelo pré-treinado (fornecido pelo desafio)
├── batch_records.json                # lote de exemplo pro endpoint de performance (fornecido)
├── environment.yml                   # alternativa via Conda
└── notebook_demonstracao.ipynb       # chamadas reais aos dois endpoints + análise visual
```

`model.pkl`, `batch_records.json` e os datasets em `datasets/credit_01/` (`train.gz`, `test.gz`, `oot.gz`) são fixtures do desafio original, incluídas neste repositório pra ele rodar de forma autocontida (clonar e seguir "Como executar" abaixo, sem precisar buscar nada em outro lugar).

## Como executar

`model.pkl` foi salvo com **scikit-learn==1.0.2**, que no Windows só tem wheel pré-compilada até **Python 3.10** (Python 3.11+ não instala essa versão via pip, e `numpy>=2.0` quebra a leitura do pickle com `ValueError: numpy.dtype size changed`). Por isso o ambiente precisa ser Python 3.10 especificamente.

**Opção A: venv**
```bash
py -3.10 -m venv monitoring/.venv
monitoring/.venv/Scripts/pip install -r monitoring/app/requirements.txt   # Windows
# source monitoring/.venv/bin/activate && pip install -r monitoring/app/requirements.txt   # Linux/Mac
```

**Opção B: Conda**
```bash
conda env create -f monitoring/environment.yml
conda activate monitoring-credit-model
```

**Subir a API** (a partir da pasta `monitoring/`):
```bash
cd monitoring
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8001
```

Documentação interativa (Swagger): `http://127.0.0.1:8001/docs`.

## Endpoints

### `POST /v1/performance`

Recebe um lote de registros (as 118 variáveis do modelo, mais `REF_DATE` e `TARGET`) e retorna a volumetria mensal e a performance (AUC-ROC) do modelo nesse lote.

Body (lista de registros, formato completo em `batch_records.json`):
```json
[
  {
    "VAR2": "M", "IDADE": 43.893, "VAR5": "PR", "...": "...",
    "REF_DATE": "2017-03-25 00:00:00+00:00",
    "TARGET": 1
  }
]
```

Resposta:
```json
{
  "n_records": 500,
  "volumetria": { "2017-01": 58, "2017-02": 55, "...": "..." },
  "auc_roc": 0.5751748251748252
}
```

### `POST /v1/aderencia`

Recebe o caminho local de um dataset `.gz`, escora essa base e a base fixa de Teste (`../datasets/credit_01/test.gz`) com o `model.pkl`, e calcula a distância entre as duas distribuições de score com o teste de Kolmogorov-Smirnov.

Body:
```json
{ "dataset_path": "../datasets/credit_01/oot.gz" }
```

Resposta:
```json
{
  "dataset_path": "../datasets/credit_01/oot.gz",
  "reference_dataset_path": ".../datasets/credit_01/test.gz",
  "n_records": 91965,
  "n_records_reference": 51751,
  "ks_statistic": 0.021736133923338036,
  "ks_pvalue": 5.0522344570608844e-14,
  "score_mean": 0.7956362963664461,
  "score_mean_reference": 0.7923805226829322,
  "unknown_categories": { "VAR121": 3 }
}
```

## Decisões técnicas

- **Schema dinâmico**: `PerformanceRecord` é construído em tempo de execução a partir de `model.feature_names_in_` e da divisão numérica/categórica do próprio `ColumnTransformer` fitado (`core/model.py::get_feature_metadata`). O contrato da API nunca diverge do que o pipeline realmente espera, mesmo sem digitar as 118 features à mão.
- **`np.nan` explícito**: nulos são forçados com `df.where(pd.notnull(df), np.nan)`, porque `None` do Python em colunas `object` nem sempre é reconhecido como ausente pelos imputadores do sklearn.
- **Categorias desconhecidas**: o `OneHotEncoder` do pipeline tem `handle_unknown="error"`. Testando com `oot.gz` de verdade, isso quebrava o `predict_proba` (categoria `"MUITO PROXIMO"`, nunca vista no treino). A correção foi sanitizar essas categorias para `NaN` antes do pipeline, deixando o imputador de moda já fitado preenchê-las, sem alterar o `model.pkl`. As contagens são devolvidas em `/v1/aderencia` como sinal adicional de deriva.
- **AUC-ROC**: métrica padrão para performance de classificadores binários por probabilidade, invariante a threshold, adequada porque o endpoint expõe `predict_proba` e não uma classe binária.
- **KS via `scipy.stats.ks_2samp`**: métrica padrão de mercado em crédito para comparar distribuições de score, sem assumir forma paramétrica. O p-valor também é retornado, pra indicar se a diferença é estatisticamente significante, não só o tamanho do KS.
- **Volumetria e performance no mesmo endpoint**: respondem juntas à pergunta "o que aconteceu neste lote?"; a volumetria contextualiza a confiabilidade da AUC (poucos registros ou poucos positivos tornam a AUC instável).
- **Validação de `dataset_path`**: o caminho é resolvido relativo a `monitoring/` e checado para não sair da raiz do repositório (guarda contra path traversal), retornando `400` para caminho inválido em vez de um erro `500` não tratado.

## Resultado observado

| Endpoint | Base | Métrica | Valor |
|---|---|---|---|
| `/v1/performance` | `batch_records.json` (500 registros) | AUC-ROC | 0.575 |
| `/v1/aderencia` | `train.gz` vs `test.gz` | KS / p-valor | 0.0019 / 0.999, sem drift |
| `/v1/aderencia` | `oot.gz` vs `test.gz` | KS / p-valor | 0.0217 / ~5e-14, com drift |

`train.gz` é estatisticamente indistinguível da base de Teste (mesmo período de coleta). `oot.gz`, coletada em meses posteriores, mostra deriva significativa na distribuição de score, o comportamento esperado de monitoramento capturando drift ao longo do tempo. Detalhes e gráficos no [`monitoring/notebook_demonstracao.ipynb`](monitoring/notebook_demonstracao.ipynb).
