# Trabalho IV — Avaliação de Métricas em Modelos de Classificação

Pipeline de dados e modelagem para detecção de falhas em sensores industriais
com classes fortemente desbalanceadas (99,5% Operacional / 0,5% Falha),
incluindo isolamento do conjunto de teste antes do pré-processamento,
comparação de métricas de classificação e ajuste dinâmico do threshold de
decisão a partir de custos financeiros simulados.

## Estrutura

```
trabalho_IV/
├── main.py                        # ponto de entrada: roda o pipeline e gera o PDF
├── requirements.txt
├── src/
│   ├── data_pipeline.py           # geração dos dados sintéticos e split treino/teste
│   ├── model.py                   # pipeline de pré-processamento + classificador
│   ├── metrics_utils.py           # matriz de confusão, precisão, recall, F-beta, AUC-ROC
│   ├── threshold_analysis.py      # curva de custo e busca do threshold ótimo
│   └── generate_report.py         # montagem do relatório em PDF
├── prints/                        # gráficos gerados (matrizes de confusão, ROC, custo)
└── Relatorio_Trabalho_IV_Metricas_Classificacao.pdf
```

## Como executar

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

O script gera os gráficos em `prints/` e o relatório final em
`Relatorio_Trabalho_IV_Metricas_Classificacao.pdf`.

## Prevenção de Data Leakage

Os dados brutos são divididos em treino e teste (`src/data_pipeline.py`,
`split_data()`) antes de qualquer normalização. O `StandardScaler` e o
`RandomForestClassifier` ficam dentro de um único `sklearn.Pipeline`
(`src/model.py`), de forma que as estatísticas de escalonamento são
calculadas apenas com `X_train` (`pipeline.fit`); o conjunto de teste é
somente transformado com essas estatísticas já fixadas.

## Custos assumidos na análise de threshold

- Falha não detectada (Falso Negativo): R$ 50.000,00
- Alarme falso (Falso Positivo): R$ 800,00

Esses valores são simulados para fins didáticos e devem ser recalibrados com
dados reais de custo de parada e de inspeção da planta.
