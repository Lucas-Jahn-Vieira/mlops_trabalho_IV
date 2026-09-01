"""Geração de dados sintéticos de sensores industriais desbalanceados
e isolamento do conjunto de teste antes de qualquer pré-processamento."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FAILURE_RATE = 0.005
N_SAMPLES = 100_000

SENSOR_NAMES = [
    "vibracao_rms",
    "temperatura_motor",
    "pressao_hidraulica",
    "corrente_eletrica",
    "velocidade_esteira",
    "ruido_acustico",
    "torque_eixo",
    "temperatura_rolamento",
    "umidade_ambiente",
    "horas_desde_manutencao",
]


def generate_synthetic_data(n_samples=N_SAMPLES, failure_rate=FAILURE_RATE, random_state=RANDOM_STATE):
    """Gera um dataset sintético de telemetria industrial com 0,5% de falhas."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(SENSOR_NAMES),
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[1 - failure_rate, failure_rate],
        flip_y=0.004,
        class_sep=1.55,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=SENSOR_NAMES)
    y_s = pd.Series(y, name="falha")
    return X_df, y_s


def split_data(X, y, test_size=0.3, random_state=RANDOM_STATE):
    """Isola o conjunto de teste ANTES de qualquer pré-processamento (scaling, imputação etc.).

    A divisão estratificada acontece sobre os dados brutos; nenhuma estatística
    (média, desvio-padrão, etc.) é calculada a partir do conjunto de teste, o que
    evita Data Leakage no pipeline.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
