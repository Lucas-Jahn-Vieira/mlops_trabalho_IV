"""Pipeline de modelagem: o scaler é ajustado (fit) somente nos dados de
treino; o conjunto de teste é apenas transformado, nunca usado para
calcular estatísticas do pré-processamento."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_pipeline(random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def train(X_train, y_train, random_state=RANDOM_STATE):
    """Ajusta o pipeline (scaler + classificador) usando apenas dados de treino."""
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_proba_failure(pipeline, X):
    """Retorna a probabilidade prevista da classe 'Falha' (classe 1)."""
    return pipeline.predict_proba(X)[:, 1]
