"""Orquestra o pipeline completo: geração de dados, isolamento de teste,
treinamento, métricas e análise financeira do threshold."""

import os

from src.data_pipeline import generate_synthetic_data, split_data
from src.model import train, predict_proba_failure
from src.metrics_utils import (
    compute_metrics, trivial_baseline_metrics, plot_confusion_matrix, plot_roc_curve,
)
from src.threshold_analysis import cost_curve, find_optimal_threshold, plot_cost_vs_threshold

BASELINE_THRESHOLD = 0.5


def run_full_pipeline(prints_dir):
    os.makedirs(prints_dir, exist_ok=True)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = train(X_train, y_train)
    y_proba = predict_proba_failure(pipeline, X_test)

    y_pred_default = (y_proba >= BASELINE_THRESHOLD).astype(int)
    metrics_default = compute_metrics(y_test, y_pred_default, y_proba)
    trivial = trivial_baseline_metrics(y_test)

    plot_confusion_matrix(
        y_test, y_pred_default, "Matriz de Confusão — Threshold Padrão (0,5)",
        os.path.join(prints_dir, "matriz_confusao_threshold_padrao.png"),
    )
    plot_roc_curve(y_test, y_proba, os.path.join(prints_dir, "curva_roc.png"))

    cost_df = cost_curve(y_test, y_proba)
    optimal_row = find_optimal_threshold(cost_df)
    baseline_row = plot_cost_vs_threshold(
        cost_df, optimal_row, BASELINE_THRESHOLD,
        os.path.join(prints_dir, "custo_vs_threshold.png"),
    )

    y_pred_optimal = (y_proba >= optimal_row["threshold"]).astype(int)
    metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_proba)
    plot_confusion_matrix(
        y_test, y_pred_optimal, f"Matriz de Confusão — Threshold Ótimo ({optimal_row['threshold']:.2f})",
        os.path.join(prints_dir, "matriz_confusao_threshold_otimo.png"),
    )

    savings = baseline_row["custo_total"] - optimal_row["custo_total"]
    savings_pct = (savings / baseline_row["custo_total"] * 100) if baseline_row["custo_total"] > 0 else 0.0

    return {
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_falhas_total": int(y.sum()),
        "n_falhas_test": int(y_test.sum()),
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "trivial": trivial,
        "baseline_row": baseline_row,
        "optimal_row": optimal_row,
        "savings": savings,
        "savings_pct": savings_pct,
        "baseline_threshold": BASELINE_THRESHOLD,
    }
