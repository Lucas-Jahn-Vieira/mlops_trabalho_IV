"""Ajuste dinâmico do threshold de decisão a partir de custos operacionais
e financeiros simulados de uma planta fabril."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custos assumidos para a simulação (podem ser recalibrados com dados reais da planta)
CUSTO_FALHA_NAO_DETECTADA = 50_000.0   # Falso Negativo: parada não planejada / acidente
CUSTO_ALARME_FALSO = 800.0             # Falso Positivo: inspeção manual desnecessária

THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


def total_cost(y_true, y_proba, threshold, cost_fn=CUSTO_FALHA_NAO_DETECTADA, cost_fp=CUSTO_ALARME_FALSO):
    y_pred = (y_proba >= threshold).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn * cost_fn + fp * cost_fp, fn, fp


def cost_curve(y_true, y_proba, thresholds=THRESHOLDS, cost_fn=CUSTO_FALHA_NAO_DETECTADA, cost_fp=CUSTO_ALARME_FALSO):
    rows = []
    for t in thresholds:
        custo, fn, fp = total_cost(y_true, y_proba, t, cost_fn, cost_fp)
        rows.append({"threshold": t, "custo_total": custo, "FN": fn, "FP": fp})
    return pd.DataFrame(rows)


def find_optimal_threshold(cost_df):
    return cost_df.loc[cost_df["custo_total"].idxmin()]


def plot_cost_vs_threshold(cost_df, optimal_row, baseline_threshold, out_path):
    baseline_row = cost_df.iloc[(cost_df["threshold"] - baseline_threshold).abs().idxmin()]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(cost_df["threshold"], cost_df["custo_total"], color="#0b3d91", linewidth=2)
    ax.axvline(optimal_row["threshold"], color="#1a9850", linestyle="--", linewidth=1.3,
               label=f"Threshold ótimo = {optimal_row['threshold']:.2f}")
    ax.axvline(baseline_threshold, color="#d73027", linestyle="--", linewidth=1.3,
               label=f"Threshold padrão = {baseline_threshold:.2f}")
    ax.scatter([optimal_row["threshold"]], [optimal_row["custo_total"]], color="#1a9850", zorder=5)
    ax.scatter([baseline_row["threshold"]], [baseline_row["custo_total"]], color="#d73027", zorder=5)

    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Custo Total Estimado (R$)")
    ax.set_title("Custo Total vs. Threshold de Decisão")
    ax.legend(loc="upper center")
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return baseline_row
