"""Cálculo e visualização das métricas de classificação para cenários
fortemente desbalanceados."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    fbeta_score, roc_auc_score, roc_curve,
)


def compute_metrics(y_true, y_pred, y_proba):
    """Calcula Acurácia, Matriz de Confusão, Precisão, Recall, F1, F2, F0.5 e AUC-ROC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "f0_5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
    }


def trivial_baseline_metrics(y_true):
    """Métricas de um classificador ingênuo que sempre prevê 'Operacional' (classe 0).

    Ilustra a armadilha da Acurácia em cenários desbalanceados: acurácia alta,
    mas recall nulo (nenhuma falha é detectada).
    """
    y_pred_trivial = np.zeros_like(y_true)
    return {
        "acuracia": accuracy_score(y_true, y_pred_trivial),
        "recall": recall_score(y_true, y_pred_trivial, zero_division=0),
        "precisao": precision_score(y_true, y_pred_trivial, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Operacional", "Falha"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Operacional", "Falha"])
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#0b3d91", linewidth=2, label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Classificador aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
