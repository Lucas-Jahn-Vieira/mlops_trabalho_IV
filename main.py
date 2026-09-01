"""Ponto de entrada: executa o pipeline completo e gera o relatório em PDF."""

import os

from src.run_pipeline import run_full_pipeline
from src.generate_report import build_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRINTS_DIR = os.path.join(BASE_DIR, "prints")
OUT_PDF = os.path.join(BASE_DIR, "Relatorio_Trabalho_IV_Metricas_Classificacao.pdf")

if __name__ == "__main__":
    print("Executando pipeline (geração de dados, treino, métricas, análise de threshold)...")
    results = run_full_pipeline(PRINTS_DIR)

    print(f"Total de amostras: {results['n_total']:,} | Falhas: {results['n_falhas_total']:,}")
    print(f"Treino: {results['n_train']:,} | Teste: {results['n_test']:,} (falhas no teste: {results['n_falhas_test']})")
    print("Métricas @ threshold 0,5:", results["metrics_default"])
    print("Métricas @ threshold ótimo:", results["metrics_optimal"])
    print(f"Threshold ótimo: {results['optimal_row']['threshold']:.2f} | "
          f"Economia vs. padrão: R$ {results['savings']:.2f} ({results['savings_pct']:.1f}%)")

    print("Gerando relatório PDF...")
    build_report(results, PRINTS_DIR, OUT_PDF)
    print(f"Relatório salvo em: {OUT_PDF}")
