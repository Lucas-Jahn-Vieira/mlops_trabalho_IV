"""Monta o Relatório Executivo em PDF a partir dos resultados do pipeline."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem,
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="MainTitle", fontSize=16, leading=19, spaceAfter=2, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="SubTitle", fontSize=10, leading=12, spaceAfter=10, fontName="Helvetica", textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=14, spaceBefore=8, spaceAfter=3, fontName="Helvetica-Bold", textColor=colors.HexColor("#0b3d91")))
styles.add(ParagraphStyle(name="Body2", fontSize=9.7, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=3, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Bullet2", fontSize=9.7, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=1, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Caption2", fontSize=8.5, leading=10, alignment=1, fontName="Helvetica-Oblique", textColor=colors.HexColor("#444444"), spaceBefore=1, spaceAfter=6))


def _fmt_money(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _metrics_table(results):
    m_default = results["metrics_default"]
    m_optimal = results["metrics_optimal"]
    trivial = results["trivial"]
    opt_t = results["optimal_row"]["threshold"]

    header = ["Métrica", "Trivial (sempre 'Operacional')", "Threshold 0,5 (padrão)", f"Threshold {opt_t:.2f} (ótimo)"]
    rows = [
        header,
        ["Acurácia", f"{trivial['acuracia']:.4f}", f"{m_default['acuracia']:.4f}", f"{m_optimal['acuracia']:.4f}"],
        ["Precisão", f"{trivial['precisao']:.4f}", f"{m_default['precisao']:.4f}", f"{m_optimal['precisao']:.4f}"],
        ["Recall", f"{trivial['recall']:.4f}", f"{m_default['recall']:.4f}", f"{m_optimal['recall']:.4f}"],
        ["F1-Score", "—", f"{m_default['f1']:.4f}", f"{m_optimal['f1']:.4f}"],
        ["F2-Score", "—", f"{m_default['f2']:.4f}", f"{m_optimal['f2']:.4f}"],
        ["F0.5-Score", "—", f"{m_default['f0_5']:.4f}", f"{m_optimal['f0_5']:.4f}"],
        ["AUC-ROC", "—", f"{m_default['auc_roc']:.4f}", f"{m_optimal['auc_roc']:.4f}"],
        ["TP / TN / FP / FN", "—",
         f"{m_default['TP']} / {m_default['TN']} / {m_default['FP']} / {m_default['FN']}",
         f"{m_optimal['TP']} / {m_optimal['TN']} / {m_optimal['FP']} / {m_optimal['FN']}"],
    ]

    t = Table(rows, colWidths=[3.1*cm, 4.3*cm, 4.3*cm, 4.3*cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_report(results, prints_dir, out_path):
    story = []

    story.append(Paragraph("Relatório Executivo — Avaliação de Métricas em Modelos de Classificação", styles["MainTitle"]))
    story.append(Paragraph("Detecção de falhas em sensores industriais com classes fortemente desbalanceadas (99,5% Operacional / 0,5% Falha)", styles["SubTitle"]))

    # ------------------------------------------------------------------
    story.append(Paragraph("1. Arquitetura da Solução", styles["H2"]))
    story.append(Paragraph(
        f"O dataset sintético contém {results['n_total']:,} leituras de sensores industriais, com apenas "
        f"{results['n_falhas_total']:,} representando falhas reais (aproximadamente 0,5% do total). "
        "Antes de qualquer etapa de pré-processamento, os dados brutos foram divididos de forma estratificada em "
        f"treino ({results['n_train']:,} amostras) e teste ({results['n_test']:,} amostras, sendo {results['n_falhas_test']} falhas).",
        styles["Body2"]
    ))
    story.append(Paragraph(
        "O isolamento acontece em <b>src/data_pipeline.py</b>, na função <b>split_data()</b>, chamada antes de qualquer "
        "normalização. O escalonamento (StandardScaler) e o classificador (RandomForest) ficam encapsulados em um "
        "<b>sklearn.Pipeline</b> (<b>src/model.py</b>): ao chamar <b>pipeline.fit(X_train, y_train)</b>, a média e o "
        "desvio-padrão usados para normalizar os dados são calculados <b>somente</b> a partir do conjunto de treino. "
        "O conjunto de teste é apenas transformado com essas estatísticas já fixadas (<b>pipeline.predict_proba(X_test)</b>), "
        "nunca participando do ajuste. Isso evita Data Leakage: se o teste influenciasse o escalonamento ou o treino do "
        "modelo, as métricas reportadas pareceriam melhores do que o desempenho real esperado em produção.",
        styles["Body2"]
    ))
    story.append(Paragraph(
        f"Como ilustração da armadilha da Acurácia citada no enunciado: um classificador trivial que sempre prevê "
        f"'Operacional' atinge <b>{results['trivial']['acuracia']:.4f}</b> de acurácia, mas <b>Recall = "
        f"{results['trivial']['recall']:.4f}</b> — ou seja, não detecta nenhuma falha real. A Acurácia isolada mascara "
        "completamente essa falha crítica do modelo.",
        styles["Body2"]
    ))

    # ------------------------------------------------------------------
    story.append(Paragraph("2. Quadro Comparativo de Métricas", styles["H2"]))
    story.append(Paragraph(
        "Métricas calculadas sobre o conjunto de teste isolado, comparando o classificador trivial, o threshold padrão "
        "(0,5) e o threshold ótimo (definido na análise financeira da Seção 3):",
        styles["Body2"]
    ))
    story.append(_metrics_table(results))
    story.append(Spacer(1, 6))

    story.append(Image(f"{prints_dir}/curva_roc.png", width=8.6*cm, height=8.6*cm*(4.2/5.2)))
    story.append(Paragraph("Figura 1 — Curva ROC do modelo no conjunto de teste.", styles["Caption2"]))

    story.append(Image(f"{prints_dir}/matriz_confusao_threshold_padrao.png", width=6.6*cm, height=6.6*cm))
    story.append(Paragraph("Figura 2 — Matriz de confusão com threshold padrão (0,5).", styles["Caption2"]))

    story.append(Image(f"{prints_dir}/matriz_confusao_threshold_otimo.png", width=6.6*cm, height=6.6*cm))
    story.append(Paragraph(f"Figura 3 — Matriz de confusão com threshold ótimo ({results['optimal_row']['threshold']:.2f}).", styles["Caption2"]))

    # ------------------------------------------------------------------
    story.append(Paragraph("3. Análise Financeira do Threshold", styles["H2"]))
    story.append(Paragraph(
        "Custos assumidos para a simulação: R$ 50.000,00 por falha não detectada (Falso Negativo — parada não "
        "planejada ou risco de acidente) e R$ 800,00 por alarme falso (Falso Positivo — inspeção manual desnecessária). "
        "O threshold de decisão foi variado de 0,01 a 0,99 e o custo total (FN × custo_FN + FP × custo_FP) foi "
        "recalculado para cada valor.",
        styles["Body2"]
    ))
    story.append(Image(f"{prints_dir}/custo_vs_threshold.png", width=14.5*cm, height=14.5*cm*(4.4/7.2)))
    story.append(Paragraph("Figura 4 — Custo total estimado em função do threshold de decisão.", styles["Caption2"]))

    story.append(Paragraph(
        f"O <b>threshold padrão (0,5)</b> resulta em um custo total estimado de "
        f"<b>{_fmt_money(results['baseline_row']['custo_total'])}</b> "
        f"({int(results['baseline_row']['FN'])} falhas não detectadas e {int(results['baseline_row']['FP'])} alarmes falsos). "
        f"O <b>threshold ótimo ({results['optimal_row']['threshold']:.2f})</b> reduz o custo total para "
        f"<b>{_fmt_money(results['optimal_row']['custo_total'])}</b> "
        f"({int(results['optimal_row']['FN'])} falhas não detectadas e {int(results['optimal_row']['FP'])} alarmes falsos), "
        f"uma economia de <b>{_fmt_money(results['savings'])}</b> ({results['savings_pct']:.1f}%) em relação ao padrão. "
        "O threshold ótimo é mais baixo que 0,5 porque o custo de uma falha não detectada é muito maior que o custo "
        "de um alarme falso, favorecendo um modelo mais sensível (maior Recall) mesmo à custa de mais falsos positivos.",
        styles["Body2"]
    ))

    # ------------------------------------------------------------------
    story.append(Paragraph("4. Recomendações MLOps para Monitoramento em Produção", styles["H2"]))
    bullets = [
        "<b>Não monitorar Acurácia isoladamente:</b> acompanhar Precisão, Recall, F2-Score (prioriza Recall, "
        "adequado quando o custo de FN é alto) e AUC-ROC em janelas móveis (ex.: diária/semanal), com alertas "
        "quando qualquer uma cair abaixo de um limiar mínimo definido em conjunto com a operação da planta.",
        "<b>Monitorar Data/Concept Drift:</b> comparar a distribuição das features de entrada em produção com a "
        "distribuição do conjunto de treino (ex.: PSI, KS-test), já que sensores industriais podem se degradar ou "
        "ser recalibrados ao longo do tempo, alterando a distribuição sem alterar o código.",
        "<b>Reavaliar o threshold periodicamente:</b> os custos de FN e FP usados na Seção 3 podem mudar (novo "
        "contrato de manutenção, novo turno, nova linha de produção); o threshold ótimo deve ser recalculado "
        "sempre que esses custos forem atualizados, não apenas uma vez no lançamento do modelo.",
        "<b>Registrar previsões e desfechos reais (feedback loop):</b> armazenar cada previsão junto ao "
        "diagnóstico real posterior (falha confirmada ou não) em um log estruturado, permitindo recalcular a "
        "matriz de confusão real de produção e comparar com a validação offline.",
        "<b>Retreinamento programado e versionamento:</b> retreinar o modelo periodicamente (ou por gatilho de "
        "queda de métrica/drift) usando um pipeline reprodutível, versionando dados, modelo e threshold em um "
        "model registry, com testes automatizados de regressão de métricas antes de promover uma nova versão.",
        "<b>Implantação gradual (shadow/canary):</b> validar um novo modelo ou threshold em modo sombra (rodando "
        "em paralelo sem agir) ou em rollout parcial antes de substituir o modelo em produção, reduzindo o risco "
        "de uma regressão silenciosa em uma métrica crítica.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, styles["Bullet2"]), leftIndent=10) for b in bullets],
                               bulletType="bullet", leftIndent=8, spaceBefore=0, bulletFontSize=7))

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.9*cm, rightMargin=1.9*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
        title="Relatório Executivo - Avaliação de Métricas em Modelos de Classificação",
    )
    doc.build(story)
