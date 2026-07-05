#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py - Etapa 3: análise estatística e geração de tabelas/figuras.

A partir de results/metrics/meteor_scores.csv produz:
  - results/tables/tabela_acuracia.tex ....... média METEOR por modelo x estratégia
  - results/tables/tabela_variabilidade.tex .. média, DP e CV por modelo x estratégia
  - results/tables/resultado_testes.txt ...... Shapiro-Wilk + ANOVA de medidas
                                               repetidas (ou Friedman) e post-hoc
  - results/figures/boxplot_meteor.png ....... distribuição por modelo/estratégia

Etapa 100% determinística a partir do CSV de entrada.
"""
from pathlib import Path

import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "metrics" / "meteor_scores.csv"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = {"gemini": "Gemini 2.5 Flash", "gpt-4o-mini": "GPT-4.1 Mini", "deepseek": "DeepSeek R1-0528"}
STRAT_LABEL = {"zero_shot": "Zero-shot", "one_shot": "One-shot", "few_shot": "Few-shot"}
STRAT_ORDER = ["zero_shot", "one_shot", "few_shot"]


def main():
    df = pd.read_csv(CSV)

    # ---------- Tabela 1: acurácia média (média sobre casos e repetições) ----------
    acc = (df.groupby(["model", "strategy"])["meteor"].mean().unstack()[STRAT_ORDER])
    with open(TAB / "tabela_acuracia.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n"
                "\\caption{Escores METEOR médios por LLM e estratégia de prompting.}\n"
                "\\label{tab:acuracia}\n\\begin{tabular}{lccc}\n\\hline\n"
                "\\textbf{Modelo} & \\textbf{Zero-shot} & \\textbf{One-shot} & \\textbf{Few-shot} \\\\\n\\hline\n")
        for m in acc.index:
            f.write(f"{MODEL_LABEL.get(m, m)} & "
                    + " & ".join(f"{acc.loc[m, s]:.2f}" for s in STRAT_ORDER) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ---------- Tabela 2: variabilidade run-to-run (CV médio entre repetições) ----------
    per_case = df.groupby(["model", "strategy", "case_id"])["meteor"].agg(["mean", "std"])
    per_case["cv"] = (per_case["std"] / per_case["mean"]).fillna(0) * 100
    var = per_case.groupby(["model", "strategy"]).agg(
        media=("mean", "mean"), dp=("std", "mean"), cv=("cv", "mean"))
    with open(TAB / "tabela_variabilidade.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n"
                "\\caption{Média, desvio-padrão e coeficiente de variação (CV) dos escores METEOR entre repetições.}\n"
                "\\label{tab:variabilidade}\n\\begin{tabular}{llccc}\n\\hline\n"
                "\\textbf{Modelo} & \\textbf{Estratégia} & \\textbf{Média} & \\textbf{DP} & \\textbf{CV (\\%)} \\\\\n\\hline\n")
        for (m, s) in [(m, s) for m in var.index.get_level_values(0).unique() for s in STRAT_ORDER]:
            if (m, s) in var.index:
                r = var.loc[(m, s)]
                f.write(f"{MODEL_LABEL.get(m, m)} & {STRAT_LABEL[s]} & "
                        f"{r['media']:.2f} & {r['dp']:.3f} & {r['cv']:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ---------- Testes estatísticos (sobre a média por caso, 1ª repetição p/ acurácia) ----------
    lines = []
    pivot = (df.groupby(["case_id", "model", "strategy"])["meteor"].mean()
               .reset_index())
    pivot["cond"] = pivot["model"] + "_" + pivot["strategy"]
    wide = pivot.pivot(index="case_id", columns="cond", values="meteor")

    cond_label = {f"{m}_{s}": f"{MODEL_LABEL[m]} - {STRAT_LABEL[s]}" for m in MODEL_LABEL for s in STRAT_ORDER}

    lines.append("== Shapiro-Wilk (normalidade por condicao) ==")
    normal = True
    for c in wide.columns:
        stat, p = sps.shapiro(wide[c])
        normal &= p > 0.05
        lines.append(f"  {cond_label.get(c, c)}: W={stat:.3f}, p={p:.4f}")

    lines.append("")
    if normal:
        lines.append("== ANOVA de Medidas Repetidas (fatores: modelo, estrategia) ==")
        try:
            from statsmodels.stats.anova import AnovaRM
            long = pivot.rename(columns={"case_id": "subject"})
            res = AnovaRM(long, "meteor", "subject",
                          within=["model", "strategy"]).fit()
            lines.append(str(res))
        except Exception as e:
            lines.append(f"  (AnovaRM indisponivel: {e}; usando Friedman)")
            normal = False
    if not normal:
        lines.append("== Teste de Friedman (nao parametrico) ==")
        stat, p = sps.friedmanchisquare(*[wide[c] for c in wide.columns])
        lines.append(f"  chi2={stat:.3f}, p={p:.4f}")

    # Post-hoc: Wilcoxon pareado com correcao de Bonferroni entre estrategias (por modelo)
    lines.append("")
    lines.append("== Post-hoc: Wilcoxon pareado (Bonferroni) entre estrategias, por modelo ==")
    from itertools import combinations
    for m in df["model"].unique():
        pairs = list(combinations(STRAT_ORDER, 2))
        for s1, s2 in pairs:
            a = wide[f"{m}_{s1}"]
            b = wide[f"{m}_{s2}"]
            try:
                stat, p = sps.wilcoxon(a, b)
                p_adj = min(p * len(pairs), 1.0)
                lines.append(f"  {MODEL_LABEL.get(m, m)}: {STRAT_LABEL[s1]} vs {STRAT_LABEL[s2]}: p={p:.4f}, p_adj={p_adj:.4f}")
            except ValueError as e:
                lines.append(f"  {MODEL_LABEL.get(m, m)}: {STRAT_LABEL[s1]} vs {STRAT_LABEL[s2]}: {e}")

    (TAB / "resultado_testes.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------- Boxplot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, s in zip(axes, STRAT_ORDER):
        sub = df[df["strategy"] == s]
        data = [sub[sub["model"] == m]["meteor"] for m in MODEL_LABEL]
        ax.boxplot(data, tick_labels=[MODEL_LABEL[m] for m in MODEL_LABEL])
        ax.set_title(STRAT_LABEL[s])
        ax.set_ylabel("METEOR")
    fig.tight_layout()
    fig.savefig(FIG / "boxplot_meteor.png", dpi=200)

    print("Tabelas em results/tables/, figura em results/figures/.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
