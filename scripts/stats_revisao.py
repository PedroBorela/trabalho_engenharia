#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_revisao.py - Análises complementares solicitadas na revisão do relatório.

A partir dos MESMOS dados brutos já coletados (results/raw/generations.jsonl e
results/metrics/meteor_scores.csv), produz de forma determinística:

  1. results/tables/tabela_estabilidade_por_caso.tex
     Distribuição do CV por caso de teste (mediana, IQR, máximo) por modelo e
     estratégia — complementa a Tabela 3, que agregava a variabilidade.
  2. results/tables/tabela_posthoc.tex
     Comparações post-hoc por modelo: p bruto, p ajustado (Bonferroni),
     diferença mediana de METEOR e tamanho de efeito (rank-biserial pareado).
  3. results/figures/boxplot_meteor_v2.png
     Versão da figura com legibilidade melhorada (fonte, contraste, resolução).

Nenhuma nova geração de LLM é realizada.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "metrics" / "meteor_scores.csv"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

MODEL_LABEL = {"gemini": "Gemini 2.5 Flash", "gpt-4o-mini": "GPT-4.1 Mini", "deepseek": "DeepSeek R1-0528"}
STRAT_LABEL = {"zero_shot": "Zero-shot", "one_shot": "One-shot", "few_shot": "Few-shot"}
STRAT_ORDER = ["zero_shot", "one_shot", "few_shot"]

df = pd.read_csv(CSV)
models = sorted(df["model"].unique())

# --------- 1. Estabilidade por caso: mediana, IQR e máximo do CV ---------
per_case = df.groupby(["model", "strategy", "case_id"])["meteor"].agg(["mean", "std"]).reset_index()
per_case["cv"] = (per_case["std"] / per_case["mean"]).fillna(0) * 100

rows = []
for m in models:
    for s in STRAT_ORDER:
        sub = per_case[(per_case.model == m) & (per_case.strategy == s)]["cv"]
        rows.append({
            "modelo": MODEL_LABEL.get(m, m), "estrategia": STRAT_LABEL[s],
            "mediana": sub.median(), "q1": sub.quantile(.25), "q3": sub.quantile(.75),
            "max": sub.max(),
            "caso_max": per_case[(per_case.model == m) & (per_case.strategy == s)].sort_values("cv").iloc[-1]["case_id"],
        })
tab = pd.DataFrame(rows)
with open(TAB / "tabela_estabilidade_por_caso.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[ht]\n\\centering\n"
            "\\caption{Distribuição do coeficiente de variação (CV, \\%) do METEOR por caso de teste, entre as 5 repetições: mediana, intervalo interquartil (IIQ) e valor máximo observado.}\n"
            "\\label{tab:estabilidade_caso}\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\hline\n"
            "\\textbf{Modelo} & \\textbf{Estratégia} & \\textbf{Mediana} & \\textbf{IIQ (P25--P75)} & \\textbf{Máximo} & \\textbf{Caso do máx.} \\\\\n\\hline\n")
    for r in rows:
        f.write(f"{r['modelo']} & {r['estrategia']} & {r['mediana']:.2f} & "
                f"{r['q1']:.2f}--{r['q3']:.2f} & {r['max']:.2f} & {r['caso_max']} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# --------- 2. Post-hoc com effect size ---------
pivot = df.groupby(["case_id", "model", "strategy"])["meteor"].mean().reset_index()
wide = pivot.pivot(index="case_id", columns=["model", "strategy"], values="meteor")

ph = []
for m in models:
    pairs = list(combinations(STRAT_ORDER, 2))
    for s1, s2 in pairs:
        a, b = wide[(m, s1)], wide[(m, s2)]
        d = b - a  # positivo = s2 melhor
        res = sps.wilcoxon(a, b)
        p_adj = min(res.pvalue * len(pairs), 1.0)
        # rank-biserial pareado: r = (W+ - W-) / (W+ + W-)
        dd = d[d != 0]
        ranks = sps.rankdata(np.abs(dd))
        w_pos = ranks[dd > 0].sum()
        w_neg = ranks[dd < 0].sum()
        r_rb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else 0.0
        ph.append({
            "modelo": MODEL_LABEL.get(m, m),
            "comp": f"{STRAT_LABEL[s1]} vs. {STRAT_LABEL[s2]}",
            "dif_mediana": d.median(), "p": res.pvalue, "p_adj": p_adj, "r": r_rb,
        })
with open(TAB / "tabela_posthoc.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[ht]\n\\centering\n"
            "\\caption{Comparações post-hoc pareadas (Wilcoxon) entre estratégias, por modelo: diferença mediana de METEOR (segunda estratégia menos a primeira), valor-p bruto, valor-p ajustado (Bonferroni) e tamanho de efeito (correlação rank-biserial pareada $r$).}\n"
            "\\label{tab:posthoc}\n\\footnotesize\n\\begin{tabular}{llcccc}\n\\hline\n"
            "\\textbf{Modelo} & \\textbf{Comparação} & \\textbf{$\\Delta$ mediana} & \\textbf{$p$} & \\textbf{$p_{adj}$} & \\textbf{$r$} \\\\\n\\hline\n")
    for r in ph:
        pfmt = "$<$0,001" if r["p"] < 0.001 else f"{r['p']:.3f}".replace(".", ",")
        pafmt = "$<$0,001" if r["p_adj"] < 0.001 else f"{r['p_adj']:.3f}".replace(".", ",")
        f.write(f"{r['modelo']} & {r['comp']} & {r['dif_mediana']:+.3f} & {pfmt} & {pafmt} & {r['r']:+.2f} \\\\\n".replace("+0.", "+0,").replace("-0.", "-0,"))
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# --------- 3. Boxplot legível ---------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})
SHORT = {"gemini": "Gemini\n2.5 Flash", "gpt-4o-mini": "GPT-4.1\nMini", "deepseek": "DeepSeek\nR1"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
order = ["gemini", "gpt-4o-mini", "deepseek"]
for ax, s in zip(axes, STRAT_ORDER):
    sub = df[df["strategy"] == s]
    data = [sub[sub["model"] == m]["meteor"] for m in order]
    bp = ax.boxplot(data, tick_labels=[SHORT[m] for m in order], widths=0.55, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#cfe3f5", linewidth=1.4)
    for el in ["whiskers", "caps", "medians"]:
        for item in bp[el]:
            item.set(linewidth=1.4)
    ax.set_title(STRAT_LABEL[s])
    ax.grid(axis="y", linestyle=":", alpha=.6)
axes[0].set_ylabel("Escore METEOR")
fig.tight_layout()
fig.savefig(FIG / "boxplot_meteor_v2.png", dpi=300)

print(open(TAB / "tabela_estabilidade_por_caso.tex").read())
print(open(TAB / "tabela_posthoc.tex").read())
