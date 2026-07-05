#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Etapa 2: avaliação das gerações com a métrica METEOR.

Compara cada cenário gerado (results/raw/generations.jsonl) com o gabarito
humano correspondente (dataset/ground_truth.json) e grava os escores em
results/metrics/meteor_scores.csv.

Esta etapa é DETERMINÍSTICA: dado o mesmo generations.jsonl, os escores
produzidos são sempre idênticos (requisito de reprodutibilidade do estudo).
"""
import csv
import json
from pathlib import Path

import nltk
from nltk.translate.meteor_score import meteor_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw" / "generations.jsonl"
GT = ROOT / "dataset" / "ground_truth.json"
OUT = ROOT / "results" / "metrics" / "meteor_scores.csv"

for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


def tokenize(text: str):
    """Tokenização simples e determinística (minúsculas, separação por não alfanuméricos)."""
    import re
    return [t for t in re.split(r"[^\wáàâãéêíóôõúüç-]+", text.lower()) if t]


def main():
    with open(GT, encoding="utf-8") as f:
        gt = json.load(f)["gabarito"]

    rows = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref = tokenize(gt[r["case_id"]])
            hyp = tokenize(r["response"])
            score = meteor_score([ref], hyp)
            rows.append({
                "model": r["model"],
                "strategy": r["strategy"],
                "case_id": r["case_id"],
                "rep": r["rep"],
                "meteor": round(score, 4),
            })

    rows.sort(key=lambda x: (x["model"], x["strategy"], x["case_id"], x["rep"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "strategy", "case_id", "rep", "meteor"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} escores gravados em {OUT}")


if __name__ == "__main__":
    main()
