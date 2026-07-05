#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py - Etapa 1 do experimento: geração dos cenários Gherkin pelas LLMs.

Para cada combinação (modelo x estratégia x caso de teste x repetição), envia o
prompt à API correspondente com temperature=0 e grava a saída em
results/raw/generations.jsonl com metadados completos (reprodutibilidade).

O script tem CHECKPOINT: combinações já geradas são puladas, então pode ser
interrompido e retomado (útil para limites diários de APIs gratuitas).

Uso:
    python scripts/generate.py                # roda tudo
    python scripts/generate.py --model gemini # roda apenas um modelo
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATASET = ROOT / "dataset" / "casos_de_teste.json"
PROMPTS_DIR = ROOT / "prompts"
OUT_FILE = ROOT / "results" / "raw" / "generations.jsonl"

STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
N_REPS = int(os.getenv("N_REPS", "5"))
SLEEP = float(os.getenv("SLEEP_SECONDS", "2"))

MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "gpt-4o-mini": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1-0528"),
}


def call_gemini(prompt: str) -> str:
    """Gemini via OpenRouter (evita a cota diária de 20 req/dia do tier gratuito do Google)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=f"google/{MODELS['gemini']}",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=MODELS["gpt-4o-mini"],
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def call_deepseek(prompt: str) -> str:
    """DeepSeek via OpenRouter (API compatível com OpenAI)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=MODELS["deepseek"],
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content.strip()
    # DeepSeek R1 pode incluir bloco de raciocínio <think>...</think>
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


CALLERS = {"gemini": call_gemini, "gpt-4o-mini": call_openai, "deepseek": call_deepseek}


def clean_output(text: str) -> str:
    """Remove cercas de código e espaços extras que violariam a instrução."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["strategy"], r["case_id"], r["rep"]))
                except json.JSONDecodeError:
                    continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), help="rodar apenas um modelo")
    parser.add_argument("--dry-run", action="store_true",
                        help="não chama APIs; gera saídas simuladas para testar o pipeline")
    args = parser.parse_args()

    with open(DATASET, encoding="utf-8") as f:
        cases = json.load(f)["casos"]
    prompts = {s: (PROMPTS_DIR / f"{s}.txt").read_text(encoding="utf-8") for s in STRATEGIES}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT_FILE)
    models = [args.model] if args.model else list(MODELS)

    total = len(models) * len(STRATEGIES) * len(cases) * N_REPS
    print(f"Total de gerações previstas: {total} | já concluídas: {len(done)}")

    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for model in models:
            for strategy in STRATEGIES:
                for case in cases:
                    for rep in range(1, N_REPS + 1):
                        key = (model, strategy, case["id"], rep)
                        if key in done:
                            continue
                        prompt = prompts[strategy].replace(
                            "{test_case_description}", case["descricao"]
                        )
                        try:
                            if args.dry_run:
                                text = f"Scenario: simulacao {case['id']}\n  Given contexto\n  When acao\n  Then resultado"
                            else:
                                text = clean_output(CALLERS[model](prompt))
                        except Exception as e:
                            print(f"[ERRO] {key}: {e} — tentarei na próxima execução.")
                            time.sleep(SLEEP * 5)
                            continue
                        record = {
                            "model": model,
                            "model_version": MODELS[model],
                            "strategy": strategy,
                            "case_id": case["id"],
                            "rep": rep,
                            "temperature": 0.0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "response": text,
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                        print(f"[OK] {model} | {strategy} | {case['id']} | rep {rep}")
                        if not args.dry_run:
                            time.sleep(SLEEP)
    print("Geração concluída.")


if __name__ == "__main__":
    sys.exit(main())
