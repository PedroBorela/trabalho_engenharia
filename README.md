# Avaliação de Estratégias de Prompting em LLMs para a Geração Automatizada de Cenários Gherkin

Estudo empírico da disciplina **Engenharia de Software 3** (IFSEMG — Manhuaçu).
Autor: **Pedro Borela Andrade**.

Replicação adaptada do estudo de **Fernandes et al. (2025)**, *"A Comparative Study
of LLMs for Gherkin Generation"* (SBES 2025), avaliando **3 LLMs** (Gemini,
GPT-4o Mini e DeepSeek) em **3 estratégias de prompting** (zero-shot, one-shot,
few-shot) sobre **20 descrições de casos de teste** extraídas de **10 repositórios
públicos de QA** no GitHub.

## Estrutura do repositório

```
├── dataset/
│   ├── casos_de_teste.json   # 20 descrições em linguagem natural (com fonte e URL)
│   └── ground_truth.json     # gabarito Gherkin de referência (revisado pelo autor)
├── prompts/                  # prompts zero-shot, one-shot e few-shot
├── scripts/
│   ├── generate.py           # Etapa 1: chamadas às APIs (temperature=0, com checkpoint)
│   ├── evaluate.py           # Etapa 2: cálculo do METEOR (determinística)
│   └── stats.py              # Etapa 3: estatística, tabelas LaTeX e figura (determinística)
├── results/
│   ├── raw/generations.jsonl # todas as saídas das LLMs com metadados
│   ├── metrics/              # meteor_scores.csv
│   ├── tables/               # tabelas .tex prontas para o artigo
│   └── figures/              # boxplot
└── latex/                    # fontes LaTeX do relatório (template SBC)
```

## Como reproduzir

1. **Instalar dependências** (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. **Configurar chaves**: copie `.env.example` para `.env` e preencha
   `GEMINI_API_KEY`, `OPENAI_API_KEY` e `OPENROUTER_API_KEY`.
3. **Gerar os cenários** (Etapa 1 — envolve chamadas de API):
   ```bash
   python scripts/generate.py
   ```
   O script salva cada geração imediatamente e **pula combinações já feitas**,
   podendo ser interrompido e retomado (útil para limites diários de APIs
   gratuitas). Total padrão: 3 modelos x 3 estratégias x 20 casos x 5 repetições
   = **900 gerações**.
4. **Avaliar com METEOR** (Etapa 2 — determinística):
   ```bash
   python scripts/evaluate.py
   ```
5. **Estatística e tabelas** (Etapa 3 — determinística):
   ```bash
   python scripts/stats.py
   ```

### Nota sobre reprodutibilidade

A geração por LLMs é estocástica por natureza; por isso **todas as saídas brutas
são versionadas** em `results/raw/generations.jsonl`. As Etapas 2 e 3 são
totalmente determinísticas: a partir do mesmo `generations.jsonl`, qualquer
pessoa obtém exatamente os mesmos escores, tabelas e testes estatísticos
apresentados no relatório. Adicionalmente, `temperature=0` foi usado em todas as
chamadas para minimizar a variação entre execuções.

### Limites de requisição (APIs gratuitas)

- **Gemini** (Google AI Studio): o tier gratuito de modelos "Pro" tem limite
  diário baixo; se necessário, use `GEMINI_MODEL=gemini-1.5-flash` no `.env`
  ou distribua a execução em mais de um dia (o checkpoint cuida disso).
- **DeepSeek via OpenRouter**: variantes `:free` têm limite diário; alternativas:
  usar a API oficial da DeepSeek (muito barata) ou créditos no OpenRouter.
- `SLEEP_SECONDS` no `.env` controla a pausa entre chamadas.

## Estudo base (citação obrigatória)

Fernandes, H. et al. *A Comparative Study of LLMs for Gherkin Generation.*
SBES 2025. Artefatos: https://github.com/hiagonfs/bdd-scenario-evaluation

## Fontes do dataset

As 20 descrições foram extraídas (e minimamente padronizadas) dos repositórios
públicos listados em `dataset/casos_de_teste.json`, campo `url`, com atribuição
de arquivo e identificador original no campo `fonte`.
