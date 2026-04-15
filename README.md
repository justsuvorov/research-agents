# Research Agents

Multi-agent pipeline that transforms a research goal into a complete scientific article.

## Overview

Provide two files — a research goal and an optional config — and the pipeline produces a literature review, a clean dataset, a fitted GLM model, and a LaTeX article.

```
research_goal.txt + agent_config.yaml
           │
           ▼
   ┌──────────────────┐
   │  ResearchAgent   │  → literature_review.md + references.bib
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │   DataAgent      │  → dataset.csv + dataset_metadata.json
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │    MLAgent       │  → model_results.json + figures/
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │  ReportAgent     │  → article.tex + article.pdf
   └──────────────────┘
```

## Agents

| Agent | Responsibility |
|-------|---------------|
| **ResearchAgent** | Searches Semantic Scholar, arXiv, MDPI, eLIBRARY.ru; filters papers by domain constraints; extracts summaries, key equations, gap analyses; synthesizes a structured literature review |
| **DataAgent** | Extracts structured data from collected sources according to user-defined extraction rules; assembles and preprocesses a dataset |
| **MLAgent** | Uses [outboxml](https://github.com/SVSemyonov/outboxml) to fit a GLM model; exports coefficients, metrics, diagnostics, and diagnostic plots |
| **ReportAgent** | Fills a LaTeX template with all artifacts; compiles to PDF |

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env — set ANTHROPIC_API_KEY at minimum
```

### 3. Create your research goal

```
# research_goal.txt
Цель: Разработать модель интенсивности износа зубьев ОПУ с внутренним
эвольвентным зацеплением при граничной смазке в условиях морского климата.

Задачи:
1. Собрать математические модели износа (Крагельский, Арчард).
2. Найти данные о влиянии динамического момента M_дин.
3. Обосновать применение GLM для прогнозирования износа.
```

### 4. Run

```bash
python main.py --goal research_goal.txt
```

With custom config:

```bash
python main.py --goal research_goal.txt --config agent_config.yaml --output ./output
```

## Configuration

All agent parameters are set in `agent_config.yaml`. If omitted, defaults from `config/default_config.yaml` are used.

```yaml
research:
  sources: [semantic_scholar, arxiv, mdpi, elibrary]
  max_papers: 30
  citation_format: APA

data:
  output_format: csv
  extraction_rules:
    - type: numeric
      name: contact_stress
      description: "Contact stress σ_H in MPa"
      source: abstract

ml:
  library: outboxml
  model: GLM
  target_variable: wear_intensity
  features: [contact_stress, dynamic_factor, lubrication_index]

report:
  template: templates/article_template.tex
  sections: [abstract, introduction, methods, results, discussion, conclusion]
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | yes | Claude API key |
| `SEMANTIC_SCHOLAR_API_KEY` | no | Higher rate limits |
| `ELIBRARY_API_TOKEN` | no | Access to eLIBRARY.ru / РИНЦ |
| `MAILTO` | no | Email for CrossRef polite pool |
| `PROMPTS_DIR` | no | Path to prompts directory (default: `./prompts`) |
| `OUTPUT_DIR` | no | Output directory (default: `./output`) |

## Outputs

All artifacts are written to `./output/` (or `OUTPUT_DIR`):

```
output/
├── run_context.json        ← pipeline state, artifact paths, status
├── literature_review.md    ← structured review with inline citations
├── references.bib          ← BibTeX bibliography
├── dataset.csv             ← extracted and preprocessed dataset
├── dataset_metadata.json   ← column descriptions and statistics
├── model_results.json      ← GLM coefficients, metrics, diagnostics
├── figures/                ← coefficient plot, residuals, QQ-plot
│   ├── coef_plot.pdf
│   ├── residuals.pdf
│   └── qq_plot.pdf
└── article.tex             ← complete LaTeX article
```

## Prompt Customization

All LLM prompts are plain text files in `prompts/`. Edit them without touching Python code:

```
prompts/research/
├── system.txt          ← domain role + constraints (cached by Claude)
├── query_builder.txt   ← search query generation template
├── paper_analyzer.txt  ← per-paper filter + summary + equation + gap
└── synthesizer.txt     ← literature review section writing template
```

## Project Structure

```
research-agents/
├── CLAUDE.md               ← AI assistant development rules
├── main.py                 ← entry point, dependency wiring
├── pyproject.toml
├── .env.example
├── prompts/                ← LLM prompt templates
├── specs/                  ← Spec Driven Development specs
├── config/
│   └── default_config.yaml
└── src/research_agents/
    ├── pydantic_models.py  ← RunContext, Artifacts, AgentStatuses
    ├── config.py           ← AgentConfig Pydantic models
    ├── base_agent.py       ← BaseAgent ABC
    ├── pipeline.py         ← ResearchPipeline orchestrator
    ├── prompt_loader.py
    └── agents/
        ├── research_agent.py
        ├── data_agent.py
        ├── ml_agent.py
        ├── report_agent.py
        └── research/
            ├── models.py
            ├── query_builder.py
            ├── paper_analyzer.py
            ├── synthesizer.py
            ├── searchers/
            └── exporters/
```

## Development

This project follows **Spec Driven Development**: specs in `specs/` are written before implementation.
See [`CLAUDE.md`](CLAUDE.md) for full coding conventions.

```bash
# Run tests
pytest tests/

# Lint
ruff check src/

# Type check
mypy src/
```

## Requirements

- Python 3.11+
- [outboxml](https://github.com/SVSemyonov/outboxml) == 0.10.0 — ML pipeline library
- Anthropic API key (Claude)
