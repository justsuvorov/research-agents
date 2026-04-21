# Research Agents

Multi-agent pipeline that transforms a research goal into a complete scientific article.

## Overview

Provide two files — a research goal and an optional config — and the pipeline produces a literature review, a clean dataset, a fitted GLM model, and a LaTeX article.

```
research_goal.txt + agent_config.yaml
           │
           ▼
   ┌──────────────────┐
   │  ResearchAgent   │  → literature_review.md + references.bib + papers.json
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
| **DataAgent** | Collects raw training data from three sources — paper extraction, standards calculations, and user CSV — and assembles a wide table via outer join |
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
  user_data: null                     # optional path to your own CSV

  extraction_rules:                   # copy measured values from paper abstracts
    - name: wear_intensity
      type: numeric
      description: "Wear intensity Ih = Δh/L in mm/km"
      unit: "mm/km"
    - name: contact_stress
      type: numeric
      description: "Hertzian contact stress σ_H at gear tooth, MPa"
      unit: "MPa"
    - name: dynamic_factor
      type: numeric
      description: "Dynamic load factor K_dyn (dimensionless)"
    - name: lubrication_mode
      type: categorical
      description: "Lubrication regime: boundary / mixed / hydrodynamic"

  calculations:                       # compute rows via RMRS/GOST/ISO formulas
    - name: rmrs_torque
      standard: "RMRS 6.2.1.7"
      description: "Slewing ring torque M_dyn from RMRS rules"
      formula: "M = (F_N * r + F_fr * r_fr) * K_dyn"
      output_columns:
        - {name: M_dyn, type: numeric, unit: "N·m"}
        - {name: contact_stress_calc, type: numeric, unit: "MPa"}
      parameter_ranges:
        F_N:   [50000, 100000, 150000, 200000]
        r:     [0.5, 0.8, 1.0, 1.2]
        K_dyn: [1.2, 1.4, 1.6, 1.8, 2.0]

ml:
  library: outboxml
  model: GLM
  target_variable: wear_intensity
  features: [contact_stress, dynamic_factor, M_dyn, lubrication_mode]

report:
  template: templates/article_template.tex
  sections: [abstract, introduction, methods, results, discussion, conclusion]
```

## DataAgent — Data Sources

The DataAgent collects raw data from up to three sources and assembles them via **outer join**:

| Source | Mechanism | Example |
|--------|-----------|---------|
| **Paper extraction** | LLM copies measured values from abstracts; one paper → multiple rows | `σ_H = 480 MPa` from experimental section |
| **Standards calculations** | LLM applies formula to Cartesian parameter grid | `M_dyn` per RMRS 6.2.1.7 for 80 combinations of `F_N × r × K_dyn` |
| **User CSV** | Load and concat user-provided file | Corrected or supplementary measurements |

Each row is tagged with `source` (paper key / standard / "user") and `source_type`.
No preprocessing is performed — all cleaning and encoding is handled by outboxml in MLAgent.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | yes | Claude API key |
| `SEMANTIC_SCHOLAR_API_KEY` | no | Higher rate limits |
| `ELIBRARY_API_TOKEN` | no | Access to eLIBRARY.ru / РИНЦ |
| `MAILTO` | no | Email for CrossRef polite pool (MDPI) |
| `PROMPTS_DIR` | no | Path to prompts directory (default: `./prompts`) |
| `OUTPUT_DIR` | no | Output directory (default: `./output`) |

## Outputs

All artifacts are written to `./output/` (or `OUTPUT_DIR`):

```
output/
├── run_context.json        ← pipeline state, artifact paths, agent statuses
├── literature_review.md    ← structured review grouped by knowledge category
├── references.bib          ← BibTeX bibliography
├── papers.json             ← analyzed papers with summaries and gap analyses
├── dataset.csv             ← raw collected dataset (paper + calc + user rows)
├── dataset_metadata.json   ← column descriptions, per-source row counts, stats
├── model_results.json      ← GLM coefficients, metrics, diagnostics
├── figures/
│   ├── coef_plot.pdf       ← coefficients with confidence intervals
│   ├── residuals.pdf       ← residuals vs fitted
│   └── qq_plot.pdf         ← QQ-plot of residuals
└── article.tex             ← complete LaTeX article
```

## Prompt Customization

All LLM prompts are plain text files in `prompts/`. Edit without touching Python:

```
prompts/
├── research/
│   ├── system.txt          ← domain role + constraints (cached by Claude)
│   ├── query_builder.txt   ← search query generation
│   ├── paper_analyzer.txt  ← per-paper filter + summary + equation + gap
│   └── synthesizer.txt     ← literature review section writing
└── data/
    ├── system.txt          ← engineering domain role (cached by Claude)
    ├── paper_extractor.txt ← extract rows from paper abstract
    └── standards_calculator.txt ← apply formula to parameter grid
```

## Project Structure

```
research-agents/
├── CLAUDE.md
├── main.py                 ← entry point, dependency wiring
├── pyproject.toml
├── .env.example
├── prompts/
│   ├── research/
│   └── data/
├── specs/
│   ├── 00_architecture.spec.md
│   ├── 01_research_agent.spec.md
│   ├── 02_data_agent.spec.md
│   ├── 03_ml_agent.spec.md
│   ├── 04_report_agent.spec.md
│   └── class_diagram.md
├── config/
│   └── default_config.yaml
└── src/research_agents/
    ├── pydantic_models.py
    ├── config.py
    ├── base_agent.py
    ├── pipeline.py
    ├── prompt_loader.py
    └── agents/
        ├── research_agent.py
        ├── data_agent.py
        ├── ml_agent.py
        ├── report_agent.py
        ├── research/
        │   ├── models.py
        │   ├── query_builder.py
        │   ├── paper_analyzer.py
        │   ├── synthesizer.py
        │   ├── searchers/
        │   └── exporters/
        └── data/
            ├── paper_extractor.py
            ├── standards_calculator.py
            └── assembler.py
```

## Development

Follows **Spec Driven Development**: specs in `specs/` are written before implementation.
See [`CLAUDE.md`](CLAUDE.md) for full coding conventions and [`specs/class_diagram.md`](specs/class_diagram.md) for the full class diagram.

```bash
pytest tests/      # run tests
ruff check src/    # lint
mypy src/          # type check
```

## Requirements

- Python 3.11+
- [outboxml](https://github.com/SVSemyonov/outboxml) == 0.10.0
- Anthropic API key (Claude)
