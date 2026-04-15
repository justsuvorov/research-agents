# Architecture Spec — Research Agent Pipeline

## Overview

A pipeline of 4 sequential AI agents that transforms a research goal into a complete scientific article.

## User Inputs

| File | Format | Description |
|------|--------|-------------|
| `research_goal.txt` | Plain text | Research goal, objectives, hypotheses |
| `agent_config.yaml` | YAML | Tools, sources, ML models, dataset rules, report rules |

If `agent_config.yaml` is not provided, each agent falls back to its **default parameters**.

## Pipeline Flow

```
research_goal.txt
agent_config.yaml
        │
        ▼
┌───────────────────┐
│  ResearchAgent    │  → literature_review.md
│                   │  → references.bib
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  DataAgent        │  → dataset.csv
│                   │  → dataset_metadata.json
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MLAgent          │  → model_results.json
│                   │  → figures/
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ReportAgent      │  → article.tex
│                   │  → article.pdf (optional)
└───────────────────┘
```

## Shared State

All agents communicate through a **RunContext** object persisted as `run_context.json` in the working directory.

```json
{
  "run_id": "uuid",
  "goal": "...",
  "config": { ... },
  "artifacts": {
    "literature_review": "path/to/literature_review.md",
    "references": "path/to/references.bib",
    "dataset": "path/to/dataset.csv",
    "dataset_metadata": "path/to/dataset_metadata.json",
    "model_results": "path/to/model_results.json",
    "figures_dir": "path/to/figures/",
    "article": "path/to/article.tex"
  },
  "agent_status": {
    "research": "completed | failed | pending",
    "data": "completed | failed | pending",
    "ml": "completed | failed | pending",
    "report": "completed | failed | pending"
  }
}
```

## agent_config.yaml — Schema

```yaml
research:
  sources: [arxiv, semantic_scholar, pubmed]   # default: [arxiv, semantic_scholar]
  max_papers: 50                                # default: 30
  citation_format: APA                          # default: APA
  language: en                                  # default: en

data:
  extraction_rules:
    - type: table | text | numeric
      description: "..."
  output_format: csv                            # default: csv
  preprocessing: []                             # list of steps

ml:
  library: mylib                                # required — user's ML library module
  model: GLM
  target_variable: "..."
  features: []
  hyperparameters: {}

report:
  template: templates/article_template.tex     # required for custom; default template used otherwise
  sections: [abstract, introduction, methods, results, discussion, conclusion]
  figures:
    format: pdf                                 # default: pdf
    dpi: 300
```

## Entry Point

```
python main.py --goal research_goal.txt [--config agent_config.yaml] [--output ./output]
```

## Invariants

- Each agent is **idempotent**: re-running with the same inputs produces the same outputs.
- Each agent can be run **independently** if prior artifacts are already present in RunContext.
- Agents do not share global state — only RunContext.
- All file paths in RunContext are absolute.
