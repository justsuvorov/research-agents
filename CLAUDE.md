# CLAUDE.md — Development Rules for research-agents

This file is read by Claude Code at the start of every session.
Follow all rules below without exception.

---

## Project Overview

Multi-agent pipeline for automated scientific article generation:
`research_goal.txt + agent_config.yaml → literature_review + dataset + GLM model + article.tex`

**Stack:** Python 3.11+, Pydantic v2, loguru, anthropic SDK, outboxml, pandas, httpx

**ML library:** `outboxml==0.10.0` — the primary tool for the ML agent (GLM, AutoML, WoE encoding).
Always use outboxml for model training, feature engineering, and result export. Do not replace it.

---

## Architecture Rules

### Dependency Injection
All components are assembled in `main.py` and injected via constructors.
No class instantiates its own dependencies. No global state outside `RunContext`.

```python
# correct
class QueryBuilder:
    def __init__(self, client: anthropic.Anthropic, system_prompt: str, user_template: str):
        ...

# wrong — instantiating dependencies inside a class
class QueryBuilder:
    def __init__(self):
        self._client = anthropic.Anthropic()  # NO
```

### No prompts in Python modules
All LLM prompt text lives in `prompts/` as `.txt` files.
Python code only receives prompt strings as constructor arguments via `PromptLoader`.

```python
# correct — prompt injected
synthesizer = Synthesizer(client=llm, system_prompt=prompts.prompt_text("research", "system.txt"), ...)

# wrong — inline string
SYSTEM_PROMPT = "You are an expert..."  # NO
```

### Agents
- All agents inherit `BaseAgent` and implement `run()`.
- `execute()` in `BaseAgent` manages status transitions and error capture — never override it.
- Agents are idempotent: skip work if `ctx.is_completed(self.name)` is True.
- Agents communicate only through `RunContext` — never call each other directly.

### Pipeline
`ResearchPipeline` in `pipeline.py` runs agents sequentially.
It receives all agents via constructor. It does not import or instantiate agents itself.

---

## Naming Convention

**Function and method names must equal the name of the object they return.**

```python
# correct
def agent_config(path) -> AgentConfig: ...
def research_goal(path) -> str: ...
def search_queries(goal, n) -> list[str]: ...
def paper_analysis(paper) -> PaperAnalysis: ...
def bib_file(papers, path) -> None: ...   # side-effect: exception to the rule

# wrong
def load_config() ...   # NO
def get_queries() ...   # NO
def analyze_paper() ... # NO
```

Classmethod names follow the same rule:
```python
@classmethod
def run_context(cls, output_dir) -> RunContext: ...       # correct
def run_context_or_new(cls, ...) -> RunContext: ...       # correct
```

---

## Code Style

### Pydantic v2
Use Pydantic `BaseModel` for all data structures. No plain dataclasses.

```python
from pydantic import BaseModel, Field
class Paper(BaseModel):
    title: str
    year: Optional[int] = None
```

### Type annotations
Full annotations on every function signature. Use `from __future__ import annotations` in every file.

```python
def papers(self, query: str, max_results: int) -> list[Paper]: ...
```

Prefer built-in generics (`list[str]`, `dict[str, Any]`) over `List`, `Dict` from `typing`.

### Classes
| Suffix / Pattern | Example |
|-----------------|---------|
| Config models | `ResearchConfig`, `MLConfig` |
| Result/container | `PaperAnalysis`, `LiteratureReport`, `AutoMLResult` |
| Errors | `ConfigError`, `LibraryImportError` |
| Abstract bases | `BaseAgent`, `BaseSearcher` |
| Agents | `ResearchAgent`, `DataAgent` |
| Builders/loaders | `QueryBuilder`, `PromptLoader` |

### Enums
Use `str, Enum` so values serialize naturally to JSON and YAML.

```python
class AgentStatus(str, Enum):
    PENDING = "pending"
```

### Error handling
Define custom exceptions per domain in an `errors.py` or at module level.
Use `raise SomeError(msg) from exc` to preserve traceback chain.
Never silence exceptions with bare `except: pass`.

### Logging
Use `loguru` exclusively. No `import logging`.

```python
from loguru import logger
logger.info("[AgentName] doing thing: {}", value)
logger.warning("[AgentName] skipping source: {}", source_id)
logger.debug("[AgentName] query={!r}", query)
```

Format: `[ClassName] message: {value}` — always prefix with class/component name.

---

## outboxml Integration (ML Agent)

`outboxml` is the sole ML library. Import pattern:

```python
from outboxml.automl_manager import AutoMLManager
from outboxml.core.pydantic_models import ModelConfig, FeatureModelConfig
```

Configuration is driven by Pydantic models (`ModelConfig`, `FeatureModelConfig`).
Use `AutoMLManager` as the orchestrator — do not bypass it with raw CatBoost/XGBoost calls.
WoE encoding is available via `OptiBinningEncoder` — use it for the GLM feature preparation.

Results are accessed through `AutoMLResult` — expose `coefficients`, `metrics`, `diagnostics`
from the result object into `model_results.json`.

---

## File & Module Layout

```
research-agents/
├── CLAUDE.md               ← this file
├── main.py                 ← wires all dependencies, CLI entry point
├── pyproject.toml
├── .env                    ← secrets (not committed)
├── .env.example
├── prompts/                ← ALL prompt text files (.txt)
│   └── research/
├── specs/                  ← SDD spec files (.spec.md) — write before implementation
├── config/
│   └── default_config.yaml
└── src/research_agents/
    ├── pydantic_models.py  ← RunContext, Artifacts, AgentStatuses
    ├── config.py           ← AgentConfig Pydantic models + load functions
    ├── base_agent.py       ← BaseAgent ABC
    ├── pipeline.py         ← ResearchPipeline orchestrator
    ├── prompt_loader.py    ← PromptLoader
    └── agents/
        ├── research_agent.py
        ├── data_agent.py
        ├── ml_agent.py
        ├── report_agent.py
        └── research/       ← sub-components of ResearchAgent
            ├── models.py
            ├── query_builder.py
            ├── paper_analyzer.py
            ├── synthesizer.py
            ├── searchers/
            └── exporters/
```

---

## Spec Driven Development (SDD)

**Write the spec before writing code.**

For every new agent or significant component:
1. Create `specs/NN_component_name.spec.md` with: inputs, outputs, behavior, success criteria.
2. Get alignment before implementing.
3. Implementation must not exceed what the spec describes.

Spec files are the source of truth for what an agent does.

---

## What NOT to do

- Do not add features beyond what is asked.
- Do not refactor code that was not part of the current task.
- Do not add comments to code you did not change.
- Do not use `logging` — use `loguru`.
- Do not use plain `dataclasses` — use Pydantic `BaseModel`.
- Do not hardcode prompt text in `.py` files.
- Do not instantiate `anthropic.Anthropic()` inside agent classes — inject it.
- Do not use `outboxml` alternatives (raw sklearn, statsmodels GLM) for ML tasks.
- Do not commit `.env`.
