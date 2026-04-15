# Spec — ReportAgent

## Responsibility

Compose a complete scientific article in LaTeX from all prior artifacts, then compile to PDF.

## Inputs

| Source | Key | Type | Description |
|--------|-----|------|-------------|
| RunContext | `goal` | str | Research goal (for abstract/intro) |
| RunContext | `artifacts.literature_review` | path | Markdown review |
| RunContext | `artifacts.references` | path | BibTeX file |
| RunContext | `artifacts.dataset_metadata` | path | Dataset description for Methods |
| RunContext | `artifacts.model_results` | path | Model results for Results section |
| RunContext | `artifacts.figures_dir` | path | Figures directory |
| agent_config.yaml | `report.template` | path | LaTeX template file |
| agent_config.yaml | `report.sections` | list[str] | Sections to include |
| agent_config.yaml | `report.figures.format` | str | pdf / png (default: pdf) |
| agent_config.yaml | `report.figures.dpi` | int | Figure DPI (default: 300) |

## Outputs

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| `article` | `output/article.tex` | LaTeX | Complete article source |
| `article_pdf` | `output/article.pdf` | PDF | Compiled article (if pdflatex available) |

## Default Parameters

```yaml
sections: [abstract, introduction, methods, results, discussion, conclusion]
figures:
  format: pdf
  dpi: 300
```

## LaTeX Template Interface

The template must contain placeholder comments that ReportAgent fills:

```latex
%% PLACEHOLDER: abstract
%% PLACEHOLDER: introduction
%% PLACEHOLDER: methods
%% PLACEHOLDER: results
%% PLACEHOLDER: discussion
%% PLACEHOLDER: conclusion
%% PLACEHOLDER: bibliography  →  \bibliography{references}
%% PLACEHOLDER: figures_dir   →  path injected into \graphicspath
```

## Section Content Mapping

| Section | Source data |
|---------|-------------|
| abstract | goal + key model metrics (via LLM) |
| introduction | goal + literature_review intro (via LLM) |
| methods | dataset_metadata + model config (via LLM) |
| results | model_results.json + figures (via LLM + \includegraphics) |
| discussion | model diagnostics + literature context (via LLM) |
| conclusion | summary of results + goal (via LLM) |

## Behavior

1. Load all input artifacts
2. Load LaTeX template (default if not provided)
3. For each section in `report.sections`:
   a. Gather relevant data from artifacts
   b. Generate section text via LLM
   c. Insert into template at `%% PLACEHOLDER: {section}`
4. Insert bibliography reference
5. Write `article.tex`
6. If `pdflatex` available → compile to `article.pdf`
7. Update RunContext

## Error Handling

- If template file not found → use built-in default template
- If a section's source artifact is missing → skip section, log warning
- If pdflatex not available → write `.tex` only, log info

## Success Criteria

- [ ] `article.tex` contains all requested sections
- [ ] All figures referenced in `.tex` exist in figures_dir
- [ ] `.bib` file referenced correctly
- [ ] RunContext updated with artifact paths and status = "completed"
