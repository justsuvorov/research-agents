# Spec — DataAgent

## Responsibility

Extract structured data from collected sources and prepare a clean dataset ready for ML modeling.

## Inputs

| Source | Key | Type | Description |
|--------|-----|------|-------------|
| RunContext | `artifacts.literature_review` | path | Markdown review from ResearchAgent |
| RunContext | `artifacts.references` | path | BibTeX file with paper metadata |
| agent_config.yaml | `data.extraction_rules` | list | What data to extract and how |
| agent_config.yaml | `data.output_format` | str | csv / json (default: csv) |
| agent_config.yaml | `data.preprocessing` | list | Preprocessing steps to apply |

## Outputs

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| `dataset` | `output/dataset.csv` | CSV | Clean dataset for modeling |
| `dataset_metadata` | `output/dataset_metadata.json` | JSON | Column descriptions, units, sources |

## Default Parameters

```yaml
output_format: csv
preprocessing:
  - drop_duplicates
  - drop_na
```

## extraction_rules Schema

```yaml
extraction_rules:
  - type: numeric        # numeric | categorical | text | boolean
    name: column_name    # output column name
    description: "..."   # what to extract and from where
    source: abstract | full_text | table
    unit: "..."          # optional
```

## dataset_metadata.json Structure

```json
{
  "created_at": "ISO timestamp",
  "source_papers": ["doi1", "doi2", ...],
  "n_rows": 0,
  "n_cols": 0,
  "columns": [
    {
      "name": "col_name",
      "type": "numeric",
      "description": "...",
      "unit": "...",
      "n_missing": 0
    }
  ]
}
```

## Behavior

1. Load literature review and references
2. For each extraction rule → extract data from papers (via LLM over abstracts/text)
3. Assemble raw DataFrame
4. Apply preprocessing steps in order
5. Validate: check types, missing values, outliers (log warnings)
6. Write dataset and metadata
7. Update RunContext

## Error Handling

- If extraction yields no rows for a rule → log warning, add column of NaN
- If dataset has 0 rows after preprocessing → raise `EmptyDatasetError`

## Success Criteria

- [ ] Dataset has at least 1 row and matches extraction_rules columns
- [ ] dataset_metadata.json reflects actual column structure
- [ ] No silent type coercions (fail loudly on type mismatch)
- [ ] RunContext updated with artifact paths and status = "completed"
