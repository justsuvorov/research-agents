# Spec — DataAgent

## Responsibility

Collect raw training data from three sources and assemble a single wide table via outer join.
No preprocessing — all cleaning and encoding is handled by MLAgent (outboxml).

## Data Sources

| # | Source | Description |
|---|--------|-------------|
| 1 | **Paper extraction** | Copy measured/experimental values from paper abstracts and text |
| 2 | **Standards calculations** | Compute values by applying formulas from RMRS, GOST, ISO rules via LLM |
| 3 | **User CSV** | Optional file provided by user to supplement or correct the dataset |

## Inputs

| Source | Key | Type | Description |
|--------|-----|------|-------------|
| RunContext | `artifacts.literature_review` | path | Markdown review with paper list |
| RunContext | `artifacts.references` | path | BibTeX — paper metadata |
| AgentConfig | `data.extraction_rules` | list[ExtractionRule] | Columns to extract from papers |
| AgentConfig | `data.calculations` | list[CalculationRule] | Formulas to apply from standards |
| AgentConfig | `data.user_data` | str | Optional path to user-provided CSV |
| AgentConfig | `data.output_format` | str | csv / json (default: csv) |

## Outputs

| Artifact | Path | Format |
|----------|------|--------|
| `dataset` | `output/dataset.csv` | CSV — raw, unprocessed |
| `dataset_metadata` | `output/dataset_metadata.json` | JSON — column descriptions and source info |

---

## agent_config.yaml — data section schema

```yaml
data:
  output_format: csv
  user_data: null             # optional path to user CSV

  extraction_rules:
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

  calculations:
    - name: rmrs_torque
      standard: "RMRS 6.2.1.7"
      description: "Slewing ring torque M_dyn from RMRS rules"
      formula: "M = (F_N * r + F_fr * r_fr) * K_dyn"
      output_columns:
        - name: M_dyn
          unit: "N·m"
          type: numeric
        - name: contact_stress_calc
          unit: "MPa"
          type: numeric
      parameter_ranges:
        F_N: [50000, 100000, 150000, 200000]   # normal force, N
        r: [0.5, 0.8, 1.0, 1.2]               # pitch radius, m
        K_dyn: [1.2, 1.4, 1.6, 1.8, 2.0]     # dynamic factor
```

---

## CalculationRule Model

```python
class CalculationRule(BaseModel):
    name: str
    standard: str                   # e.g. "RMRS 6.2.1.7"
    description: str
    formula: str                    # human-readable formula string
    output_columns: list[OutputColumn]
    parameter_ranges: dict[str, list[float]]
```

LLM receives the formula, the standard reference, and the parameter grid.
It returns a list of rows — one row per parameter combination.

---

## Behavior

1. **Paper extraction** (per paper, per extraction_rule)
   - For each paper in the literature review:
     - Send abstract + extraction rules to LLM
     - LLM returns rows (1 or more per paper) with extracted values
     - Each row tagged with `source = bibtex_key`, `source_type = "paper"`

2. **Standards calculations** (per calculation_rule)
   - For each CalculationRule:
     - Build parameter grid from `parameter_ranges` (Cartesian product)
     - Send formula + standard + grid to LLM
     - LLM computes output_columns for each grid row
     - Each row tagged with `source = standard`, `source_type = "calculation"`

3. **User data** (optional)
   - If `data.user_data` is set and file exists → load CSV
   - Tag rows with `source = "user"`, `source_type = "user"`

4. **Assembly**
   - Outer join all three sources on common columns (by name)
   - Columns not present in a source → NaN for that source's rows
   - Add metadata columns: `source`, `source_type`

5. **Export**
   - Write dataset.csv and dataset_metadata.json
   - Update RunContext artifacts

---

## LLM Contracts

### Paper extraction — response per paper

```json
[
  {
    "wear_intensity": 0.14,
    "contact_stress": 480.0,
    "dynamic_factor": 1.4,
    "lubrication_mode": "boundary",
    "quote": "measured wear rate of 0.14 mm/km under boundary lubrication"
  },
  {
    "wear_intensity": 0.22,
    "contact_stress": 610.0,
    "dynamic_factor": 1.8,
    "lubrication_mode": "boundary",
    "quote": "second test series at higher load showed 0.22 mm/km"
  }
]
```

Returns empty list `[]` if paper contains no relevant data.

### Standards calculation — response per CalculationRule

```json
[
  {"F_N": 50000, "r": 0.5, "K_dyn": 1.2, "M_dyn": 30000.0, "contact_stress_calc": 412.5},
  {"F_N": 50000, "r": 0.5, "K_dyn": 1.4, "M_dyn": 35000.0, "contact_stress_calc": 481.3}
]
```

---

## dataset_metadata.json

```json
{
  "created_at": "...",
  "n_rows": 87,
  "n_cols": 8,
  "sources": {
    "papers": 34,
    "calculations": 48,
    "user": 5
  },
  "columns": [
    {"name": "wear_intensity", "type": "numeric", "unit": "mm/km", "n_missing": 12}
  ]
}
```

---

## Error Handling

- Paper yields no rows → skip, log info
- Calculation LLM response unparseable → log warning, skip that rule
- User CSV not found → log warning, continue without it
- All sources yield 0 rows → raise `EmptyDatasetError`

## Success Criteria

- [ ] Dataset contains columns from extraction_rules + calculation output_columns
- [ ] `source` and `source_type` columns present
- [ ] dataset_metadata.json reflects actual shape and per-source row counts
- [ ] RunContext updated with artifact paths, status = "completed"
