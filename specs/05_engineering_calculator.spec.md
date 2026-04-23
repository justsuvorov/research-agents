# Spec — EngineeringCalculator

## Responsibility

Apply chained engineering calculations from published standards (DNV, FEM, ISO, RMRS)
to a user-defined parameter grid, returning a structured dataset of computed results.

The LLM acts as the "formula engine": it knows the standard calculation procedures
and executes the full chain (intermediate variables included) for each parameter combination.
The user does not write formulas — they specify the mechanism, the standards, and the input ranges.

---

## Contrast with StandardsCalculator

| | StandardsCalculator | EngineeringCalculator |
|---|---|---|
| Formula source | User writes formula string in config | LLM knows the standard |
| Chain support | Single formula | Multi-step chain with intermediates |
| Standards lookup | User provides reference | LLM resolves from standard name |
| Use case | Known, simple formula | Full mechanism design per DNV/FEM/ISO |

---

## Config Schema — `engineering_calculations`

```yaml
data:
  engineering_calculations:
    - name: lifting_mechanism
      mechanism: "Механизм подъёма"
      standards:
        - "FEM 1.001"
        - "DNV-ST-0378"
        - "ISO 4301-1"
      description: >
        Расчёт основных параметров механизма подъёма:
        натяжение каната, диаметр барабана, скорость и момент мотора,
        объём гидромотора.
      input_parameters:
        SWL_tonne: [1.0, 2.0, 2.75, 3.0, 5.0]
        polispast_ratio: [2, 4]
        rope_class: ["T3", "T4"]
        load_group: ["M3", "M4", "M5"]
      output_columns:
        - name: rope_tension_kN
          unit: kN
          type: numeric
        - name: rope_diameter_mm
          unit: mm
          type: numeric
        - name: drum_diameter_mm
          unit: mm
          type: numeric
        - name: drum_width_mm
          unit: mm
          type: numeric
        - name: drum_torque_nom_kNm
          unit: kN·m
          type: numeric
        - name: motor_speed_rpm
          unit: rpm
          type: numeric
        - name: hydraulic_volume_cm3
          unit: cm³
          type: numeric
        - name: system_pressure_bar
          unit: bar
          type: numeric

    - name: slewing_mechanism
      mechanism: "Механизм поворота"
      standards:
        - "FEM 1.001"
        - "DNV-ST-0378"
        - "ISO 4301-2"
      description: >
        Расчёт механизма поворота крана: момент на ОПУ,
        выбор редуктора и гидромотора.
      input_parameters:
        SWL_tonne: [1.0, 2.0, 2.75, 3.0]
        outreach_m: [3.0, 5.0, 7.0, 10.0]
        slewing_speed_rpm: [0.5, 1.0, 1.5]
        load_group: ["M3", "M4"]
      output_columns:
        - name: slewing_torque_kNm
          unit: kN·m
          type: numeric
        - name: gearbox_ratio
          unit: ""
          type: numeric
        - name: motor_torque_Nm
          unit: N·m
          type: numeric
        - name: motor_speed_rpm
          unit: rpm
          type: numeric
        - name: hydraulic_volume_cm3
          unit: cm³
          type: numeric

    - name: hydraulic_cylinder
      mechanism: "Гидроцилиндр"
      standards:
        - "DNVGL-ST-0194"
        - "DNVGL-ST-0378"
        - "EN 1993-1-1"
      description: >
        Расчёт штока и гильзы гидроцилиндра: выбор диаметра штока
        по устойчивости (Эйлер / DNVGL), проверка прочности гильзы.
      input_parameters:
        rod_force_kN: [100, 200, 300, 500, 800, 1200]
        stroke_mm: [500, 1000, 1500, 2000]
        system_pressure_bar: [200, 250, 300, 350]
        rod_material: ["20MnV6", "42CrMo4"]
      output_columns:
        - name: rod_diameter_mm
          unit: mm
          type: numeric
        - name: cylinder_bore_mm
          unit: mm
          type: numeric
        - name: wall_thickness_mm
          unit: mm
          type: numeric
        - name: buckling_safety_factor
          unit: ""
          type: numeric
        - name: yield_safety_factor
          unit: ""
          type: numeric
```

---

## Pydantic Models

```python
class EngineeringCalculationRule(BaseModel):
    name: str
    mechanism: str                           # human description, passed to LLM as-is
    standards: list[str]                     # e.g. ["FEM 1.001", "DNV-ST-0378"]
    description: str
    input_parameters: dict[str, list[float | str]]
    output_columns: list[OutputColumn]
```

`DataConfig` gains a new field:
```python
class DataConfig(BaseModel):
    ...
    engineering_calculations: list[EngineeringCalculationRule] = []
```

---

## EngineeringCalculator Component

### Constructor (all injected)

```python
class EngineeringCalculator:
    def __init__(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,       # prompts/data/engineering_calculator_system.txt
        user_template: str,       # prompts/data/engineering_calculator_user.txt
        batch_size: int = 20,     # parameter combinations per LLM call
    ) -> None:
```

### Method

```python
def calculated_rows(self, rule: EngineeringCalculationRule) -> list[dict]:
```

---

## Behavior

### Step 1 — Build parameter grid

Cartesian product of all `input_parameters` ranges. Same logic as `StandardsCalculator._parameter_grid()`.

### Step 2 — Batch LLM calls

Grid may be large. Split into batches of `batch_size` rows.
For each batch, send one LLM message.

### Step 3 — LLM contract

**System prompt** establishes the LLM role:
> You are a certified marine and offshore crane engineer.
> You know the following standards by heart and apply them exactly:
> FEM 1.001, DNV-ST-0378, DNVGL-ST-0194, ISO 4301, RMRS.
> When asked to compute a calculation chain, you follow the standard procedure step by step,
> show intermediate variables, and return structured JSON.

**User prompt** per batch:

```
Mechanism: {mechanism}
Standards to apply: {standards}
Task: {description}

Input parameter combinations (JSON array):
{parameter_grid_json}

For each input combination, compute the following output columns:
{output_columns_block}

Rules:
- Follow the exact calculation procedure from the listed standards.
- Show intermediate variables inside each row under key "_steps" (dict of name→value).
- If a combination is physically impossible or out of standard scope, set outputs to null
  and add "_error": "<reason>".
- Return ONLY a JSON array. No markdown, no prose.

Expected schema (one element shown):
{result_schema_example}
```

### Step 4 — Parse & tag rows

```json
[
  {
    "SWL_tonne": 2.75,
    "polispast_ratio": 2,
    "rope_class": "T3",
    "load_group": "M3",
    "rope_tension_kN": 14.2,
    "rope_diameter_mm": 12,
    "drum_diameter_mm": 180,
    "drum_width_mm": 294,
    "drum_torque_nom_kNm": 2.35,
    "motor_speed_rpm": 833.5,
    "hydraulic_volume_cm3": 15.2,
    "system_pressure_bar": 276,
    "_steps": {
      "F_rope_nom": 13.5,
      "F_rope_max": 20.2,
      "k_polispast": 0.97,
      "d_rope_min": 11.4
    }
  }
]
```

Rows are tagged: `source = rule.name`, `source_type = "engineering_calculation"`.
`_steps` columns are kept in dataset (useful for interpretability in ML).

---

## Integration in DataAgent

```python
def run(self) -> None:
    ...
    engineering_rows = self._engineering_rows(cfg)
    ...
    dataset_path, metadata_path = self._assembler.assembled_dataset(
        paper_rows=paper_rows,
        calculation_rows=calc_rows,
        engineering_rows=engineering_rows,
        user_rows=user_rows,
        ...
    )

def _engineering_rows(self, cfg: DataConfig) -> list[dict]:
    if not cfg.engineering_calculations:
        return []
    rows = []
    for rule in cfg.engineering_calculations:
        rows.extend(self._engineering_calculator.calculated_rows(rule))
    return rows
```

`DataAgent.__init__` gains `engineering_calculator: EngineeringCalculator` parameter.

---

## Error Handling

| Condition | Action |
|-----------|--------|
| LLM returns unparseable JSON | log warning, skip batch |
| Row has `_error` key | keep row, log debug |
| All rows null for a rule | log warning, rule produces 0 rows |
| `input_parameters` is empty | log warning, skip rule |

---

## Success Criteria

- [ ] `EngineeringCalculationRule` validated by Pydantic from config YAML
- [ ] Cartesian product grid built correctly
- [ ] LLM called once per batch; batches concatenated
- [ ] `_steps` intermediate values present in output rows
- [ ] `source_type = "engineering_calculation"` on every row
- [ ] DataAgent wired: new calculator injected in `main.py`
- [ ] `dataset_metadata.json` counts `engineering_calculations` rows separately
