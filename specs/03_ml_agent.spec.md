# Spec — MLAgent

## Responsibility

Load the user's ML library, fit a GLM model on the prepared dataset, and export results and figures.

## Inputs

| Source | Key | Type | Description |
|--------|-----|------|-------------|
| RunContext | `artifacts.dataset` | path | CSV dataset from DataAgent |
| RunContext | `artifacts.dataset_metadata` | path | Column metadata |
| agent_config.yaml | `ml.library` | str | Python module name of user's ML library |
| agent_config.yaml | `ml.model` | str | Model class name within library |
| agent_config.yaml | `ml.target_variable` | str | Name of target column |
| agent_config.yaml | `ml.features` | list[str] | Feature columns (empty = all non-target) |
| agent_config.yaml | `ml.hyperparameters` | dict | Passed directly to model constructor |

## Outputs

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| `model_results` | `output/model_results.json` | JSON | Coefficients, metrics, diagnostics |
| `figures_dir` | `output/figures/` | PDF/PNG | Diagnostic and result plots |

## model_results.json Structure

```json
{
  "model": "GLM",
  "library": "mylib",
  "target": "col_name",
  "features": ["f1", "f2"],
  "coefficients": {
    "intercept": 0.0,
    "f1": 0.0,
    "f2": 0.0
  },
  "metrics": {
    "aic": 0.0,
    "bic": 0.0,
    "r2": 0.0,
    "deviance": 0.0
  },
  "diagnostics": {
    "p_values": { "f1": 0.0 },
    "confidence_intervals": { "f1": [0.0, 0.0] },
    "residuals_normality_p": 0.0
  }
}
```

## Expected figures

| File | Description |
|------|-------------|
| `figures/coef_plot.pdf` | Coefficient plot with confidence intervals |
| `figures/residuals.pdf` | Residual vs fitted plot |
| `figures/qq_plot.pdf` | QQ-plot of residuals |
| `figures/feature_importance.pdf` | Feature importance / effect sizes |

## Behavior

1. Load dataset and metadata
2. Dynamically import `ml.library` module
3. Instantiate model: `model = library.ml.model(target=..., features=..., **hyperparameters)`
4. Fit model on dataset
5. Extract coefficients, metrics, diagnostics
6. Generate figures via model's plotting interface or matplotlib fallback
7. Write `model_results.json` and figures
8. Update RunContext

## Integration Contract with User's Library

The user's library must expose:

```python
# Minimum required interface
model = MyLibrary.GLM(target="y", features=["x1", "x2"], **kwargs)
model.fit(df: pd.DataFrame)
model.coefficients()   → dict
model.metrics()        → dict
model.diagnostics()    → dict
model.plot(output_dir: str)   # optional — fallback to matplotlib if absent
```

## Error Handling

- If `ml.library` cannot be imported → raise `LibraryImportError` with module name
- If model fitting fails → raise `ModelFitError`, preserve traceback
- If `plot()` not available → generate standard matplotlib fallback plots

## Success Criteria

- [ ] `model_results.json` contains all required keys
- [ ] All 4 figures generated in `output/figures/`
- [ ] No silent NaN in coefficients
- [ ] RunContext updated with artifact paths and status = "completed"
