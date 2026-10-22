# Reports folder

Store experiment evidence for your lab report here.

## Generated automatically by `python -m src.train_model`

| Path | Content |
|------|---------|
| `model_comparison.csv` | Validation and test MAE / RMSE / R² for every model, including the persistence baseline. |
| `test_predictions.csv` | Timestamp, actual, model prediction and persistence prediction for the test period. |
| `results_summary.md` | Markdown summary of the latest run (copy it into the README's *Results* section). |
| `figures/` | Actual-vs-predicted (time series + scatter), residual plots, permutation importance, model comparison. |

Regenerate the figures without retraining: `python -m src.evaluate_model`.

## Suggested practice

1. Keep one sub-folder per experiment you want to preserve, e.g. `reports/2026-11-02_sectorA_baseline/`, and copy
   `model_comparison.csv`, `results_summary.md`, `figures/` and `models/model_metadata.json` into it.
2. Note the data file, sector, any `config.local.json` overrides, the commit hash and the date next to the results.
3. Never edit numbers by hand and never paste numbers from the synthetic test fixtures; report only values
   produced by training on the real dataset.
4. Git-ignored by default: `figures/`, `model_comparison.csv`, `test_predictions.csv`. Remove the corresponding lines in
   `.gitignore` (or use `git add -f`) for the experiments you want in the repository.
5. Report material that belongs here: screenshots of the dashboard, the data inspection output
   (`python -m src.data_loader > reports/data_inspection.txt`), and your interpretation notes.
