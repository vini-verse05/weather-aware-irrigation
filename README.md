# Weather-Aware Smart Irrigation Prediction Using Machine Learning

A software-only machine-learning project that **predicts soil moisture 24 hours ahead** from a public precision-irrigation
dataset and combines that prediction with a live **Open-Meteo weather forecast** to give an explained irrigation
suggestion in a Streamlit dashboard. No ESP32, pump or other hardware is needed, and **nothing is ever switched automatically**.

> **Project status**
> - Code, tests, dashboard and documentation are complete.
> - **Model training on the real dataset is PENDING.** The dataset must be downloaded by you (see [Dataset setup](#dataset-setup));
>   this project's results section stays empty until you run training. No performance figures are claimed anywhere in this repository.

## Contents
1. [Problem statement and objectives](#problem-statement-and-objectives)
2. [Dataset](#dataset)
3. [Installation and setup (Windows / VS Code)](#installation-and-setup-windows--vs-code)
4. [Dataset setup](#dataset-setup)
5. [Running the project](#running-the-project)
6. [Project structure](#project-structure)
7. [Preprocessing and feature engineering](#preprocessing-and-feature-engineering)
8. [Machine-learning algorithms](#machine-learning-algorithms)
9. [Evaluation metrics](#evaluation-metrics)
10. [Experimental results](#experimental-results)
11. [Weather API integration](#weather-api-integration)
12. [Irrigation recommendation](#irrigation-recommendation)
13. [Dashboard](#dashboard)
14. [Testing](#testing)
15. [Limitations](#limitations)
16. [Future enhancements](#future-enhancements)
17. [Licence](#licence)
18. [References](#references)

## Problem statement and objectives

Fixed irrigation schedules waste water or stress crops. Knowing tomorrow's soil moisture and whether rain is coming supports
better decisions.

The application answers:
1. What will soil moisture be in 24 hours?
2. Will it increase or decrease?
3. Is rainfall expected soon?
4. Should the user consider irrigation, delay it because of rain, or keep monitoring?
5. Which inputs influence the prediction?

Objectives: 24 h-ahead regression; compare a persistence baseline, Linear Regression, Random Forest and Gradient Boosting with
leakage-safe time-series methodology; integrate a public weather API; keep recommendation rules separate, configurable and
explained; stay ready for later integration with a separate IoT logger.

## Dataset

**Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation**
Adamo, T., Colizzi, L., Dimauro, G., Guerriero, E., Lomonte, N. - Mendeley Data, V1,
[doi:10.17632/c837v6p8ph.1](https://doi.org/10.17632/c837v6p8ph.1) - **CC BY 4.0** (please cite it in reports).

Per the dataset page: a full growing season (Feb-Sep 2025) on an IoT-instrumented farm in Arnesano (Lecce, Italy), five sectors
(open-field tomato x2, potted tomato, zucchini, blueberry), raw sensor/valve exports, a processed layer with per-sector 10-minute
time series (sensors + applied water + ERA5 reanalysis weather) and analysis-ready tables with 24 h-ahead targets.
Alternative: [Zenodo 10.5281/zenodo.15745040](https://zenodo.org/records/15745040) (Gallipoli, 15-min data, CC BY 4.0).

The dataset is **not redistributed** here. The files could not be opened when this project was generated, so the real column
names, units and the chosen sector are **not yet verified**. The loader detects columns by name patterns and prints them for you to
check against the dataset's `DATA_DICTIONARY.md`. Details and limitations: [`data/README.md`](data/README.md).

## Installation and setup (Windows / VS Code)

Requirements: Python 3.10 or newer (python.org installer, "Add python.exe to PATH" ticked) and VS Code.

1. Extract the project, then in VS Code: **File > Open Folder...** and choose `weather-aware-irrigation`. Open a terminal (**Terminal > New Terminal**, PowerShell).
2. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Activate it (PowerShell):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   *If PowerShell refuses with "running scripts is disabled"*, either allow scripts for this terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

   or skip activation and call the environment's Python directly (fallback for **every** command below):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m src.train_model
   .\.venv\Scripts\python.exe -m pytest -q
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

   (In `cmd.exe` use `.venv\Scripts\activate.bat`.)
4. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. In VS Code choose the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

## Dataset setup

1. Download the dataset ZIP from <https://data.mendeley.com/datasets/c837v6p8ph/1> (**Download All**).
2. Extract it into the `data/` folder (sub-folders are fine).
3. Inspect what the project found:

   ```powershell
   python -m src.data_loader
   ```

   Check the printed file, column mapping, sampling interval, missing values and ignored target-like columns against the
   dataset's `DATA_DICTIONARY.md`. If anything is wrong, copy `config.local.example.json` to `config.local.json` and set `data_file`,
   `column_overrides`, `sector_column`/`sector_value` and `soil_moisture_unit`.
4. Record the chosen subset in `docs/methodology.md`.

## Running the project

```powershell
python -m src.train_model        # train + evaluate (needs the dataset); writes models/ and reports/
python -m pytest -q              # run the automated tests (no internet or dataset needed)
python -m streamlit run app.py   # launch the dashboard
```

Without the dataset, `python -m src.train_model` prints "TRAINING PENDING" and exits with code 2; the dashboard still starts and
explains what is missing. Optional: `python -m src.evaluate_model` regenerates the figures from saved results.

## Project structure

```
weather-aware-irrigation/
  app.py                        Streamlit dashboard (7 sections)
  requirements.txt  pytest.ini  .gitignore  LICENSE  config.local.example.json
  data/README.md                dataset download and placement instructions (data itself is git-ignored)
  models/README.md              trained artifacts are written here (git-ignored)
  reports/README.md             where results/plots live
  notebooks/01_data_exploration.ipynb
  src/
    config.py                   paths, constants, column-role patterns, API settings
    data_loader.py              discovery, column mapping, validation, cleaning, inspection CLI
    preprocessing.py            imputation/scaling pipeline, input validation
    feature_engineering.py      past-only features, timestamp-based target, chronological split
    train_model.py              reproducible training and model selection
    evaluate_model.py           metrics, tables, plots
    predict.py                  model loading, validated prediction, prediction from history
    weather_service.py          Open-Meteo forecast/historical client, cache, rainfall summary
    irrigation_recommendation.py  rule-based, configurable, explained recommendation
  tests/                        pytest suite (mocked HTTP, synthetic fixtures only)
  docs/                         methodology.md, viva_questions.md, team_contributions.md
```

## Preprocessing and feature engineering

- Sort chronologically, parse timestamps, merge duplicate timestamps, blank impossible values, verify the sampling interval.
- **Target:** soil moisture at *t* + 24 h, looked up by timestamp (rows with no such observation are dropped).
- **Features (past/current only):** current and lagged soil moisture (1, 3, 6, 12, 24 h), changes, rolling means, past temperature,
  humidity, precipitation sums, evapotranspiration and irrigation sums when those columns exist, hour/day-of-year as sine/cosine.
- **Leakage controls:** past-only features (unit-tested), no dataset-supplied target/engineered columns as inputs, chronological
  70/15/15 split with a 24 h purge gap, imputer/scaler fitted on training data only, test set evaluated after model selection.
- Future *actual* weather is never used as a substitute for a forecast. Full detail: [`docs/methodology.md`](docs/methodology.md).

## Machine-learning algorithms

Persistence baseline (current value), Linear Regression (standardised), Random Forest Regressor, Gradient Boosting Regressor
(scikit-learn, `random_state=42`). The best ML model is the one with the lowest **validation** RMSE. Hyper-parameters are
modest fixed values, not tuned.

## Evaluation metrics

MAE, RMSE (both in soil-moisture units) and R². Outputs: comparison table, actual-vs-predicted plots, residual plots and
permutation feature importance, saved to `reports/` and shown in the dashboard.

## Experimental results

**Pending - not yet trained on the real dataset.** After you run `python -m src.train_model`, copy the table from
`reports/results_summary.md` here, together with the dataset file, sector and date. Report only numbers produced by your own run, and say
so if the selected model does not beat the persistence baseline.

## Weather API integration

[Open-Meteo](https://open-meteo.com/) public endpoints, no API key:

- Forecast: `https://api.open-meteo.com/v1/forecast` - hourly temperature, relative humidity, precipitation, precipitation probability, evapotranspiration, wind (future values).
- Historical: `https://archive-api.open-meteo.com/v1/archive` - past reanalysis (labelled as historical in the UI; the archive lags real time by several days).
- User-editable latitude/longitude (default: the dataset's farm, approximate); the location, timezone, fetch time and units are displayed.
- Timeouts, retries for transient failures, HTTP/JSON error handling, missing-variable detection, and a 30-minute file cache in `.cache/` (git-ignored).
- If the API is unreachable the dashboard shows the reason; offline predictions keep working and the recommendation never assumes rain.
- The model is trained on past weather from the dataset; **forecasts are used only by the recommendation rules**, not as model inputs.
- Open-Meteo's free API is intended for non-commercial use with fair-use limits; check their terms and attribute them in reports.

## Irrigation recommendation

Separate module with transparent rules and configurable thresholds (dry, optional critical, rain amount, rain probability, rain window).
Outputs: *Monitor*, *Irrigation may be needed*, *Consider delaying because rainfall is forecast*, or *Insufficient information*, each with
reasons and warnings. **Thresholds are not universal**: they need crop-specific and soil-specific calibration. The default values pre-filled in the
dashboard are percentiles of the training data for illustration only. No water quantities are suggested and no equipment is controlled.

## Dashboard

`python -m streamlit run app.py` - sections: **Overview** (objectives, workflow, dataset, model status), **Soil-Moisture Prediction** (input form,
24 h prediction, direction, model info, validation), **Weather Forecast** (location, 24 h charts, units), **Irrigation Recommendation** (explained decision,
warnings), **Dataset Analysis** (summary, missing values, distributions, trends, relationships), **Model Evaluation** (comparison, plots, importance,
interpretation generated from actual numbers), **Project Information**. Missing dataset, missing model and API failures produce clear messages.

## Testing

```powershell
python -m pytest -q
```

Covers dataset validation, invalid/missing values, target construction, chronological splitting and leakage safety, prediction and input validation,
model loading and missing/corrupt artifacts, weather API error handling and missing fields (all HTTP mocked), recommendation logic and invalid thresholds,
and an end-to-end training run. Tests use **synthetic fixtures** (`tests/helpers.py`) purely to exercise the software; their numbers are meaningless and
must never be reported as model performance. The Streamlit UI itself is not covered by automated tests; check it manually against real data.

## Limitations

- One farm and season; one sector per trained model; results may not transfer.
- Column mapping is heuristic and unit/precipitation semantics must be confirmed from the data dictionary.
- The model never sees future weather; rain/irrigation after the prediction time is unknown to it.
- Forecast uncertainty is not modelled; moisture thresholds are uncalibrated; no confidence intervals.
- Limited hyper-parameter tuning; one chronological split rather than rolling-origin validation.
- Not a validated agronomic advisory tool.

## Future enhancements

IoT integration via `predict_from_history`; training with archived forecasts; rolling-origin cross-validation and tuning; prediction intervals;
per-crop calibration with an agronomist; multi-sector or transfer models.

## Licence

The code is released under the **MIT licence** (see `LICENSE`) as a common default for coursework. **Before publishing, replace the placeholder names
in `LICENSE` and confirm the choice** with your team and your university's coursework/IP policy; alternatives include Apache-2.0 (adds a patent grant)
or GPL-3.0 (requires derivatives to stay open) or no licence (all rights reserved). This is not legal advice. The dataset has its own licence
(CC BY 4.0) and Open-Meteo has its own terms; neither is covered by this project's licence.

## References

- Adamo, T., Colizzi, L., Dimauro, G., Guerriero, E., Lomonte, N. *Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation.* Mendeley Data, V1. doi:10.17632/c837v6p8ph.1
- Bruni, M., Selicato, L., Vivaldi, G. A., Vurro, M., Berardi, M. *Time-Series Analysis and Learning Approaches for Soil Moisture Prediction in Precision Irrigation: Dataset and Code.* Zenodo. doi:10.5281/zenodo.15745040
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API.* https://open-meteo.com
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825-2830.
- Breiman, L. (2001). Random Forests. *Machine Learning* 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29(5), 1189-1232.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146, 1999-2049.
