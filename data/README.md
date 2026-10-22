# Data folder

The dataset is **not included** in this repository. Download it yourself and place it here.
Everything in `data/` except this file is git-ignored.

## Primary dataset

**Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation**

- Authors: Tommaso Adamo, Lucio Colizzi, Giovanni Dimauro, Emanuela Guerriero, Nunzia Lomonte
- Publisher: Mendeley Data, Version 1
- DOI: [10.17632/c837v6p8ph.1](https://doi.org/10.17632/c837v6p8ph.1)
- Page: <https://data.mendeley.com/datasets/c837v6p8ph/1>
- Licence: **CC BY 4.0** (attribution required - cite the dataset in any report or publication)

### What the dataset page says it contains

Taken from the dataset's public description (the files themselves could not be opened when this
project was generated, so the **real column names are not yet verified** - see step 3 below):

- One growing season (February-September 2025) at an IoT-instrumented farm in Arnesano (Lecce, Apulia, Italy).
- Five sectors: open-field tomato (x2), potted tomato, zucchini, blueberry.
- A raw layer (sensor measurements and valve states), a processed layer, and a code layer.
- The processed layer is described as including **five per-sector time series on a 10-minute grid
  (33,259 rows each)** combining sensors, applied water volume and ERA5 reanalysis weather, plus
  **five analysis-ready tables with engineered features and 24-hour-ahead soil-moisture targets**.
- A `README.md` and a `DATA_DICTIONARY.md` (column-by-column reference) ship with the dataset.

## How to set it up

1. Open the dataset page and click **Download All** (a ZIP file).
2. Extract the ZIP **into this `data/` folder**. Sub-folders are fine; the loader searches recursively.
   Example: `data/<extracted-folder>/processed/...csv`.
3. Read the dataset's own `DATA_DICTIONARY.md` and note the **soil-moisture unit** and the **precipitation unit**
   (per-interval amount vs rate vs cumulative). Then check what the project detects:

   ```powershell
   python -m src.data_loader
   ```

   This lists candidate CSV files, the detected column for each role (timestamp, soil moisture,
   temperature, humidity, precipitation, evapotranspiration, irrigation), the sampling interval,
   missing values and any column that looks like a pre-computed future target.
4. If the wrong file or column is picked, copy `config.local.example.json` to `config.local.json`
   and set `data_file`, `column_overrides`, `sector_column` / `sector_value`, and `soil_moisture_unit`.
5. Train: `python -m src.train_model`

## Which file to use

Use **one sector's time-series CSV** (regular grid of sensor + weather + applied-water columns).

- Do **not** use the raw layer or the irrigation-event table as the model input.
- The "analysis-ready" tables already contain lags, rolling statistics and target columns. This project builds
  its own features and target from timestamps, so such columns are deliberately **never used as inputs**
  (they would risk target leakage). If you point `data_file` at such a table, the loader still maps only the raw
  columns and reports the target-like columns it ignored. Its supplied target can be checked against the
  timestamp-derived one with `src.data_loader.validate_existing_target`.
- **Sector choice (suggestion, not verified):** start with one open-field sector, because rainfall can only
  influence sectors exposed to it, and the project combines the forecast with the prediction. Record the sector
  you choose in `docs/methodology.md` ("Dataset subset") and justify it after you have read the data dictionary.

## Alternative dataset (supported through the same mechanism, not tested)

Bruni, M., Selicato, L., Vivaldi, G. A., Vurro, M., Berardi, M. - *Time-Series Analysis and Learning Approaches for
Soil Moisture Prediction in Precision Irrigation: Dataset and Code*, Zenodo, DOI
[10.5281/zenodo.15745040](https://doi.org/10.5281/zenodo.15745040), CC BY 4.0. Gallipoli (southern Italy), soil water content
at 30 and 60 cm every 15 minutes, precipitation, temperature and irrigation events (`data.zip`, ~19.7 MB).
It has no humidity column, so humidity features are simply omitted. Choose one depth via `column_overrides`.
The Zenodo page notes a newer version exists. If you switch datasets, say so in your report.

## Limitations to remember

- One farm and one season: results describe that site only.
- Several sensors/depths/sectors exist; each model is trained on **one** series.
- Sensor dropouts, flatlines and spikes exist in raw data (per the dataset description); the processed series
  may already be cleaned. This project additionally blanks impossible values and merges duplicate timestamps.
