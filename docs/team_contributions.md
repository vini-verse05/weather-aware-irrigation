# Team contributions

Fill in your names. Both members must be able to explain the **whole** project (see `viva_questions.md`);
the split below says who leads each part and reviews the other's.

| | Member 1 | Member 2 |
|---|----------|----------|
| Name / roll no. | _<name>_ | _<name>_ |

## Member 1 - data and machine learning

| Area | Files | Tasks |
|------|-------|-------|
| Dataset inspection and preprocessing | `src/data_loader.py`, `src/preprocessing.py`, `data/README.md` | Download the dataset, read its data dictionary, run `python -m src.data_loader`, confirm the column mapping and units, choose and justify the sector/subset, document it in `docs/methodology.md`. |
| Feature engineering | `src/feature_engineering.py` | Verify the sampling interval, review lags/rolling windows, check target construction and leakage safety, adjust `config.py` if needed. |
| Model training | `src/train_model.py` | Run training, compare models, inspect the selected model, keep the run reproducible. |
| Model evaluation | `src/evaluate_model.py`, `reports/` | Interpret MAE/RMSE/R² honestly (including whether the baseline wins), produce plots, write the Results section from `reports/results_summary.md`. |
| Tests | `tests/test_data_loader.py`, `tests/test_features.py`, `tests/test_training_and_evaluation.py` | Extend with real-data sanity checks. |

## Member 2 - weather, recommendation, application

| Area | Files | Tasks |
|------|-------|-------|
| Weather API integration | `src/weather_service.py` | Study the Open-Meteo docs, test with several coordinates, check units and timezone handling, confirm failure behaviour offline. |
| Recommendation engine | `src/irrigation_recommendation.py` | Explain every rule, discuss threshold calibration with the instructor, extend the tests. |
| Streamlit dashboard | `app.py` | Run every section against the real model/dataset/API, polish wording, capture screenshots for the report. |
| Integration testing and documentation | `tests/test_prediction.py`, `tests/test_weather_service.py`, `tests/test_recommendation.py`, `README.md`, `docs/` | Run the complete test-suite, verify the Windows commands on a clean machine, complete the README Results section. |

## Shared

- Review each other's pull requests; use short feature branches and meaningful commit messages.
- Prepare the viva together: each member explains the other's half once.
- Decide the open-source licence for the code together (see the README licence note).
