# Results of the latest training run (auto-generated)

- Trained at (UTC): 2026-10-21T12:54:27+00:00
- Dataset file: `dataset_zone_1_preprocessed.csv`
- Forecast horizon: 24 h; sampling interval: 0 days 00:10:00
- Soil-moisture unit: relative soil-moisture probe output, 0-100 scale (%)
- Selected model (lowest validation RMSE): **Linear Regression**
- Splits (chronological, with a purge gap of one horizon): train: 4436 rows (2025-03-17 to 2025-08-20); val: 838 rows (2025-08-21 to 2025-08-26); test: 981 rows (2025-08-27 to 2025-09-03)

| model                |   val_MAE |   val_RMSE |   val_R2 |   test_MAE |   test_RMSE |   test_R2 |
|:---------------------|----------:|-----------:|---------:|-----------:|------------:|----------:|
| Persistence baseline |    0.3091 |     0.8677 |  -0.5131 |     0.2951 |      1.3267 |   -0.9978 |
| Linear Regression    |    0.4577 |     0.7148 |  -0.0266 |     0.7369 |      1.2207 |   -0.6911 |
| Random Forest        |    0.5368 |     0.8273 |  -0.3752 |     0.9934 |      1.3476 |   -1.0611 |
| Gradient Boosting    |    0.7872 |     1.0137 |  -1.0648 |     1.4024 |      1.7332 |   -2.4093 |

Lower MAE/RMSE is better; R2 closer to 1 is better. Test metrics are from held-out, later-in-time data.
