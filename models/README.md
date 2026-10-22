# Models folder

`python -m src.train_model` writes two files here:

| File | Content |
|------|---------|
| `best_model.joblib` | The selected scikit-learn `Pipeline` (median imputer fitted on the training split, optional scaler, regressor). |
| `model_metadata.json` | Feature names and training statistics, column mapping, forecast horizon, chronological split dates, **all measured metrics**, permutation importance, library versions, training time. |

Both are **git-ignored** (a Random Forest can be tens of MB and is reproducible from the data).
If your instructor needs the trained model, share it separately or use Git LFS.

The dashboard and `src.predict.load_model()` need both files. If they are missing, the app shows
"model training pending" and nothing is invented.

A joblib file can execute code when loaded - only load model files you trained yourself or fully trust.
