# Viva questions and model answers

Answers are concise and describe **this** project. Quote measured numbers only from your own training run.

## Dataset

**1. Which dataset do you use and why?**
The Mendeley "Soil Moisture, Irrigation Actuator and Weather Dataset from a Multi-Sector Precision-Irrigation" (CC BY 4.0). It has soil-moisture time series, applied irrigation water and weather on a regular grid for a full season, which suits 24 h-ahead forecasting.

**2. Why did you not hard-code the column names?**
We could not rely on an assumed schema. The loader detects columns by name patterns, prints the mapping for verification, and allows overrides in `config.local.json`.

**3. What is the sampling interval and why does it matter?**
It is the median spacing between timestamps (the dataset description states 10 minutes for the processed series; the loader verifies it). It decides how far lags reach and how the 24 h target is located.

**4. How do you handle duplicates, missing and invalid values?**
Duplicate timestamps are averaged, unparseable timestamps dropped, non-numeric values become NaN, impossible values (negative moisture, humidity outside 0-100) are blanked, and remaining gaps are imputed with training-set medians inside the pipeline.

**5. Why are the dataset's own "24 h target" columns not used as inputs?**
They are derived from future data. Using them as features would leak the answer. We build our own target from timestamps and only compare against the supplied one for validation.

## Problem formulation and features

**6. Is this classification or regression? Why?**
Regression: soil moisture is continuous. We predict its value 24 h ahead.

**7. How is the target built?**
For each time *t*, the target is the soil-moisture observation closest to *t* + 24 h (within half an interval). Rows without such an observation are dropped, so gaps never create wrong labels.

**8. Which features do you use?**
Current and lagged soil moisture (1-24 h), changes over those lags, rolling means, past temperature/humidity, past precipitation sums, evapotranspiration and irrigation sums where available, and sine/cosine of hour and day of year.

**9. Why encode time with sine and cosine?**
Hour 23 and hour 0 are close in time but far apart numerically. Sine/cosine put the cycle on a circle so the model sees that closeness.

**10. Why is current soil moisture such a strong feature?**
Soil moisture changes slowly, so tomorrow's value is usually close to today's. That is why persistence is a strong baseline.

## Models

**11. What is the persistence baseline and why include it?**
It predicts that soil moisture in 24 h equals the current value. Any ML model must beat it to justify its complexity.

**12. How does Linear Regression work and what are its limits here?**
It fits a weighted sum of the features by least squares. It cannot capture non-linear effects such as saturation, or interactions like rain only mattering when the soil is dry.

**13. How does a Random Forest regressor work?**
It averages many decision trees, each trained on a bootstrap sample and using random feature subsets at each split. Averaging reduces variance and handles non-linearity and interactions without scaling.

**14. What are the drawbacks of Random Forest for time series?**
It cannot extrapolate beyond the target range seen in training, ignores temporal order unless we supply lag features, and can overfit if the training period is unrepresentative.

**15. Why add Gradient Boosting?**
It builds trees sequentially, each correcting the previous errors. It gives a different bias-variance trade-off for comparison at little extra code.

**16. Which hyper-parameters did you set and did you tune them?**
Modest fixed values (e.g. 150 trees, minimum 5 samples per leaf). We did not run an extensive search; this is a limitation and a possible extension using time-series cross-validation.

**17. How is the final model chosen?**
By lowest RMSE on the validation split among the three ML models. The test split is used once afterwards for the reported result.

## Evaluation

**18. Explain MAE, RMSE and R².**
MAE is the average absolute error. RMSE is the square root of the mean squared error and penalises large errors more. R² is the fraction of variance explained relative to predicting the mean; it can be negative.

**19. Why report RMSE and MAE together?**
If RMSE is much larger than MAE, a few large errors dominate.

**20. What does a negative R² mean?**
The model is worse than always predicting the test-set mean. With a slowly varying target and a narrow test period this can happen even when MAE looks small.

**21. What if the ML model does not beat persistence?**
Then we report that honestly. It means the extra features add no measurable value on this data. The training script flags the case.

## Time-series methodology and leakage

**22. Why not use a random train/test split?**
Neighbouring samples are highly correlated. Random splitting puts near-duplicates of test points in the training set and trains on the future, giving over-optimistic scores.

**23. How do you split the data?**
Chronologically: the first 70% for training, the next 15% for validation and the final 15% for testing.

**24. What is the purge gap?**
A sample at time *t* has its target at *t*+24 h. Without a gap, the targets of the last training samples would lie inside the validation period. We remove the last 24 h of training (and validation) rows.

**25. What is data leakage? Give an example from this project and how you prevent it.**
Information from the future or from the answer entering the model. Examples: using the dataset's pre-computed target column as a feature, or fitting the imputer on all data. We use past-only features (tested), exclude target-like columns, and fit preprocessing on the training split only.

**26. Why is preprocessing inside the Pipeline?**
So the imputer/scaler statistics come from the training data only and are reused unchanged for validation, test and live predictions.

## Weather API and recommendation

**27. Which weather API do you use and what does it return?**
Open-Meteo (no API key). The forecast endpoint gives hourly temperature, humidity, precipitation, precipitation probability and more for any latitude/longitude; a separate archive endpoint gives past reanalysis data.

**28. Did the model use weather forecasts in training? Why does that matter?**
No. It was trained on past weather measured in the dataset. Training with actual future weather and then using forecasts at run time would overstate accuracy, because real forecasts have errors. Forecasts only inform the recommendation rules.

**29. How does the app behave when the weather API fails?**
The error is caught and shown, ML predictions still work, and the recommendation is made without rain information, never assuming rain, with a warning to check the weather.

**30. How does the recommendation engine decide?**
Configurable rules: insufficient information if no threshold/value; monitor if moisture is not below the dry threshold now or in 24 h; if dry and meaningful rain (amount and probability thresholds) is forecast, consider delaying, unless moisture is critically low; otherwise irrigation may be needed.

**31. Why are the rules separate from the ML model, and why no trained classifier for the recommendation?**
We have no real, validated labels for "should irrigate". Training on invented labels would be meaningless. Transparent rules are testable, explainable and adjustable per crop.

**32. Why does the app not give water quantities or switch the pump?**
Quantities need crop, soil and system data we do not have, and unvalidated advice could harm crops. The project is decision support only.

## Limitations and future work

**33. What are the main limitations?**
One farm and season, thresholds not agronomically calibrated, sensor noise and drift, forecast uncertainty, limited hyper-parameter tuning, and a column mapping that depends on the real data dictionary.

**34. How would you integrate it with an ESP32 IoT system?**
Log timestamp, soil moisture, temperature and humidity (same column names as the training data) and call `predict_from_history` with at least 24 h of readings; units and sensor calibration must match the training data.

**35. What would you improve with more time?**
Time-series cross-validation and tuning, training with archived forecast data, uncertainty intervals, per-crop calibration of thresholds with an agronomist, multi-sector models.
