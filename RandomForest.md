# Random Forest

---

## Slide 1: Random Forest

Bagging ensemble of decision trees: each tree trains on a bootstrap sample with a random feature subset; predictions are averaged to reduce variance. We train on log₁₊(AQI) and evaluate on raw AQI.

**Why?**
- Captures nonlinear thresholds (e.g., PM2.5 > 150 AND winter → high AQI)
- Same 70-feature pipeline as baselines (pollutants, lags, calendar, city dummies)
- Inverse-frequency sample weighting upweights rare Severe days

**Baseline Parameters:**
- n_estimators = 200, max_depth = None, min_samples_leaf = 5, max_features = 'sqrt'

**Baseline Results:**
- R² = 0.892 | RMSE = 29.78 | MAE = 13.58

**Hyperparameter Search (RandomizedSearchCV + TimeSeriesSplit, 5 folds):**

| Parameter Explored | Values |
|---|---|
| n_estimators | 200, 300, 500 |
| max_depth | 10, 15, 20, 25, None |
| min_samples_leaf | 5, 10, 20 |
| max_features | 'sqrt', 0.3, 0.5 |

- **RandomizedSearchCV:** Samples 20 random configs from the search space — more efficient than exhaustive grid search
- **TimeSeriesSplit (5 folds):** Data split into 6 equal chunks; each fold trains on all prior chunks and validates on the next. Fold 1: train on chunk 1, validate on chunk 2. Fold 5: train on chunks 1–5, validate on chunk 6. Training set grows each fold — prevents future leakage through lag features.

---

## Slide 2: Random Forest (Tuned)

**Tuned Parameters:**
- n_estimators = 300, max_depth = 25, min_samples_leaf = 5, max_features = 0.5

**Before vs After Tuning:**

| Metric | Before | After | Change |
|---|---|---|---|
| RMSE | 29.78 | 22.47 | -24.5% |
| MAE | 13.58 | 11.26 | -17.1% |
| R² | 0.892 | 0.938 | +5.2% |
| MAE (Severe) | 134.46 | 76.19 | -43.3% |

**Visuals:**
- Top drivers: AQI lag-1, PM2.5, PM10, rolling means — model relies on recent pollution and autoregressive structure
- Errors are largest on Severe days (MAE ≈ 76); Good/Moderate are predicted within ~7 AQI points

---

## Script

**Slide 1:**

"Random Forest is a bagging ensemble — we train 200 independent decision trees, each on a bootstrap sample with a random feature subset, and average their predictions. The key advantage over Ridge is nonlinearity: trees naturally learn threshold rules like 'if PM2.5 is above 150 and it's winter, predict high AQI.' Out of the box we get an R-squared of 0.89 and RMSE of about 30. To improve, we ran RandomizedSearchCV with TimeSeriesSplit — 20 configurations across these four parameters, using 5-fold temporal CV so we never leak future data through our lag features."

**Slide 2:**

"After tuning, RMSE dropped 25% from 29.8 to 22.5, and Severe-day MAE dropped 43% from 134 to 76. The biggest improvement came from sample weighting — forcing the model to pay attention to rare Severe days — and increasing max_features from sqrt to 0.5, letting each tree see more of the feature space. The top drivers are yesterday's AQI, PM2.5, and PM10 — the model is essentially an autoregressive forecaster adjusted by current pollution levels."
