# Tree & Boosting Models — Walkthrough and Improvement Plan

**MSE 546 — AQI Forecasting (Group 2)**

---

## 1. What the Tree Models Notebook Does

The `tree_models.ipynb` notebook trains two tree-based regressors — **Random Forest** and **XGBoost** — to predict daily AQI across 26 Indian cities. Both models share the same preprocessing pipeline defined in `preprocessing.py`.

### 1.1 Data Pipeline (shared across all models)

| Step | What happens | Why |
|------|-------------|-----|
| **Load** | Read `city_day.csv`, lowercase columns, parse dates, sort by city+date | Consistent starting point |
| **Impute** | Drop rows with missing AQI; add 12 `_missing` binary flags; fill pollutant NaNs with per-city medians | Preserve sensor-offline signal; median is robust to skew |
| **Features** | 70 features total: 12 pollutants, 12 missing flags, 11 calendar (month, DOW, Fourier, season dummies), 10 lag/rolling (AQI lags 1/7/14/30, rolling mean/std 7/30, PM2.5/PM10 lag1), 25 city dummies | Captures autocorrelation, seasonality, spatial baselines |
| **Target** | Train on `log1p(AQI)`; evaluate on original scale via `expm1` | Reduces right-skew (4.4 → <1), stabilises variance |
| **Split** | Temporal cutoff at 2019-12-01 (79.5% train / 20.5% test) | Prevents future-data leakage |

### 1.2 Random Forest

```
RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_leaf=10, max_features='sqrt',
    random_state=42, n_jobs=-1
)
```

**How it works:** Trains 200 independent decision trees on bootstrap samples, each splitting on a random subset (`sqrt`) of features. Predictions are averaged across all trees, reducing variance compared to a single deep tree.

**Why it helps here:**
- Captures nonlinear interactions (e.g., "if PM2.5 > 150 AND season == winter → high AQI") that Ridge cannot model.
- Robust to outliers and doesn't require feature scaling.
- Averaging smooths out individual tree noise.

**Results:**

| Metric | Value |
|--------|-------|
| RMSE | 30.00 |
| MAE | 13.93 |
| R² | 0.890 |
| MAE (Severe) | 134.6 |

### 1.3 XGBoost

```
XGBRegressor(
    n_estimators=200, max_depth=6,
    learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    objective='reg:squarederror'
)
```

**How it works:** Sequentially adds shallow trees, where each new tree corrects the residual errors of the ensemble so far. Learning rate (0.1) shrinks each tree's contribution to prevent overfitting. Row and column subsampling add regularisation.

**Why it helps here:**
- Gradient boosting is specifically designed to reduce bias — it progressively focuses on the hardest-to-predict samples (e.g., severe AQI days).
- Shallower trees (depth 6 vs 15) with boosting typically outperform deep bagged trees.
- Built-in L1/L2 regularisation in the objective.

**Results:**

| Metric | Value |
|--------|-------|
| RMSE | 21.80 |
| MAE | 11.39 |
| R² | 0.942 |
| MAE (Severe) | 73.8 |

### 1.4 Feature Importance

Both models expose feature importances. The notebook plots the top 20 features for each. Typically, `aqi_lag1`, `pm2.5`, `pm10`, and rolling means dominate — confirming the autoregressive nature of AQI and the dominance of particulate matter.

### 1.5 Optional Hyperparameter Tuning

A commented-out `RandomizedSearchCV` block is provided for Random Forest. It searches over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features` using 3-fold CV. This is left optional because the default hyperparameters already perform well, and temporal CV would be more appropriate than random k-fold.

---

## 2. How Tree Models Compare to Other Approaches

| Model | RMSE | MAE | R² | MAE (Severe) |
|-------|------|-----|-----|--------------|
| Persistence (lag-1) | 44.34 | 21.52 | 0.759 | 162.4 |
| Ridge | 42.02 | 26.20 | 0.784 | 193.1 |
| k-NN | 45.00 | 27.59 | 0.752 | 157.2 |
| LSTM | 46.11 | 27.19 | 0.733 | 164.8 |
| **Random Forest** | **30.00** | **13.93** | **0.890** | **134.6** |
| FeedForward NN | 28.22 | 15.42 | 0.903 | 102.2 |
| **XGBoost** | **21.80** | **11.39** | **0.942** | **73.8** |

**Key takeaways:**
- **XGBoost is the best model across all metrics.** It cuts RMSE by 51% vs persistence and MAE by 47%.
- **Random Forest is strong but trails XGBoost** by ~8 RMSE points — boosting's bias-reduction advantage is clear.
- **Tree models crush linear approaches** (Ridge, k-NN) on tail events: Severe-day MAE drops from 193 → 74.
- **LSTM underperforms** — likely because 26 cities with varying date coverage don't provide enough long contiguous sequences for the recurrent model to learn well.
- **FNN is competitive** on Severe days (MAE 102) but XGBoost still wins by a wide margin.

---

## 3. Weaknesses in `tree_models.ipynb` (now addressed)

The original combined notebook had several weaknesses. The split notebooks (`random_forest.ipynb` and `xgboost_model.ipynb`) address them:

| # | Weakness | Status | How addressed |
|---|----------|--------|---------------|
| 3.1 | No hyperparameter tuning | **Fixed** | `RandomizedSearchCV` with full search spaces in both notebooks |
| 3.2 | Random k-fold CV (leaks future data) | **Fixed** | `TimeSeriesSplit(n_splits=5)` used for all CV |
| 3.3 | Severe-day performance weak | **Fixed** | Inverse-frequency `sample_weight` upweights rare/extreme AQI categories |
| 3.4 | No validation set for early stopping | **Fixed** | Temporal val window (Oct–Nov 2019); XGBoost uses `early_stopping_rounds=30` |
| 3.5 | Single global model | **Open** | Still a global model; per-city/cluster models remain a future improvement |
| 3.6 | No residual analysis | **Fixed** | 4-panel diagnostics + per-city/month breakdowns + time-series overlays |

---

## 4. Improvement Plan

### Priority 1 — Proper Hyperparameter Tuning (High Impact, Moderate Effort)

**What:** Run `RandomizedSearchCV` or `Optuna` for both RF and XGBoost with a temporal validation strategy.

**Search space for XGBoost:**
| Parameter | Range | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300–1000 | Current 200 may underfit |
| `max_depth` | 4–10 | Deeper trees for complex interactions |
| `learning_rate` | 0.01–0.1 | Lower LR + more trees = better generalisation |
| `min_child_weight` | 1–10 | Regularisation for leaf splits |
| `gamma` | 0–0.5 | Minimum loss reduction for splits |
| `subsample` | 0.6–0.9 | Row sampling rate |
| `colsample_bytree` | 0.5–0.9 | Feature sampling rate |
| `reg_alpha` | 0–1 | L1 regularisation |
| `reg_lambda` | 1–5 | L2 regularisation |

**Validation strategy:** Use `TimeSeriesSplit` or a fixed temporal holdout (e.g., Oct–Nov 2019, matching the FNN/LSTM approach) instead of random k-fold.

### Priority 2 — Address Severe-Day Underperformance (High Impact, Moderate Effort)

**Option A — Sample weighting:** Assign higher `sample_weight` to training rows with AQI > 300. XGBoost supports this natively. This forces the model to pay more attention to extreme events.

**Option B — Custom loss function:** Replace `reg:squarederror` with a custom objective that penalises underprediction of high-AQI days more than overprediction. For example, an asymmetric MSE:

```python
def asymmetric_mse(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_true - y_pred
    # penalise underprediction 2x more for high-AQI days
    weight = np.where(residual > 0, 2.0, 1.0)
    grad = -2 * weight * residual
    hess = 2 * weight * np.ones_like(residual)
    return grad, hess
```

**Option C — Two-stage model:** Train one model for AQI classification (Good/Moderate/Severe buckets) and a separate regression model per bucket. The classification stage routes samples to specialised regressors.

### Priority 3 — Temporal Validation Split (Medium Impact, Low Effort)

**What:** Add a validation window (2019-10-01 to 2019-11-30) carved from the training data, consistent with FNN and LSTM notebooks. Use this for:
- XGBoost early stopping (`eval_set` + `early_stopping_rounds`)
- Model selection between RF and XGBoost without touching the test set

```python
VAL_CUTOFF = pd.Timestamp('2019-10-01')
train_sub = full_train_df[full_train_df['date'] < VAL_CUTOFF]
val_sub   = full_train_df[full_train_df['date'] >= VAL_CUTOFF]
```

### Priority 4 — Richer Features (Medium Impact, Medium Effort)

**Additional lag features:**
- Pollutant lags for NO2, SO2, CO, O3 (currently only PM2.5 and PM10 have lag1)
- AQI lag2, lag3 (short-term autoregressive signal)
- Difference features: `aqi_lag1 - aqi_lag7` (trend direction)

**Interaction features:**
- `pm2.5 * is_weekend` (traffic-driven pollution varies by day type)
- `pm2.5 / pm10` ratio (particle size distribution as a proxy for pollution source)

**External data (if available):**
- Temperature, humidity, wind speed (meteorological drivers)
- Holiday calendar (Diwali causes extreme AQI spikes)

### Priority 5 — Ensemble Methods (Medium Impact, Medium Effort)

**Stacking:** Use RF, XGBoost, and FNN predictions as inputs to a simple meta-learner (e.g., Ridge or another XGBoost). This often captures complementary strengths — RF's variance reduction + XGBoost's bias reduction + FNN's smooth function approximation.

**Blending:** Weighted average of XGBoost (0.6) + FNN (0.3) + RF (0.1), tuned on the validation set.

### Priority 6 — Per-City or Clustered Models (Lower Impact, Higher Effort)

**What:** Group cities by pollution profile (e.g., using k-means on mean pollutant levels) and train separate XGBoost models per cluster. Cities like Delhi (extreme pollution) and Aizawl (clean air) have fundamentally different AQI dynamics.

**Risk:** Smaller per-cluster training sets may hurt performance for cities with limited data. Mitigate by falling back to the global model when cluster size < threshold.

### Priority 7 — Residual Analysis and Diagnostics (Low Effort, Useful for Reporting)

Add the following diagnostic plots to the notebook:
- **Residual vs predicted** scatter plot (check heteroscedasticity)
- **Residual by city** box plot (identify cities where the model fails)
- **Residual by month** (seasonal error patterns)
- **Actual vs predicted time series** for top 4 cities (visual sanity check)
- **SHAP values** for XGBoost (model-agnostic feature importance with interaction effects)

---

## 5. Quick-Win Implementation Checklist

- [ ] Add temporal validation split (Priority 3) — 10 min
- [ ] Enable XGBoost early stopping with `eval_set` — 10 min
- [ ] Add sample weights for Severe-day rows (Priority 2A) — 15 min
- [ ] Run Optuna or RandomizedSearchCV with temporal CV (Priority 1) — 30 min
- [ ] Add residual diagnostic plots (Priority 7) — 20 min
- [ ] Add SHAP feature importance plot — 15 min
- [ ] Try stacking ensemble with RF + XGB + FNN (Priority 5) — 30 min

---

## 6. Expected Impact

| Improvement | Estimated RMSE Reduction | Confidence |
|-------------|-------------------------|------------|
| Hyperparameter tuning | 1–3 points | High |
| Early stopping + validation | 0.5–1 point | High |
| Sample weighting for extremes | 1–2 points overall, 15–30 on Severe MAE | Medium |
| Richer features (pollutant lags, interactions) | 1–2 points | Medium |
| Stacking ensemble | 1–3 points | Medium |
| Per-city models | 0–2 points | Low-Medium |

**Realistic target:** With tuning + early stopping + sample weighting + stacking, XGBoost RMSE could drop from 21.8 to approximately **17–19**, with Severe-day MAE improving from 73.8 to approximately **50–60**.
