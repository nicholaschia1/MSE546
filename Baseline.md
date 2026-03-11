# Baseline Models

---

## Slide 1: Persistence Baseline (Lag-1)

Simplest possible forecast: predict tomorrow's AQI = today's AQI. No model, no training — just the lag-1 feature. Any useful model must beat this.

**Why start here?**
- AQI is highly autocorrelated — today's AQI is a strong predictor of tomorrow's
- Sets the floor: if a model can't beat "just repeat yesterday," it's not learning anything useful
- Same test set (Dec 2019 – Jul 2020) as all other models for fair comparison

**Parameters:**
- None — prediction = aqi_lag1

**Results:**
- R² = 0.759 | RMSE = 44.34 | MAE = 21.52

**MAE by AQI Category:**

| Category | MAE | n |
|---|---|---|
| Good | 7.01 | 535 |
| Satisfactory | 10.82 | 2,116 |
| Moderate | 24.58 | 1,746 |
| Poor | 50.29 | 406 |
| Very Poor | 41.80 | 236 |
| Severe | 162.36 | 61 |

**Plot:** `plots/persistence_delhi_timeseries.png` — Delhi test set: Actual AQI vs yesterday's AQI, with red dots highlighting days where the error exceeds 100 AQI points

**Key insight:** Persistence is surprisingly strong (R² = 0.76) because AQI changes slowly day-to-day. But it completely fails on Severe days (MAE ≈ 162) — it can't anticipate sudden spikes.

---

## Slide 2: Ridge Regression (Linear Baseline)

L2-regularized linear model on all 70 features, trained on log₁₊(AQI) and evaluated on original AQI scale. Features are standardized before fitting.

**Parameters:**
- alpha = 1.0
- StandardScaler on all 70 features
- Trained on log₁₊(AQI), evaluated via expm1

**Results:**
- R² = 0.784 | RMSE = 42.02 | MAE = 26.20

**Plot:** `plots/baseline_comparison.png` — Predicted vs Actual scatter for Persistence (left) and Ridge (right) side by side, with metrics in subtitles

**Key insight:** Ridge has better R² than persistence (0.784 vs 0.759) but *worse* MAE (26.2 vs 21.5). It over-smooths predictions toward the mean — catastrophic on Severe days (MAE ≈ 193, worse than just repeating yesterday). This motivates nonlinear models like Random Forest and XGBoost.

---

## Script

**Slide 1 (Persistence):**

"Our first baseline is the simplest possible: predict tomorrow's AQI equals today's AQI. No model, no training — just repeat yesterday. Surprisingly, this gives an R-squared of 0.76 because AQI is highly autocorrelated. But it completely fails on Severe days — it's off by 162 AQI points on average because it can't anticipate sudden spikes. Any useful model has to beat this."

**Slide 2 (Ridge):**

"Our second baseline is Ridge regression — a regularized linear model on all 70 features. It improves R-squared to 0.78, but actually has worse MAE than persistence. Ridge smooths everything toward the mean, which is catastrophic for extremes — it's off by 193 points on Severe days, even worse than just repeating yesterday. This tells us we need nonlinear models that can capture threshold interactions, which is exactly what Random Forest and XGBoost do."
