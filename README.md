# India Air Quality Index (AQI) Forecasting

Machine learning project for predicting Air Quality Index across 26 Indian cities using temporal and pollutant features.

**Course:** MSE 546 — Advanced Machine Learning
**Team:** Group 2
**Members:** Nicholas Chia, Callum Gillies, Matthew Ho, Abhishek Srikaran, Agishan Thaya
**Date:** January 2026

---

## Project Overview

This project develops predictive models for the Air Quality Index (AQI) across India's National Capital Region (NCR) and 25 other major cities. Using daily air quality measurements from 2015-2020, we implement and compare a range of models — from linear baselines to tree ensembles and deep learning architectures — evaluating their ability to capture temporal patterns, spatial variation, and pollutant dynamics.

**Key Objectives:**
- Predict daily AQI values for 26 Indian cities
- Identify critical pollutants and temporal patterns affecting air quality
- Benchmark a diverse set of models from simple baselines to neural networks
- Address data quality issues (missing values, interpolation artifacts)

---

## Dataset

**Source:** `city_day.csv`
**Coverage:** 29,531 rows × 16 columns
**Time Period:** January 1, 2015 → July 1, 2020
**Cities:** 26 Indian cities including Delhi, Mumbai, Chennai, Bengaluru, Hyderabad, and others

### Features

**Pollutants (µg/m³):**
- Particulate Matter: PM2.5, PM10
- Nitrogen compounds: NO, NO2, NOx, NH3
- Gases: CO, SO2, O3
- Volatile compounds: Benzene, Toluene, Xylene

**Target Variable:**
- `AQI`: Air Quality Index (13–2049 range)
- `AQI_Bucket`: Categorical labels (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

**Temporal:**
- Daily measurements per city
- 88,488 missing values addressed via per-city median imputation

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of all pollutants and AQI
- Temporal patterns: seasonal, weekly, and daily trends
- Spatial patterns: city-level pollution baselines
- Correlation analysis with AQI (PM10: r=0.803, CO: r=0.683)
- Data quality audits (interpolation detection, censorship at AQI=500)

### 2. Feature Engineering

**Created Features (59 total):**
- **Pollutants:** 12 original pollutant measurements
- **Missing Flags:** 12 binary indicators for sensor offline status
- **Calendar:** month, day_of_week, is_weekend, season (one-hot)
- **Lag Features:** aqi_lag1, aqi_lag7
- **Rolling Statistics:** 7-day mean and standard deviation
- **Spatial:** City dummies (25 categories, drop-first encoding)

**Target Transformation:**
log1p(AQI) for training → expm1() to original scale for evaluation
(Reduces skewness from 4.4 to <1)

### 3. Train/Test Split

**Strategy:** Temporal split (no random splitting to prevent leakage)
**Cutoff:** December 1, 2019
**Distribution:**
- Train: 19,750 rows (79.5%) | 2015-01-01 to 2019-11-30
- Test: 5,100 rows (20.5%) | 2019-12-01 to 2020-07-01

### 4. Models

#### Baseline Models

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Persistence (lag-1) | 44.34 | 21.52 | 0.759 |
| Ridge (α=1.0) | 42.02 | 26.20 | 0.784 |

#### Advanced Models

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| KNN (k=15, Euclidean, PCA 15d) | 43.81 | 26.86 | 0.765 |
| Random Forest | 22.47 | 11.26 | 0.938 |
| **XGBoost (tuned)** | **21.94** | **11.73** | **0.941** |
| FeedForward NN (128→64, dropout=0.2) | 30.53 | 16.60 | 0.886 |
| LSTM (w=7, h=128, L=1) | 37.23 | 21.06 | 0.831 |

#### Per-Category MAE Breakdown

| Category | Persistence | Ridge | KNN | Random Forest | XGBoost | FNN | LSTM |
|----------|------------|-------|-----|---------------|---------|-----|------|
| Good | 7.01 | 30.14 | 24.61 | 6.01 | 5.02 | 13.00 | 13.94 |
| Satisfactory | 10.82 | 19.40 | 19.93 | 6.96 | 7.67 | 9.83 | 11.43 |
| Moderate | 24.58 | 17.62 | 24.68 | 12.69 | 13.48 | 16.66 | 23.00 |
| Poor | 50.29 | 51.63 | 49.21 | 22.38 | 21.74 | 31.35 | 44.70 |
| Very Poor | 41.80 | 54.75 | 37.78 | 15.16 | 17.78 | 30.64 | 40.68 |
| Severe | 162.36 | 193.14 | 158.23 | 76.19 | 71.29 | 128.84 | 143.44 |

---

## Key Findings

1. **PM10 is the strongest AQI predictor** (r=0.803), followed by CO (0.683) and PM2.5 (0.659)
2. **Winter months show 2.1× higher AQI** than monsoon season
3. **City-specific behavior:** O3 has negative correlation with AQI in some cities (Gurugram: r=-0.127) due to photochemical titration
4. **Multicollinearity:** PM10 and PM2.5 show moderate VIF (4.5, 4.3) but remain below the severe threshold
5. **XGBoost achieves the best overall performance** (R²=0.941, RMSE=21.94), narrowly edging Random Forest (R²=0.938)
6. **Tree ensembles strongly outperform neural networks** on this tabular dataset; LSTM and FNN underperform relative to their complexity
7. **Severe AQI remains the hardest category** to predict across all models due to rarity and high variability

---

## Repository Structure

```
MSE546/
│
├── city_day.csv                     # Raw dataset
├── preprocessing.py                 # Shared feature engineering pipeline
│
├── AQI_EDA.ipynb                    # Exploratory data analysis
├── baseline.ipynb                   # Persistence & Ridge baseline models
├── knn.ipynb                        # K-Nearest Neighbors model
├── random_forest.ipynb              # Random Forest model
├── xgboost_model.ipynb              # XGBoost model (baseline + tuned)
├── fnn_tabular.ipynb                # Feedforward Neural Network model
├── lstm.ipynb                       # LSTM model
├── comparison.ipynb                 # Cross-model comparison and visualizations
│
├── results/                         # Saved metrics (JSON) for each model
│   ├── persistence_metrics.json
│   ├── ridge_metrics.json
│   ├── knn_metrics.json
│   ├── random_forest_metrics.json
│   ├── xgboost_metrics.json
│   ├── fnn_tabular_metrics.json
│   └── lstm_metrics.json
│
├── plots/                           # Generated visualizations
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**Python Version:** 3.8+

### Running the Analysis

1. **Exploratory Data Analysis:**
   ```bash
   jupyter notebook AQI_EDA.ipynb
   ```

2. **Baseline Models (Persistence, Ridge):**
   ```bash
   jupyter notebook baseline.ipynb
   ```

3. **Advanced Models:**
   ```bash
   jupyter notebook knn.ipynb
   jupyter notebook random_forest.ipynb
   jupyter notebook xgboost_model.ipynb
   jupyter notebook fnn_tabular.ipynb
   jupyter notebook lstm.ipynb
   ```

4. **Cross-Model Comparison:**
   ```bash
   jupyter notebook comparison.ipynb
   ```

Results are saved automatically to `results/` and plots to `plots/`.

---

## Evaluation Metrics

**Primary Metrics (on original AQI scale):**
- **RMSE:** Penalizes large errors (critical for severe pollution days)
- **MAE:** Interpretable average absolute error in AQI units
- **R²:** Proportion of variance explained

**Secondary Analysis:**
- Per-category MAE breakdown (Good → Severe)
- Residual distribution analysis
- Temporal prediction consistency

---

## Contributors

- **Nicholas Chia**
- **Callum Gillies**
- **Matthew Ho**
- **Abhishek Srikaran**
- **Agishan Thaya**

---

## License

Educational project for MSE 546 (Winter 2026)

---

## Acknowledgments

Dataset sourced from Indian air quality monitoring stations (2015-2020).
Project completed as part of the Advanced Machine Learning course at the University of Waterloo.

Large language models (Claude, ChatGPT) were used throughout this project for debugging assistance, code formatting, and editorial review of written content.

- Anthropic. (2026). *Claude Code* [AI coding assistant]. https://claude.ai/code
- Anysphere. (2026). *Cursor* [AI-powered code editor]. https://www.cursor.com
