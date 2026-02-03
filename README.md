# India Air Quality Index (AQI) Forecasting

Machine learning project for predicting Air Quality Index across 26 Indian cities using temporal and pollutant features.

**Course:** MSE 546 — Advanced Machine Learning
**Team:** Group 2
**Members:** Nicholas Chia, Callum Gillies, Matthew Ho, Abhishek Srikaran, Agishan Thaya
**Date:** January 2026

---

## Project Overview

This project develops predictive models for the Air Quality Index (AQI) across India's National Capital Region (NCR) and 25 other major cities. Using daily air quality measurements from 2015-2020, we implement baseline regression models and explore temporal patterns, spatial variations, and pollutant correlations.

**Key Objectives:**
- Predict daily AQI values for 26 Indian cities
- Identify critical pollutants and temporal patterns affecting air quality
- Build interpretable baseline models before advancing to complex architectures
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
- `AQI`: Air Quality Index (13-2049 range)
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

### 4. Baseline Models

| Model | RMSE | MAE | R² | Description |
|-------|------|-----|----|----|
| **Persistence (lag-1)** | 44.34 | 21.52 | 0.759 | Predict yesterday's AQI |
| **Ridge Regression** | 43.77 | 26.85 | 0.765 | Linear model on log1p(AQI), α=1.0 |

**Per-Category Performance (Ridge MAE):**
- Good: 30.38
- Satisfactory: 19.88
- Moderate: 18.08
- Poor: 51.72
- Very Poor: 58.29
- Severe: 201.50

---

## Key Findings

1. **PM10 is the strongest AQI predictor** (r=0.803), followed by CO (0.683) and PM2.5 (0.659)
2. **Winter months show 2.1× higher AQI** than monsoon season
3. **City-specific behavior:** O3 has negative correlation with AQI in some cities (Gurugram: r=-0.127) due to photochemical titration
4. **Multicollinearity:** PM10 and PM2.5 show moderate VIF (4.5, 4.3) but remain below severe threshold
5. **Persistence baseline performs well** (MAE=21.52), suggesting strong autocorrelation

---

## Repository Structure

```
546_Project/
│
├── city_day.csv                     # Raw data
├── AQI_EDA.ipynb                    # Exploratory data analysis
├── baseline.ipynb                   # Baseline model implementation
├── plots/                           # Generated visualizations
│   ├── 05_station_map.png
│   ├── 06_aqi_distribution.png
│   ├── 07_pollutants.png
│   ├── 09_correlations.png
│   ├── 10a_daily_timeseries.png
│   └── ...
├── MSE546__Project_Specifics__W26.pdf  # Project specifications
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

**Python Version:** 3.8+

### Running the Analysis

1. **Exploratory Data Analysis:**
   ```bash
   jupyter notebook AQI_EDA.ipynb
   ```

2. **Baseline Model Training:**
   ```bash
   jupyter notebook baseline.ipynb
   ```

3. **Visualizations:**
   Generated plots are saved automatically to `plots/` directory

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
