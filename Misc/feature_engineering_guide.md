# Feature Engineering Ideas for AQI Forecasting
**MSE 546 Project - Delhi NCR Air Quality Prediction**

Date: January 26, 2026

---

## Table of Contents
1. [Lag Features](#lag-features)
2. [Rolling/Window Statistics](#rolling-window-statistics)
3. [Temporal Features](#temporal-features)
4. [Interaction Features](#interaction-features)
5. [Domain-Specific Features](#domain-specific-features)
6. [Trend and Change Features](#trend-and-change-features)
7. [External Data Integration](#external-data-integration)
8. [Dimensionality Reduction](#dimensionality-reduction)
9. [Implementation Priority](#implementation-priority)

---

## 1. Lag Features

### Basic Lags
Create lagged versions of target and key features to capture temporal dependencies.

**AQI Lags:**
- `aqi_lag1`, `aqi_lag2`, `aqi_lag3` (6, 12, 18 hours ago)
- `aqi_lag4` (24 hours ago - same time previous day)
- `aqi_lag8` (48 hours ago - two days back)
- **Rationale:** High autocorrelation (≈0.85-0.90) means recent values are predictive

**Pollutant Lags:**
- `pm25_lag1`, `pm25_lag2` (most correlated with AQI)
- `pm10_lag1`, `pm10_lag2`
- `no2_lag1`, `co_lag1`
- **Rationale:** Pollutants strongly correlate with AQI; their history matters

**Weather Lags:**
- `visibility_lag1`, `visibility_lag2` (strongest weather predictor)
- `temperature_lag1`, `wind_speed_lag1`
- **Rationale:** Weather conditions change gradually; lags capture trends

### Advanced Lags
**Same-time-of-day lags:**
- `aqi_lag_same_hour_yesterday` (24 hours back)
- `aqi_lag_same_hour_week_ago` (168 hours back)
- **Rationale:** Captures daily/weekly patterns (e.g., traffic patterns)

**Seasonal lags:**
- `aqi_lag_same_season_last_year` (if multi-year data available)
- **Rationale:** Seasonal effects are strong (winter AQI >> summer)

---

## 2. Rolling/Window Statistics

### Simple Rolling Features
Capture local trends and variability over recent time windows.

**Rolling means (smoothing):**
- `aqi_rolling_mean_3` (last 3 measurements ≈ 18 hours)
- `aqi_rolling_mean_6` (last 6 measurements ≈ 1.5 days)
- `aqi_rolling_mean_12` (last 12 measurements ≈ 3 days)
- `pm25_rolling_mean_3`, `visibility_rolling_mean_3`

**Rolling standard deviation (volatility):**
- `aqi_rolling_std_3`, `aqi_rolling_std_6`
- `pm25_rolling_std_3`
- **Rationale:** High variability might indicate harder-to-predict conditions

**Rolling min/max:**
- `aqi_rolling_min_6`, `aqi_rolling_max_6`
- **Rationale:** Captures range of recent values

### Advanced Rolling Features
**Exponentially weighted moving average (EWMA):**
- `aqi_ewma_alpha_0.3` (gives more weight to recent values)
- **Rationale:** Better than simple average for time series

**Rolling quantiles:**
- `aqi_rolling_median_6`, `aqi_rolling_q25_6`, `aqi_rolling_q75_6`
- **Rationale:** Robust to outliers

**Rolling slope (trend strength):**
```python
# Linear regression slope over last N points
aqi_rolling_slope_6 = df.groupby('station')['aqi'].transform(
    lambda x: x.rolling(6).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0])
)
```
- **Rationale:** Captures whether AQI is improving or deteriorating

---

## 3. Temporal Features

### Cyclical Encoding
Time features should be encoded cyclically (not linear) since hour 23 is close to hour 0.

**Hour encoding:**
```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Month encoding:**
```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

**Day of week encoding:**
```python
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

### Time-of-Day Categories
**Rush hour indicators:**
- `is_morning_rush` (hour == 6)
- `is_evening_rush` (hour == 18)
- **Rationale:** Traffic patterns affect pollution

**Day part:**
- `day_part`: 'early_morning' (6), 'afternoon' (12), 'evening' (18), 'night' (23)
- One-hot encode for models that benefit from it

### Calendar Features
**Special days:**
- `is_holiday` (if holiday calendar available)
- `is_festive_season` (Diwali period has high pollution)
- `days_since_diwali`, `days_until_diwali`

**Year effects:**
- `year` (as numeric or one-hot)
- **Rationale:** Long-term trends (improving/worsening air quality)

---

## 4. Interaction Features

### Pollutant Interactions
**Ratios:**
- `pm25_to_pm10_ratio = pm25 / pm10`
- `no2_to_co_ratio = no2 / co`
- **Rationale:** Relative composition matters for AQI

**Products (joint effects):**
- `pm25_x_humidity` (humidity affects particulate matter suspension)
- `temperature_x_wind_speed` (combined effect on dispersion)
- `pm25_x_visibility` (both related to atmospheric conditions)

### Temporal Interactions
**Hour × Feature:**
- `hour_x_aqi` (AQI behavior varies by time of day)
- `hour_x_pm25`, `hour_x_temperature`
- **Rationale:** Different features matter more at different times

**Season × Feature:**
- `is_winter_x_pm25` (PM2.5 especially bad in winter)
- `is_monsoon_x_humidity`

### Spatial Interactions
**City × Feature:**
- `is_delhi_x_pm25` (Delhi has worst pollution)
- One-hot encode city and interact with key pollutants

---

## 5. Domain-Specific Features

### Air Quality Index Specifics
**Pollutant contributions to AQI:**
```python
# AQI is typically determined by the worst pollutant
df['max_pollutant'] = df[['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']].idxmax(axis=1)
```
- One-hot encode `max_pollutant`

**Sub-indices for each pollutant:**
If you have the AQI breakpoint formulas, compute individual sub-indices:
- `pm25_subindex`, `pm10_subindex`, etc.
- **Rationale:** AQI is driven by worst pollutant; knowing which one helps

### Meteorological Domain Knowledge
**Atmospheric stability indicators:**
- `temp_inversion_indicator = temperature - temperature_lag1`
  - Positive = warming (stable atmosphere, pollution trapped)
- `wind_dispersion_strength = wind_speed * visibility`
  - Higher = better dispersion

**Moisture and precipitation:**
- If you get precipitation data: `rain_washout_effect`
- `humidity_category`: Low (<40%), Medium (40-70%), High (>70%)

**Ventilation coefficient:**
```python
# Rough proxy for atmospheric mixing
df['ventilation'] = df['wind_speed'] * df['temperature'] / df['humidity']
```

### Emission Sources Proxies
**Traffic indicators:**
- `is_weekday` (more traffic on weekdays)
- `hour == 6 or hour == 18` (rush hours)

**Industrial activity:**
- If you have data on industrial zones: distance to nearest zone
- Interaction: `is_industrial_station × is_weekday`

---

## 6. Trend and Change Features

### First-Order Changes (Velocity)
**Absolute change:**
- `aqi_change = aqi - aqi_lag1`
- `pm25_change = pm25 - pm25_lag1`
- **Rationale:** Rate of change matters (is it getting worse?)

**Percentage change:**
- `aqi_pct_change = (aqi - aqi_lag1) / aqi_lag1 * 100`
- `pm25_pct_change`

### Second-Order Changes (Acceleration)
**Change in change:**
```python
df['aqi_acceleration'] = df['aqi_change'] - df['aqi_change'].shift(1)
```
- **Rationale:** Is the rate of increase speeding up or slowing down?

### Cumulative Features
**Recent deterioration:**
- `cumulative_increase_3 = sum of positive changes over last 3 periods`
- **Rationale:** Multiple periods of increase might signal sustained bad air quality

**Days since last good AQI:**
```python
# Count time steps since AQI was < 100
df['periods_since_good_aqi'] = ...
```

---

## 7. External Data Integration

### Weather Forecasts
If available from APIs or datasets:
- `temp_forecast`, `wind_forecast`, `precip_forecast`
- **Rationale:** Forecasted weather is available in real-time for predictions

### Satellite Data
- **AOD (Aerosol Optical Depth):** Satellite measure of atmospheric particles
- **Fire locations:** Crop burning contributes to pollution

### Traffic Data
- Real-time or historical traffic density
- `traffic_density_nearby`

### Social/Economic Indicators
- `construction_activity` (if data available)
- `festival_indicators` (Diwali firecrackers, etc.)

---

## 8. Dimensionality Reduction

If you create too many features, consider:

### PCA on Pollutants
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pollutant_pcs = pca.fit_transform(df[['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']])
# Create 'pc1', 'pc2', 'pc3' features
```
- **Rationale:** Pollutants are correlated; PCA captures main patterns

### Feature Selection
- Use feature importance from tree models (Random Forest, XGBoost)
- Remove features with correlation > 0.95 (redundant)
- Use Lasso (L1 regularization) to automatically select features

---

## 9. Implementation Priority

### High Priority (Implement First) ✅
These have strong theoretical justification and are likely to help:

1. **Lag features (1-3 lags)** for AQI, PM2.5, PM10, visibility
2. **Rolling means (3, 6 windows)** for AQI, PM2.5
3. **Trend features:** `aqi_change`, `aqi_pct_change`
4. **Cyclical time encoding:** hour_sin/cos, month_sin/cos
5. **Interaction:** `pm25 × humidity`, `temperature × wind_speed`

### Medium Priority (Next Steps) 🔄
6. **Rolling std** for AQI (volatility measure)
7. **Same-time-yesterday lag** (24-hour lag)
8. **Hour × AQI interaction**
9. **Season × PM2.5 interaction**
10. **Visibility rolling mean**

### Lower Priority (If Time Permits) ⏳
11. Rolling slope/trend strength
12. EWMA features
13. Sub-indices for individual pollutants
14. Ventilation coefficient
15. Station-specific features

### Advanced/Experimental (For Neural Networks) 🚀
16. PCA on pollutants
17. Attention-based feature combinations (in neural nets)
18. Learned embeddings for station/city (in deep learning models)

---

## Code Templates

### Template 1: Create All Basic Lag Features
```python
def create_lag_features(df, n_lags=3):
    df = df.sort_values(['station', 'datetime'])
    
    # AQI lags
    for lag in range(1, n_lags + 1):
        df[f'aqi_lag{lag}'] = df.groupby('station')['aqi'].shift(lag)
    
    # Pollutant lags
    for pollutant in ['pm25', 'pm10', 'no2', 'co', 'visibility']:
        for lag in [1, 2]:
            df[f'{pollutant}_lag{lag}'] = df.groupby('station')[pollutant].shift(lag)
    
    return df
```

### Template 2: Rolling Statistics
```python
def create_rolling_features(df, windows=[3, 6]):
    df = df.sort_values(['station', 'datetime'])
    
    for window in windows:
        # Rolling mean
        df[f'aqi_rolling_mean_{window}'] = df.groupby('station')['aqi'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Rolling std
        df[f'aqi_rolling_std_{window}'] = df.groupby('station')['aqi'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        
        # Rolling min/max
        df[f'aqi_rolling_min_{window}'] = df.groupby('station')['aqi'].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f'aqi_rolling_max_{window}'] = df.groupby('station')['aqi'].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    
    return df
```

### Template 3: Cyclical Encoding
```python
def create_cyclical_features(df):
    # Hour (0-23, period = 24)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Month (1-12, period = 12)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # Day of week (0-6, period = 7)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df
```

### Template 4: Trend Features
```python
def create_trend_features(df):
    df = df.sort_values(['station', 'datetime'])
    
    # First-order change
    df['aqi_change'] = df['aqi'] - df.groupby('station')['aqi'].shift(1)
    df['aqi_pct_change'] = (df['aqi_change'] / df.groupby('station')['aqi'].shift(1)) * 100
    
    # Second-order change (acceleration)
    df['aqi_acceleration'] = df.groupby('station')['aqi_change'].diff()
    
    return df
```

### Template 5: Interaction Features
```python
def create_interaction_features(df):
    # Pollutant × weather
    df['pm25_x_humidity'] = df['pm25'] * df['humidity']
    df['pm25_x_visibility'] = df['pm25'] * df['visibility']
    df['temp_x_wind'] = df['temperature'] * df['wind_speed']
    
    # Temporal interactions
    df['hour_x_aqi'] = df['hour'] * df['aqi']
    df['hour_x_pm25'] = df['hour'] * df['pm25']
    
    # Pollutant ratios
    df['pm25_to_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # avoid division by zero
    df['no2_to_co_ratio'] = df['no2'] / (df['co'] + 1e-6)
    
    return df
```

### Template 6: Master Feature Engineering Function
```python
def engineer_features(df):
    """
    Apply all feature engineering steps.
    """
    print("Creating lag features...")
    df = create_lag_features(df, n_lags=3)
    
    print("Creating rolling features...")
    df = create_rolling_features(df, windows=[3, 6])
    
    print("Creating cyclical features...")
    df = create_cyclical_features(df)
    
    print("Creating trend features...")
    df = create_trend_features(df)
    
    print("Creating interaction features...")
    df = create_interaction_features(df)
    
    # Create target
    df['target'] = df.groupby('station')['aqi'].shift(-1)
    
    # Drop rows with NaN in critical features
    print("Cleaning NaN values...")
    df = df.dropna(subset=['target', 'aqi_lag1', 'aqi_lag2'])
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df
```

---

## Feature Selection Tips

After creating many features:

1. **Check for multicollinearity:**
```python
import seaborn as sns
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix)
# Remove features with correlation > 0.95
```

2. **Use tree model feature importance:**
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
# Keep top N features
```

3. **Try Lasso for automatic selection:**
```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)
# Features with non-zero coefficients are selected
```

---

## Expected Impact

Based on similar time series forecasting problems:

- **Lag features:** +5-10% improvement in R²
- **Rolling statistics:** +2-5% improvement
- **Cyclical encoding:** +1-3% improvement
- **Interactions:** +1-5% improvement (if domain-appropriate)
- **All combined:** Could improve R² from ~0.85 (persistence) to 0.92-0.95

Remember: Diminishing returns apply. The first few well-chosen features matter most!

---

## Resources for Further Reading

1. **Time Series Feature Engineering:**
   - [tsfresh library](https://tsfresh.readthedocs.io/) - automatic feature extraction
   
2. **Air Quality Specific:**
   - EPA AQI calculation guidelines
   - Research papers on air quality forecasting

3. **General Feature Engineering:**
   - "Feature Engineering for Machine Learning" by Alice Zheng
   - Kaggle feature engineering tutorials

---

**Good luck with your project! 🚀**
