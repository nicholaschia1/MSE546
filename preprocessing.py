"""
preprocessing.py — shared data loading, feature engineering, and evaluation
for the mse 546 aqi forecasting project (group 2).

usage:
    from preprocessing import (
        load_and_prepare, impute, build_features,
        get_train_test, get_arrays, evaluate, save_results,
        POLLUTANT_COLS, AQI_BINS, AQI_LABELS, CUTOFF_DATE,
    )
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLLUTANT_COLS = [
    'pm2.5', 'pm10', 'no', 'no2', 'nox', 'nh3',
    'co', 'so2', 'o3', 'benzene', 'toluene', 'xylene',
]

AQI_BINS   = [0, 50, 100, 200, 300, 400, 5000]
AQI_LABELS = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

CUTOFF_DATE = pd.Timestamp('2019-12-01')

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(path='city_day.csv'):
    """load csv, lowercase columns, parse dates, sort by city then date."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def impute(df):
    """
    drop rows with missing aqi, add per-pollutant missingness flags,
    then fill missing pollutant values with per-city medians.
    """
    df = df.dropna(subset=['aqi']).reset_index(drop=True)
    for col in POLLUTANT_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(int)
    for col in POLLUTANT_COLS:
        city_medians = df.groupby('city')[col].transform('median')
        df[col] = df[col].fillna(city_medians)
    return df


def _season(month):
    if month in [12, 1, 2]:   return 'winter'
    if month in [3, 4, 5]:    return 'summer'
    if month in [6, 7, 8, 9]: return 'monsoon'
    return 'post_monsoon'


def _fourier(series, period):
    """return sin and cos encoding of a numeric series for the given period."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def build_features(df):
    """
    add all engineered features and return (df, feature_cols).

    feature groups
    --------------
    pollutants     : 12 raw pollutant concentrations (already imputed)
    missing flags  : 12 binary indicators for which pollutants were imputed
    calendar       : month, day_of_week, is_weekend, season dummies,
                     fourier sin/cos for day-of-year and day-of-week
    aqi lags       : lag1, lag7, lag14, lag30
    aqi rolling    : 7-day and 30-day rolling mean and std
    pollutant lags : pm2.5 and pm10 lag1 (strongest predictors from eda)
    city dummies   : one-hot city encoding (drop_first=True, 25 columns)

    returns (df_with_features, feature_cols list)
    """
    df = df.copy()

    # log-transform the target; invert with expm1 at eval time
    df['aqi_log'] = np.log1p(df['aqi'])

    # --- calendar ---
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # season dummies
    df['season']   = df['month'].apply(_season)
    season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)

    # fourier encoding: continuous representation of seasonal/weekly cycles
    # avoids the discontinuity at month/week boundaries
    df['sin_doy'], df['cos_doy'] = _fourier(df['day_of_year'], 365.25)
    df['sin_dow'], df['cos_dow'] = _fourier(df['day_of_week'], 7)

    # --- lag and rolling features (per city, shift(1) prevents leakage) ---
    df = df.sort_values(['city', 'date']).reset_index(drop=True)

    for lag in [1, 7, 14, 30]:
        col = f'aqi_lag{lag}'
        df[col] = df.groupby('city')['aqi'].shift(lag)

    for window in [7, 30]:
        df[f'aqi_roll{window}_mean'] = df.groupby('city')['aqi'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(3, window // 3)).mean()
        )
        df[f'aqi_roll{window}_std'] = df.groupby('city')['aqi'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=max(3, window // 3)).std()
        )

    aqi_lag_roll_cols = (
        ['aqi_lag1', 'aqi_lag7', 'aqi_lag14', 'aqi_lag30'] +
        ['aqi_roll7_mean', 'aqi_roll7_std', 'aqi_roll30_mean', 'aqi_roll30_std']
    )
    for col in aqi_lag_roll_cols:
        df[col] = df[col].fillna(df.groupby('city')[col].transform('median'))

    # key pollutant lags — pm2.5 and pm10 are the two strongest aqi drivers
    # using sanitized column names (no dot) for the lag columns
    for raw_col, lag_name in [('pm2.5', 'pm25_lag1'), ('pm10', 'pm10_lag1')]:
        df[lag_name] = df.groupby('city')[raw_col].shift(1)
        df[lag_name] = df[lag_name].fillna(df.groupby('city')[lag_name].transform('median'))

    pollutant_lag_cols = ['pm25_lag1', 'pm10_lag1']

    # --- city one-hot encoding ---
    city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True)
    df = pd.concat([df, city_dummies], axis=1)

    # --- build ordered feature list ---
    missing_flags  = [c for c in df.columns if c.endswith('_missing')]
    calendar_feats = (
        ['month', 'day_of_week', 'day_of_year', 'is_weekend',
         'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow'] +
        [c for c in df.columns if c.startswith('season_')]
    )
    city_feats   = [c for c in df.columns if c.startswith('city_')]
    feature_cols = (
        POLLUTANT_COLS +
        missing_flags +
        calendar_feats +
        aqi_lag_roll_cols +
        pollutant_lag_cols +
        city_feats
    )

    return df, feature_cols

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def get_train_test(df, cutoff=CUTOFF_DATE):
    """temporal split: train < cutoff, test >= cutoff."""
    train_df = df[df['date'] < cutoff].copy().reset_index(drop=True)
    test_df  = df[df['date'] >= cutoff].copy().reset_index(drop=True)
    print(f'train: {len(train_df):,} rows | test: {len(test_df):,} rows | cutoff: {cutoff.date()}')
    return train_df, test_df


def get_arrays(train_df, test_df, feature_cols):
    """
    extract numpy arrays from train/test dataframes.
    fills any residual nans: dummy/flag columns → 0, continuous → train median.

    returns x_train, x_test, y_train_log, y_test_raw, y_test_log
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    dummy_cols = [
        c for c in feature_cols
        if c.startswith(('city_', 'season_')) or c.endswith('_missing') or c == 'is_weekend'
    ]
    cont_cols = [c for c in feature_cols if c not in dummy_cols]

    for col in dummy_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col]  = test_df[col].fillna(0)

    train_medians = train_df[cont_cols].median()
    for col in cont_cols:
        train_df[col] = train_df[col].fillna(train_medians[col])
        test_df[col]  = test_df[col].fillna(train_medians[col])

    X_train     = train_df[feature_cols].values.astype(np.float64)
    y_train_log = train_df['aqi_log'].values.astype(np.float64)
    X_test      = test_df[feature_cols].values.astype(np.float64)
    y_test_raw  = test_df['aqi'].values.astype(np.float64)
    y_test_log  = test_df['aqi_log'].values.astype(np.float64)

    assert not np.isnan(X_train).any(), 'nans remain in X_train'
    assert not np.isnan(X_test).any(),  'nans remain in X_test'

    return X_train, X_test, y_train_log, y_test_raw, y_test_log

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label='Model', verbose=True):
    """
    compute rmse, mae, r² and per-category mae on the original aqi scale.
    prints results if verbose=True. returns a flat dict for json serialisation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    cats    = pd.cut(pd.Series(y_true), bins=AQI_BINS, labels=AQI_LABELS)
    cat_mae = {}
    cat_n   = {}
    for cat in AQI_LABELS:
        mask = (cats == cat).values
        if mask.sum() > 0:
            cat_mae[f'MAE_{cat}'] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            cat_n[cat]            = int(mask.sum())

    result = {'label': label, 'RMSE': rmse, 'MAE': mae, 'R2': r2, **cat_mae}

    if verbose:
        sep = '─' * 54
        print(f'\n{sep}')
        print(f'  {label}')
        print(sep)
        print(f'  RMSE : {rmse:8.2f}')
        print(f'  MAE  : {mae:8.2f}')
        print(f'  R²   : {r2:8.4f}')
        print(f'  MAE by AQI category:')
        for cat in AQI_LABELS:
            key = f'MAE_{cat}'
            if key in cat_mae:
                print(f'    {cat:>14s}: {cat_mae[key]:7.2f}  (n={cat_n[cat]:,})')

    return result


def save_results(result, model_name):
    """save metrics dict to results/<model_name>_metrics.json."""
    os.makedirs('results', exist_ok=True)
    path = os.path.join('results', f'{model_name}_metrics.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved → {path}')
