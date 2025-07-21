import pandas as pd
from xgboost import XGBRegressor
import numpy as np

from countyPred import data_prex


def create_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df = df.dropna()
    return df

def xgb_forecast(series: pd.Series, n_lags: int, n_forecast: int, **xgb_params) -> pd.Series:

    df = create_lag_features(series, n_lags)
    X, y = df.drop(columns='y'), df['y']
    model = XGBRegressor(**xgb_params)
    model.fit(X, y)

    last_vals = series.values[-n_lags:].tolist()
    preds = []
    for _ in range(n_forecast):
        x_input = np.array(last_vals[-n_lags:]).reshape(1, -1)
        y_pred = model.predict(x_input)[0]
        preds.append(y_pred)
        last_vals.append(y_pred)

    last_date = series.index[-1]
    freq = pd.infer_freq(series.index) or 'MS'
    future_idx = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                               periods=n_forecast, freq=freq)
    return pd.Series(preds, index=future_idx)

def forecast_all(df: pd.DataFrame,
                 n_lags: int = 3,
                 n_forecast: int = 12,
                 xgb_params: dict = None) -> pd.DataFrame:
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        }

    forecasts = {}
    for col in df.columns:
        print(f"[{col}] 모델 학습 및 예측 중...")
        forecasts[col] = xgb_forecast(df[col], n_lags, n_forecast, **xgb_params)

    # 결과 합치기
    forecast_df = pd.concat(forecasts, axis=1)
    return forecast_df


result, min_start, max_end = data_prex.data_prex()
future_preds = forecast_all(result, n_lags=6, n_forecast=12)
print(future_preds)
