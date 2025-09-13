import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np

from countyPred import data_prex


def create_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({'y': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df

def xgb_forecast(series: pd.Series, n_lags: int, n_forecast: int,
                 use_cv: bool = True, cv_nfold: int = 5, cv_early_stopping: int = 50, cv_shuffle: bool = False,
                 **xgb_params) -> pd.Series:

    df = create_lag_features(series, n_lags)
    x, y = df.drop(columns='y'), df['y']

    def _to_cv_params(_p: dict) -> tuple[dict, int]:
        _p = (_p or {}).copy()
        num_round = int(_p.pop('n_estimators', 100))
        params = {
            'objective': _p.pop('objective', 'reg:squarederror'),
            'eta': _p.pop('learning_rate', 0.1),           # learning_rate -> eta
            'max_depth': _p.pop('max_depth', 6),
            'subsample': _p.pop('subsample', 1.0),
            'colsample_bytree': _p.pop('colsample_bytree', 1.0),
            'eval_metric': _p.pop('eval_metric', 'rmse'),
            'seed': _p.pop('random_state', 42),
            **_p
        }
        return params, num_round

    best_rounds = xgb_params.get('n_estimators', 100)

    if use_cv:
        try:
            dtrain = xgb.DMatrix(x.values, label=y.values)
            params, num_round = _to_cv_params(xgb_params)
            cv_res = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_round,
                nfold=cv_nfold,
                early_stopping_rounds=cv_early_stopping,
                shuffle=cv_shuffle,
                verbose_eval=False
            )
            best_rounds = int(cv_res.shape[0])
        except Exception as e:
            print(f"[WARN] xgboost.cv 실패 → 기본 n_estimators 사용 ({best_rounds}). 이유: {e}")

    xgb_params_for_fit = {**xgb_params, 'n_estimators': best_rounds}
    model = XGBRegressor(**xgb_params_for_fit)
    model.fit(x, y)

    last_vals = series.values[-n_lags:].tolist()  # ad hoc, sliding window
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
                 n_lags: int = 3,  # 그 전 데이터
                 n_forecast: int = 12,
                 xgb_params: dict = None,
                 use_cv: bool = True, cv_nfold: int = 5, cv_early_stopping: int = 50, cv_shuffle: bool = False
                 ) -> pd.DataFrame:
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror'
        }

    forecasts = {}
    for col in df.columns:
        forecasts[col] = xgb_forecast(
            df[col], n_lags, n_forecast,
            use_cv=use_cv, cv_nfold=cv_nfold, cv_early_stopping=cv_early_stopping, cv_shuffle=cv_shuffle,
            **xgb_params
        )

    forecast_df = pd.concat(forecasts, axis=1)
    return forecast_df


# 사용 예
result, min_start, max_end = data_prex.data_prex()
future_preds = forecast_all(
    result, n_lags=6, n_forecast=12,
    use_cv=True, cv_nfold=5, cv_early_stopping=50, cv_shuffle=False,
    xgb_params={'n_estimators': 300, 'learning_rate': 0.05, 'objective': 'reg:squarederror'}
)
print(future_preds)
