import pandas as pd

from countyPred.data_prex import data_prex
from countyPred.xgBoost import forecast_all


def calculate_with_last_index(actual: pd.DataFrame,
                              forecast: pd.DataFrame) -> pd.Series:
    preds = []
    prev = actual['population'].iloc[-1] #한칸 땡기기 (이전)

    for idx in forecast.index:
        immigrant = forecast.at[idx, 'immigrants']
        born = forecast.at[idx, 'born']
        death = forecast.at[idx, 'death']
        calc_result = prev + immigrant + born - death
        preds.append(calc_result)
        prev = calc_result #애드혹

    return pd.Series(preds, index=forecast.index, name='expected_population')

actual, _, _ = data_prex()
forcast = forecast_all(actual, n_lags=6, n_forecast=12)
print(calculate_with_last_index(actual, forcast))