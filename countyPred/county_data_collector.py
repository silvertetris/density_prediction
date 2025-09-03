import json
import pandas as pd
from countyPred.data_prex import data_prex
from countyPred.xgBoost import forecast_all

def county_data_collector():
    with open("county.json", "r", encoding="utf-8") as f:
        county_names = json.load(f)
    population_data = pd.DataFrame(index=['data'])

    for name in county_names:
        data, _, _ = data_prex(county=name)
        future = forecast_all(data, n_lags=6, n_forecast=12)
        if isinstance(future.columns, pd.MultiIndex):
            future = future.copy()
            future.columns = future.columns.get_level_values(0)

        val = float(future.at[future.index[0], 'population'])
        population_data.loc['data', str(name)] = val
    return population_data

print(county_data_collector())