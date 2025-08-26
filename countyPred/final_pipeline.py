from countyPred.cohort_calculate import calculate_with_last_index
from countyPred.data_err_calc import data_err_calc
from countyPred.data_prex import data_prex
from countyPred.geopandas_visualization import build_seoul_population_df, plot_seoul_choropleth
from countyPred.xgBoost import forecast_all


def final_pipeline():
    path = "../PopulationData/seoul_map.geojson"
    result, min_start, max_end = data_prex()
    #data_err_calc(result) #원본 err calc
    future_preds = forecast_all(result, n_lags=6, n_forecast=12) #xgBoost
    expected = calculate_with_last_index(result, future_preds) #cohort calc
    seoul_df = build_seoul_population_df(shape_path=path)
    plot_seoul_choropleth(
        shape_path=path,
        pop_df=seoul_df,
        merge_on="sgg",  # GeoJSON & pop_df 공통 키
        value_col="population",
        filter_col=None,
        filter_values=None,
        cmap="OrRd",
        title="서울: 1년 후 예측 인구"
    )

final_pipeline()