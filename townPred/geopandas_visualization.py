import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional, Sequence

from townPred.arima_prediction import arima_forecast_all
from townPred.data_prex import data_prex, town_wide_data


# --- 1) 예측 테이블에서 특정 시점의 동별 값 뽑기 (열=10자리 동코드) ---
def build_dong_pred_df(
    preds_wide: pd.DataFrame,     # index=예측 월(DatetimeIndex), columns=동코드(10자리)
    at_date: Optional[pd.Timestamp] = None,
    horizon: Optional[int] = 0,
    value_col: str = "predicted_value",
) -> pd.DataFrame:
    if at_date is not None:
        if at_date not in preds_wide.index:
            raise KeyError(f"{at_date} 가 preds_wide.index에 없습니다. 예: {preds_wide.index[:5]}")
        row = preds_wide.loc[at_date]
        used_label = at_date
    else:
        if horizon is None:
            raise ValueError("at_date 또는 horizon 중 하나는 지정해야 합니다.")
        if horizon < 0 or horizon >= len(preds_wide.index):
            raise IndexError(f"horizon={horizon} 범위를 벗어났습니다 (0~{len(preds_wide.index)-1})")
        row = preds_wide.iloc[horizon]
        used_label = preds_wide.index[horizon]

    df = row.rename_axis("adm_cd2").reset_index(name=value_col)
    # 코드 문자열/제로패딩 통일(10자리)
    df["adm_cd2"] = df["adm_cd2"].astype(str).str.strip().str.zfill(10)
    return df, used_label


# --- 2) GeoJSON(adm_cd2)와 조인하여 코로플레스 ---
def plot_dong_choropleth(
    shape_path: str,
    dong_pred_df: pd.DataFrame,            # columns: ['adm_cd2', value_col]
    value_col: str = "predicted_value",
    code_cols_in_shape: Sequence[str] = ("adm_cd2", "EMD_CD", "adm_cd", "adm_cd10", "dong_cd"),
    cmap: str = "OrRd",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (8, 8),
):
    # Geo 레이어 로드
    g = gpd.read_file(shape_path)
    # GeoJSON 안의 동코드 컬럼 자동 선택 (adm_cd2 우선)
    chosen = next((c for c in code_cols_in_shape if c in g.columns), None)
    if chosen is None:
        raise KeyError(f"동 코드 컬럼을 찾지 못했습니다. 후보={code_cols_in_shape}, 실제={list(g.columns)}")

    g = g.copy()
    g[chosen] = g[chosen].astype(str).str.strip().str.zfill(10)

    t = dong_pred_df.copy()
    if "adm_cd2" not in t.columns:
        raise KeyError("dong_pred_df에 'adm_cd2' 컬럼이 필요합니다.")
    if value_col not in t.columns:
        raise KeyError(f"dong_pred_df에 '{value_col}' 컬럼이 필요합니다.")
    t["adm_cd2"] = t["adm_cd2"].astype(str).str.strip().str.zfill(10)

    merged = g.merge(t[["adm_cd2", value_col]], left_on=chosen, right_on="adm_cd2", how="left")

    # 매칭 안 된 동 코드 경고
    miss = merged[merged[value_col].isna()][chosen].unique().tolist()
    if miss:
        print(f"[WARN] 값 누락 동코드 {len(miss)}개 예시: {miss[:10]}")

    # 보기 좋은 투영으로 변환 후 플롯
    merged = merged.to_crs(3857)
    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=value_col, ax=ax, cmap=cmap, legend=True,
        edgecolor="white", linewidth=0.3,
        missing_kwds={"color": "lightgray", "hatch": "///", "label": "No data"},
    )
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # 자동 줌
    minx, miny, maxx, maxy = merged.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    plt.tight_layout()
    plt.show()
    return merged

df_long = data_prex()
wide = town_wide_data(df_long)
future_arima = arima_forecast_all(wide_df=wide, n_years=3, order=(1,1,1), seasonal_order=(0,1,1,12))

# 2) 특정 예측 시점 선택 (첫 예측월: horizon=0)
dong_pred_df, used_label = build_dong_pred_df(future_arima, horizon=0, value_col="predicted_value")

# 3) GeoJSON(adm_cd2)와 병합해서 플롯
merged = plot_dong_choropleth(
    shape_path="../PopulationData/seoul_map.geojson",  # 너의 동 경계 파일 경로
    dong_pred_df=dong_pred_df,
    value_col="predicted_value",
    title=f"Dong-level ARIMA prediction • {pd.Timestamp(used_label).date()}",
)
