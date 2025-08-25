import geopandas as gpd

def abstract_geojson(path: str, sample_n: int = 5):
    gdf = gpd.read_file(path)
    print("✅ Loaded:", path)
    print("- shape:", gdf.shape)
    print("- columns:", list(gdf.columns))
    print("- dtypes:\n", gdf.dtypes)
    print("- crs:", gdf.crs)
    print("- geometry type:", gdf.geometry.geom_type.unique().tolist())
    print("- total_bounds [minx, miny, maxx, maxy]:", gdf.total_bounds)

    print(f"\n- sample({sample_n}):")
    display_cols = [c for c in gdf.columns if c != "geometry"][:10]
    print(gdf[display_cols].head(sample_n))

    return gdf

gdf = abstract_geojson("../PopulationData/seoul_map.geojson")

import matplotlib.pyplot as plt

def quick_plot(gdf, title="Quick view"):
    ax = gdf.to_crs(4326).plot(figsize=(6,6), edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.axis("off")
    plt.show()

quick_plot(gdf, "quick")

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from countyPred.data_prex import data_prex  # 네가 만든 함수

def build_seoul_population_df(
    shape_path: str,
    county_col_in_shape: str = "sgg",   # GeoJSON에서 '구 이름' 컬럼명 (영문 가정)
    sido_col_in_shape: str = "sido",       # 시/도 컬럼명
    seoul_values=("Seoul", "서울특별시"),      # 서울 식별 값(영/한 둘 다 허용)
    data_path: str = "../PopulationData",
    value_at: pd.Timestamp | None = None      # 특정 시점 값을 쓰고 싶다면 지정(없으면 최신값)
) -> pd.DataFrame:
    gdf = gpd.read_file(shape_path)
    # 좌표계 정리(없거나 지역좌표면 4326으로 맞춤)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    # 서울만 필터
    seoul_gdf = gdf[gdf[sido_col_in_shape].isin(seoul_values)].copy()

    # 서울 구 이름 목록
    gu_names = (
        seoul_gdf[county_col_in_shape]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []
    for gu in gu_names:
        try:
            # 각 구의 시계열
            result, _, _ = data_prex(path=data_path, county=gu)  # result: (월별) population, immigrants, born, death
            pop_series = result["population"].dropna()
            if pop_series.empty:
                print(f"[WARN] {gu}: population 시계열이 비어 있어 스킵합니다.")
                continue

            # 사용할 값 선택: 지정 시점 or 최신값
            if value_at is not None and value_at in pop_series.index:
                pop_value = float(pop_series.loc[value_at])
            else:
                pop_value = float(pop_series.iloc[-1])  # 최신 값

            rows.append({county_col_in_shape: gu, "population": pop_value})
        except Exception as e:
            print(f"[WARN] {gu}: 데이터 수집 실패 → {e}")

    return pd.DataFrame(rows)


def plot_seoul_choropleth(
    shape_path: str,
    pop_df: pd.DataFrame,
    merge_on: str = "C1_NM_ENG",
    sido_col_in_shape: str = "SIDO_NM",
    seoul_values=("Seoul", "서울특별시"),
    cmap: str = "Blues",
    title: str = "Seoul Population"
):
    """
    서울만 코로플레스(색칠)로 그리기.
    """
    gdf = gpd.read_file(shape_path)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    seoul_gdf = gdf[gdf[sido_col_in_shape].isin(seoul_values)].copy()

    merged = seoul_gdf.merge(pop_df, left_on=merge_on, right_on=merge_on, how="left")

    fig, ax = plt.subplots(figsize=(7, 7))
    merged.plot(
        column="population",
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.5,
        legend=True,
        missing_kwds={"color": "lightgray", "hatch": "///", "label": "No data"},
    )
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    # 자동 줌(서울 경계)
    minx, miny, maxx, maxy = merged.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    plt.tight_layout()
    plt.show()



shape_path = "../PopulationData/seoul_map.geojson"
seoul_pop_df = build_seoul_population_df(shape_path=shape_path)
plot_seoul_choropleth(shape_path=shape_path, pop_df=seoul_pop_df, title="Seoul Population (latest)")
