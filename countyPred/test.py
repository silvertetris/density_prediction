import json
from typing import Optional, Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from countyPred.county_data_collector import county_data_collector


def plot_seoul_by_sgg_code(
    shape_path: str,
    wide_df: pd.DataFrame,                       # 행=1, 열=영문 구이름
    mapping_json_path: str,                      # {"Gangnam-gu":"11680", ...}
    at_row: int | str = 0,                       # 사용할 행(정수 index 또는 라벨; 예: "data")
    cmap: str = "OrRd",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
):
    #일단 1번째 행
    if isinstance(at_row, int):
        row = wide_df.iloc[at_row]
        used_label = wide_df.index[at_row] if len(wide_df.index) > at_row else at_row
    else:
        row = wide_df.loc[at_row]
        used_label = at_row
    series = row.dropna()

    with open(mapping_json_path, "r", encoding="utf-8") as f:
        name_to_sgg: dict[str, str] = json.load(f)

    tbl = pd.DataFrame({
        "ENG": series.index.astype(str),
        "value": series.values
    })
    tbl["SIG_CD"] = (
        tbl["ENG"].map(name_to_sgg)
                  .astype(str)
                  .str.strip()
                  .str.zfill(5)
    )

    gdf = gpd.read_file(shape_path)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    code_col = 'sgg'
    gdf[code_col] = gdf[code_col].astype(str).str.strip().str.zfill(5)

    merged = gdf.merge(tbl[["SIG_CD", "value"]], left_on=code_col, right_on="SIG_CD", how="left")

    merged = merged.to_crs(3857) #단순히 웹페이지에서 지도를 보여주기 위함이라고 한다면, ESPG:3857을 사용하는 것
    #plot
    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column="value", ax=ax, cmap=cmap,
        edgecolor="white", linewidth=0.5, legend=True,
        missing_kwds={"color": "lightgray", "hatch": "///", "label": "No data"},
    )
    if title is None:
        title = f"Seoul population (row={used_label})"
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    ax.set_aspect("equal")

    # 자동 줌
    minx, miny, maxx, maxy = merged.total_bounds
    pad_x, pad_y = (maxx-minx)*0.05, (maxy-miny)*0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    plt.tight_layout(); plt.show()

    return merged

df = county_data_collector()

plot_seoul_by_sgg_code(
    shape_path="../PopulationData/seoul_map.geojson",
    wide_df=df,
    mapping_json_path="../PopulationData/seoul_sgg_code.json",
    at_row="data",
    cmap="OrRd",
    title="seoul population prediction (XGBOOST)"
)
