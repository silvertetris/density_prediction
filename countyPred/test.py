import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def load_txt(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=r"\^", engine="python", header=None,  # split
        names=["year", "cell_id", "metric", "value"],
        dtype={"year": int, "cell_id": str, "metric": str, "value": float},
        encoding=encoding
    )
    return df

#개요
def overview(df: pd.DataFrame) -> None:
    print("head")
    print(df.head())
    print("years")
    print(df["year"].value_counts().sort_index())
    print("metrics")
    print(df["metric"].value_counts())
    print("value")
    print(df["value"].describe())


def to_geopandas(df: pd.DataFrame, grid_path: str,
                 grid_id_col: str, year: int, metric: str) -> gpd.GeoDataFrame:
    # (1) 연도/지표 선택
    dsel = df[(df["year"] == year) & (df["metric"] == metric)].copy()
    dsel["cell_id"] = dsel["cell_id"].astype(str).str.strip()

    # (2) Shapefile 읽기 (구성파일 .shx/.dbf/.prj 모두 있는 상태)
    ggrid = gpd.read_file(grid_path)  # engine 기본(pyogrio)로 충분
    if ggrid.crs is None:
        # prj가 있으면 보통 자동 설정됨. 없을 때만 임시로 WGS84
        ggrid = ggrid.set_crs(4326)

    # (3) 키 정규화 & 컬럼 체크
    if grid_id_col not in ggrid.columns:
        raise KeyError(f"'{grid_id_col}' 컬럼을 Shapefile에서 찾지 못했습니다. gdf.columns={list(ggrid.columns)}")
    ggrid[grid_id_col] = ggrid[grid_id_col].astype(str).str.strip()

    # (4) 코드 조인 (좌측: 모든 격자 유지)
    merged = ggrid.merge(dsel[["cell_id", "value"]],
                         left_on=grid_id_col, right_on="cell_id", how="left")
    return merged


# 4) 플롯
def plot_choropleth(gdf: gpd.GeoDataFrame, value_col: str = "value",
                    cmap: str = "OrRd", title: str | None = None):
    gm = gdf.to_crs(3857)
    ax = gm.plot
    ax.set_axis_off()
    ax.set_title(title or value_col)
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    plt.tight_layout();
    plt.show()


df = load_txt("../PopulationData/sgis/2015년_인구_다사_1K.txt", encoding="utf-8")
overview(df)
gmerged = to_geopandas(
    df,
    grid_path="../PopulationData/sgis/grid_다사_1K.shp",
    grid_id_col="GRID_1K_CD",
    year=2015,
    metric="to_in_001"
)
plot_choropleth(gmerged, value_col="value", title="2015 to_in_001 (grid)")
