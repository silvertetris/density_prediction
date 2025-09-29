import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def load_txt(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\^", engine="python", header=None,
        names=["year", "cell_id", "metric", "value"],
        dtype={"year": int, "cell_id": str, "metric": str, "value": float},
        encoding=encoding
    )

def overview(df: pd.DataFrame) -> None:
    print("head"); print(df.head())
    print("years"); print(df["year"].value_counts().sort_index())
    print("metrics"); print(df["metric"].value_counts())
    print("value"); print(df["value"].describe())

def to_geopandas(df: pd.DataFrame, grid_path: str,
                 grid_id_col: str, year: int, metric: str) -> gpd.GeoDataFrame:
    dsel = df[(df["year"] == year) & (df["metric"] == metric)].copy()
    dsel["cell_id"] = dsel["cell_id"].astype(str).str.strip()
    ggrid = gpd.read_file(grid_path)
    if ggrid.crs is None:
        ggrid = ggrid.set_crs(4326)
    if grid_id_col not in ggrid.columns:
        raise KeyError(f"'{grid_id_col}' not in {list(ggrid.columns)}")
    ggrid[grid_id_col] = ggrid[grid_id_col].astype(str).str.strip()
    merged = ggrid.merge(dsel[["cell_id", "value"]],
                         left_on=grid_id_col, right_on="cell_id", how="left")
    return merged

def add_density(gdf: gpd.GeoDataFrame, value_col: str = "value") -> gpd.GeoDataFrame:
    gm = gdf.to_crs(3857)
    gm["area_m2"] = gm.geometry.area
    gm["rho"] = gm[value_col].fillna(0) / gm["area_m2"]
    return gm.to_crs(4326)

def plot_choropleth(gdf: gpd.GeoDataFrame, value_col: str = "value",
                    cmap: str = "OrRd", title: str | None = None):
    gm = gdf.to_crs(3857)
    fig, ax = plt.subplots(figsize=(8, 8))
    gm.plot(column=value_col, cmap=cmap, ax=ax, legend=True,
            edgecolor="white", linewidth=0.2,
            missing_kwds={"color":"#e0e0e0","label":"No data"})
    ax.set_axis_off()
    ax.set_title(title or value_col)
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    plt.tight_layout()
    plt.show()

path = "../PopulationData/sgis/2023년_인구_다사_1K.txt"
df = load_txt(path, encoding="utf-8")
overview(df)
year_match = re.search(r"(20\d{2})", path)
year = int(year_match.group(1)) if year_match else df["year"].max()
metric = "to_in_001"
gmerged = to_geopandas(
    df,
    grid_path="../PopulationData/sgis/grid_다사_1K.shp",
    grid_id_col="GRID_1K_CD",
    year=year,
    metric=metric
)
gmerged = add_density(gmerged, value_col="value")
plot_choropleth(gmerged, value_col="value", title=f"{year} {metric}")
plot_choropleth(gmerged, value_col="rho", cmap="Blues", title=f"{year} density (people/m²)")
