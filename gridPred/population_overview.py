# rho_map.py
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

try:
    import contextily as cx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

def _infer_year_from_path(path: str, fallback: int | None = None) -> int:
    m = re.search(r"(20\d{2})", path)
    if m: return int(m.group(1))
    if fallback is None: raise ValueError("year not found")
    return fallback

def load_population_rho(
    txt_path: str,
    shp_path: str,
    grid_id_col: str = "GRID_100M_",
    metric: str = "to_in_001",
    year: int | None = None
) -> gpd.GeoDataFrame:
    df = pd.read_csv(
        txt_path, sep=r"\^", engine="python", header=None,
        names=["year", "cell_id", "metric", "value"],
        dtype={"year": int, "cell_id": str, "metric": str, "value": float},
        encoding="utf-8"
    )
    y = year if year is not None else _infer_year_from_path(txt_path, df["year"].max())
    dsel = df[(df["year"] == y) & (df["metric"] == metric)].copy()
    dsel["cell_id"] = dsel["cell_id"].astype(str).str.strip()

    ggrid = gpd.read_file(shp_path)
    if ggrid.crs is None:
        ggrid = ggrid.set_crs(4326)
    if grid_id_col not in ggrid.columns:
        raise KeyError(f"{grid_id_col} not in {list(ggrid.columns)}")
    ggrid[grid_id_col] = ggrid[grid_id_col].astype(str).str.strip()

    g = ggrid.merge(dsel[["cell_id","value"]], left_on=grid_id_col, right_on="cell_id", how="left")
    gm = g.to_crs(3857)
    gm["area_m2"] = gm.geometry.area
    gm["rho"] = gm["value"].fillna(0.0) / gm["area_m2"]
    return gm.to_crs(4326)[["rho","value","area_m2","geometry"]]

def plot_rho_map(
    gdf: gpd.GeoDataFrame,
    value_col: str = "rho",
    title: str | None = "Population Density ρ (persons/m², 100m)",
    cmap: str = "viridis",
    alpha: float = 0.9,
    figsize: tuple[int, int] = (10, 10),
    use_basemap: bool = False,
    add_labels: bool = False,
    save_path: str | None = None,
    clip_quantile: float = 0.99,
    log_scale: bool = True,
    scale_factor: float = 1.0,
):
    g = gdf.copy()
    if g.crs is None:
        g.set_crs(4326, inplace=True)
    gm = g.to_crs(3857)
    vals = pd.to_numeric(gm[value_col], errors="coerce").fillna(0) * float(scale_factor)
    if log_scale:
        vals = np.log1p(vals)
    gm["_plot_val"] = vals
    vmax = float(vals.quantile(min(max(clip_quantile, 0.5), 0.999)))
    vmin = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x); ax.set_ylim(miny - pad_y, maxy + pad_y)
    if use_basemap and HAS_CTX:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=True)
            if add_labels:
                cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels, attribution=False)
        except Exception as e:
            print(f"[WARN] basemap: {e}")
    gm.plot(column="_plot_val", ax=ax, cmap=cmap, legend=True,
            vmin=vmin, vmax=vmax, edgecolor="white", linewidth=0.08, alpha=alpha,
            missing_kwds={"color":"#d9d9d9","hatch":"///","label":"No data","alpha":0.7})
    ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=220, bbox_inches="tight"); print(f"[INFO] saved → {save_path}")
    plt.show()

def compute_and_plot_rho(
    pop_txt_path: str,
    pop_shp_path: str,
    grid_id_col: str = "GRID_100M_",
    metric: str = "to_in_001",
    year: int | None = None,
    **plot_kwargs,
):
    gdf = load_population_rho(
        txt_path=pop_txt_path,
        shp_path=pop_shp_path,
        grid_id_col=grid_id_col,
        metric=metric,
        year=year,
    )
    plot_rho_map(gdf, **plot_kwargs)
    return gdf

rho_gdf = compute_and_plot_rho(pop_txt_path="../PopulationData/sgis/100res_grid_data/2023년_인구_다사_100M.txt",pop_shp_path="../PopulationData/sgis/grid_다사_100M.shp",year=2023,title="서울시 인구밀도 ρ (100m)",clip_quantile=0.995, log_scale=True, scale_factor=1.0, cmap="viridis"
)