# bcr_map.py
import re, zipfile, tempfile, shutil
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

try:
    import contextily as cx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

def _guess_value_col(gdf: gpd.GeoDataFrame) -> str:
    cols = [c for c in gdf.columns if c.lower() != "geometry"]
    for c in cols:
        if re.search(r"건폐율", str(c), flags=re.IGNORECASE):
            return c
    keys = ["bcr","building_coverage","coverage","cover","ratio","rate","value","val"]
    for c in cols:
        if any(k in str(c).lower() for k in keys):
            return c
    num = [c for c in cols if pd.api.types.is_numeric_dtype(gdf[c])]
    if not num:
        raise ValueError("value column not found")
    return num[0]

def _normalize_bcr_column(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    key = _guess_value_col(g)
    s = pd.to_numeric(g[key], errors="coerce")
    if s.dropna().quantile(0.95) > 1:
        s = s / 100.0
    g = g.copy()
    g["S"] = s.clip(0, 1)
    return g[["S","geometry"]]

def load_bcr_100m_from_nested_zip(outer_zip_path: str) -> gpd.GeoDataFrame:
    p = Path(outer_zip_path)
    if not p.exists():
        raise FileNotFoundError(p)
    tmp = Path(tempfile.mkdtemp(prefix="unz_bcr_"))
    parts = []
    try:
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp)
        for iz in tmp.rglob("*.zip"):
            tgt = iz.with_suffix("")
            tgt.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(iz, "r") as zf:
                    zf.extractall(tgt)
            except Exception:
                pass
        for shp in tmp.rglob("*.shp"):
            parts.append(gpd.read_file(shp))
        for gpkg in tmp.rglob("*.gpkg"):
            parts.append(gpd.read_file(gpkg))
        if not parts:
            raise RuntimeError("no shp/gpkg found")
        fixed, ref = [], None
        for g in parts:
            if g.crs is None:
                g = g.set_crs(4326)
            ref = ref or g.crs
            if g.crs != ref:
                g = g.to_crs(ref)
            fixed.append(_normalize_bcr_column(g))
        return gpd.GeoDataFrame(pd.concat(fixed, ignore_index=True), crs=ref).to_crs(4326)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def bcr_overview(gdf: gpd.GeoDataFrame, value_col: str = "S") -> None:
    s = pd.to_numeric(gdf[value_col], errors="coerce")
    print("=== BCR Overview (S: fraction 0–1) ===")
    print("rows:", len(gdf))
    print("crs:", gdf.crs)
    print("geometry non-null:", gdf.geometry.notna().sum())
    print("nulls:", s.isna().sum(), " zeros:", (s==0).sum())
    print("describe:\n", s.describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    q = s.quantile([.9,.95,.99]).to_dict()
    print("quantiles(90/95/99):", {int(k*100): float(v) for k,v in q.items()})
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    bc = pd.cut(s, bins=bins, include_lowest=True)
    print("\nbin counts (S in [0,1]):")
    print(bc.value_counts().sort_index())
    print("\nrecommended clip_quantile ~", 0.99, "→ vmax≈", float(s.quantile(0.99)))

def plot_bcr_map(
    gdf: gpd.GeoDataFrame,
    value_col: str = "S",
    title: str | None = "Building Coverage Ratio (S, 100m)",
    cmap: str = "YlOrRd",
    alpha: float = 0.9,
    figsize: tuple[int, int] = (10, 10),
    use_basemap: bool = False,
    add_labels: bool = False,
    save_path: str | None = None,
    clip_quantile: float = 0.99,
    log_scale: bool = False,
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

def compute_and_plot_bcr(
    bcr_outer_zip_path: str,
    print_overview: bool = True,
    **plot_kwargs,
):
    gdf = load_bcr_100m_from_nested_zip(bcr_outer_zip_path)
    if print_overview:
        bcr_overview(gdf, value_col="S")
    plot_bcr_map(gdf, **plot_kwargs)
    return gdf

bcr_gdf = compute_and_plot_bcr(
    "../PopulationData/서울시 건폐율 데이터 100m.zip",
    print_overview=True,
    title="서울시 건폐율(100m)",
    clip_quantile=0.99,
    log_scale=False,
    scale_factor=1.0
)
