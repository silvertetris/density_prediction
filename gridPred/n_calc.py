# n_calc.py
import re, zipfile, tempfile, shutil
from pathlib import Path
import pandas as pd
import geopandas as gpd

def _guess_value_col(gdf: gpd.GeoDataFrame) -> str:
    cols = [c for c in gdf.columns if c.lower() != "geometry"]
    for c in cols:
        if re.search(r"건폐율", str(c), flags=re.IGNORECASE):
            return c
    keys = ["bcr","building_coverage","coverage","cover","ratio","rate","value","val","pop","인구"]
    for c in cols:
        if any(k in str(c).lower() for k in keys):
            return c
    num = [c for c in cols if pd.api.types.is_numeric_dtype(gdf[c])]
    if not num:
        raise ValueError("numeric value column not found")
    return num[0]

def _infer_year_from_path(path: str, fallback: int | None = None) -> int:
    m = re.search(r"(20\d{2})", path)
    if m:
        return int(m.group(1))
    if fallback is None:
        raise ValueError("year not found")
    return fallback

def load_population_rho(txt_path: str,
                        shp_path: str,
                        grid_id_col: str = "GRID_1K_CD",
                        metric: str = "to_in_001",
                        year: int | None = None) -> gpd.GeoDataFrame:
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
    tmp_root = Path(tempfile.mkdtemp(prefix="unz_bcr_"))
    parts = []
    try:
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_root)
        for iz in tmp_root.rglob("*.zip"):
            tgt = iz.with_suffix("")
            tgt.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(iz, "r") as zf:
                    zf.extractall(tgt)
            except Exception:
                pass
        for shp in tmp_root.rglob("*.shp"):
            parts.append(gpd.read_file(shp))
        for gpkg in tmp_root.rglob("*.gpkg"):
            parts.append(gpd.read_file(gpkg))
        if not parts:
            raise RuntimeError("no shp/gpkg found")
        fixed = []
        ref = None
        for g in parts:
            if g.crs is None:
                g = g.set_crs(4326)
            if ref is None:
                ref = g.crs
            if g.crs != ref:
                g = g.to_crs(ref)
            fixed.append(_normalize_bcr_column(g))
        g_all = gpd.GeoDataFrame(pd.concat(fixed, ignore_index=True), crs=ref)
        return g_all.to_crs(4326)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

def ensure_rho(pop_gdf: gpd.GeoDataFrame, value_col: str | None = None) -> gpd.GeoDataFrame:
    g = pop_gdf.copy()
    if "rho" in g.columns:
        return g
    if value_col is None:
        value_col = _guess_value_col(g)
    gm = g.to_crs(3857)
    gm["area_m2"] = gm.geometry.area
    gm["rho"] = pd.to_numeric(gm[value_col], errors="coerce").fillna(0) / gm["area_m2"]
    out = gm.to_crs(4326)[["rho","geometry"]]
    for c in g.columns:
        if c not in out.columns:
            out[c] = g[c]
    return out

def ensure_S(bcr_gdf: gpd.GeoDataFrame, value_col: str | None = None) -> gpd.GeoDataFrame:
    g = bcr_gdf.copy()
    if "S" in g.columns:
        return g
    if value_col is None:
        value_col = _guess_value_col(g)
    s = pd.to_numeric(g[value_col], errors="coerce")
    if s.dropna().quantile(0.95) > 1:
        s = s / 100.0
    g["S"] = s.clip(0, 1)
    return g

def compute_casualty(pop_1km: gpd.GeoDataFrame,
                     bcr_100m: gpd.GeoDataFrame,
                     A_exp: float = 79.36704,
                     P_fall: float = 1.0,
                     P_fatality: float = 1.0) -> tuple[float, gpd.GeoDataFrame]:
    pop = ensure_rho(pop_1km)
    bcr = ensure_S(bcr_100m)
    pop3857 = pop.to_crs(3857)[["rho","geometry"]]
    bcr3857 = bcr.to_crs(3857)[["S","geometry"]]
    joined = gpd.sjoin(bcr3857, pop3857, how="left", predicate="within")
    joined["rho"] = joined["rho"].fillna(0.0)
    joined["S"] = joined["S"].fillna(0.0)
    joined["N_k"] = P_fall * A_exp * joined["rho"] * (1 - joined["S"]) * P_fatality
    total = float(joined["N_k"].sum())
    return total, joined.to_crs(4326)


import matplotlib.pyplot as plt
import geopandas as gpd

try:
    import contextily as cx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

def plot_casualty_map(
        per_cell: gpd.GeoDataFrame,
        value_col: str = "N_k",
        title: str | None = None,
        cmap: str = "magma",
        alpha: float = 0.85,
        figsize: tuple[int, int] = (10, 10),
        use_basemap: bool = True,
        add_labels: bool = True,
        save_path: str | None = None,
):
    if value_col not in per_cell.columns:
        raise KeyError(f"'{value_col}' not found in columns: {list(per_cell.columns)}")
    g = per_cell.copy()
    if g.crs is None:
        g.set_crs(4326, inplace=True)
    gm = g.to_crs(3857)
    fig, ax = plt.subplots(figsize=figsize)
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    if use_basemap and HAS_CTX:
        try:
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=True)
            if add_labels:
                try:
                    cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, attribution=False)
                except Exception:
                    cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels, attribution=False)
        except Exception as e:
            print(f"[WARN] basemap: {e}")
    gm.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        edgecolor="white",
        linewidth=0.15,
        alpha=alpha,
        missing_kwds={"color": "#d9d9d9", "hatch": "///", "label": "No data", "alpha": 0.7},
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or value_col, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] saved → {save_path}")
    plt.show()

def compute_and_plot_from_sources(
        pop_txt_path: str,
        pop_shp_path: str,
        bcr_outer_zip_path: str,
        grid_id_col: str = "GRID_1K_CD",
        metric: str = "to_in_001",
        year: int | None = None,
        A_exp: float = 79.36704,
        P_fall: float = 1.0,
        P_fatality: float = 1.0,
        title: str | None = None,
        cmap: str = "magma",
        save_path: str | None = None,
        use_basemap: bool = True,
        add_labels: bool = True,
):
    pop_gdf = load_population_rho(
        txt_path=pop_txt_path,
        shp_path=pop_shp_path,
        grid_id_col=grid_id_col,
        metric=metric,
        year=year,
    )
    bcr_gdf = load_bcr_100m_from_nested_zip(bcr_outer_zip_path)
    total, per_cell = compute_casualty(
        pop_1km=pop_gdf,
        bcr_100m=bcr_gdf,
        A_exp=A_exp,
        P_fall=P_fall,
        P_fatality=P_fatality,
    )
    if title is None:
        try:
            y = year if year is not None else _infer_year_from_path(pop_txt_path, None)
        except Exception:
            y = None
        title = f"Casualty per 100m cell{f' ({y})' if y else ''}"
    plot_casualty_map(
        per_cell,
        value_col="N_k",
        title=title,
        cmap=cmap,
        save_path=save_path,
        use_basemap=use_basemap,
        add_labels=add_labels,
    )
    return total, per_cell

total, per_cell = compute_and_plot_from_sources(
    pop_txt_path="../PopulationData/sgis/2023년_인구_다사_1K.txt",
    pop_shp_path="../PopulationData/sgis/grid_다사_1K.shp",
    bcr_outer_zip_path="../PopulationData/서울시 건폐율 데이터 100m.zip",
    metric="to_in_001",
    year=2023,
    cmap="magma",
    save_path=None,       # 예: "casualty_map.png" 로 저장하려면 경로 지정
    use_basemap=False,
    add_labels=False,
)
print("총 피해 기대값 합계:", total)