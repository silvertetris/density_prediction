# n_calc.py
import re
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

def ensure_rho(pop_gdf: gpd.GeoDataFrame, value_col: str | None = None) -> gpd.GeoDataFrame:
    g = pop_gdf.copy()
    if "rho" in g.columns:
        return g
    if value_col is None:
        value_col = _guess_value_col(g)
    gm = g.to_crs(3857)
    gm["area_m2"] = gm.geometry.area
    gm["rho"] = pd.to_numeric(gm[value_col], errors="coerce").fillna(0) / gm["area_m2"]
    out = gm.to_crs(4326)[["rho", "geometry"]]
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
    pop3857 = pop.to_crs(3857)[["rho", "geometry"]]
    bcr3857 = bcr.to_crs(3857)[["S", "geometry"]]
    joined = gpd.sjoin(bcr3857, pop3857, how="left", predicate="within")
    joined["rho"] = joined["rho"].fillna(0.0)
    joined["S"] = joined["S"].fillna(0.0)
    joined["N_k"] = P_fall * A_exp * joined["rho"] * (1 - joined["S"]) * P_fatality
    total = float(joined["N_k"].sum())
    return total, joined.to_crs(4326)
