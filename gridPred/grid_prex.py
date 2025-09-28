# visualize_nested_zip_grids.py
import os, zipfile, tempfile, shutil
from pathlib import Path
import re
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# (선택) 베이스맵
try:
    import contextily as cx
    HAS_CTX = True
except Exception:
    HAS_CTX = False

# 한글 폰트(원하는 대로 수정)
plt.rcParams["axes.unicode_minus"] = False
for fam in ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        pass


# ---------- 공통 유틸 ----------
def safe_read_geodata(path: Path) -> gpd.GeoDataFrame:
    """SHP/GPKG 경로를 안전하게 읽기"""
    if path.suffix.lower() == ".gpkg":
        return gpd.read_file(str(path))
    if path.suffix.lower() == ".shp":
        return gpd.read_file(str(path))
    raise ValueError(f"지원하지 않는 포맷: {path}")

def guess_value_col(gdf: gpd.GeoDataFrame) -> str:
    """건폐율/지표 컬럼 자동 추정: '건폐율' -> 키워드 -> 첫 숫자형"""
    cols = [c for c in gdf.columns if c.lower() != "geometry"]

    for c in cols:
        if re.search(r"건폐율", str(c), flags=re.IGNORECASE):
            return c
    keys = ["bcr","building_coverage","coverage","cover","ratio","rate","value","val"]
    for c in cols:
        cl = str(c).lower()
        if any(k in cl for k in keys):
            return c
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(gdf[c])]
    if num_cols:
        return num_cols[0]
    raise ValueError("값 컬럼을 찾지 못했습니다. gdf.columns를 확인하세요.")

def normalize_percent_series(s: pd.Series) -> pd.Series:
    """0~1 비율로 보이면 %로 변환"""
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if not valid.empty and 0 <= valid.quantile(0.95) <= 1:
        s = s * 100.0
    return s


# ---------- 핵심: 중첩 ZIP 통째로 읽기 ----------
def load_all_layers_from_nested_zip(outer_zip_path: str) -> list[gpd.GeoDataFrame]:
    """
    바깥 ZIP을 임시폴더에 풀고, 내부의 모든 ZIP/SHP/GPKG를 찾아 전부 로드.
    서로 다른 구(레이어)들을 리스트로 반환.
    """
    p = Path(outer_zip_path)
    if not p.exists():
        raise FileNotFoundError(p)

    tmp_root = Path(tempfile.mkdtemp(prefix="unz_outer_"))
    gdfs: list[gpd.GeoDataFrame] = []
    try:
        # 1) 바깥 ZIP 전체 추출
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp_root)

        # 2) 내부에 또 zip이 있으면 모두 풀기
        inner_zips = list(tmp_root.rglob("*.zip"))
        for iz in inner_zips:
            target_dir = iz.with_suffix("")  # zip과 같은 이름 폴더
            target_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(iz, "r") as zf:
                    zf.extractall(target_dir)
            except Exception as e:
                print(f"[WARN] 내부 ZIP 해제 실패: {iz} → {e}")

        # 3) 최종적으로 모든 SHP/GPKG 수집
        shp_list = list(tmp_root.rglob("*.shp"))
        gpkg_list = list(tmp_root.rglob("*.gpkg"))
        candidates = shp_list + gpkg_list
        if not candidates:
            raise RuntimeError("SHP/GPKG를 찾지 못했습니다. 압축 구조를 확인하세요.")

        # 4) 전부 읽기
        for i, path in enumerate(candidates, start=1):
            try:
                gdf = safe_read_geodata(path)
                gdf["__source__"] = str(path)
                gdfs.append(gdf)
                print(f"[INFO] loaded ({i}/{len(candidates)}): {path.name}, rows={len(gdf)}")
            except Exception as e:
                print(f"[WARN] 로딩 실패: {path} → {e}")

        if not gdfs:
            raise RuntimeError("어떤 레이어도 읽지 못했습니다.")
        return gdfs

    finally:
        # 임시폴더 유지하고 싶으면 주석 처리
        shutil.rmtree(tmp_root, ignore_errors=True)


def concat_and_unify(gdfs: list[gpd.GeoDataFrame], target_epsg: int = 4326) -> gpd.GeoDataFrame:
    """여러 GeoDataFrame을 CRS 통일 후 concat"""
    fixed = []
    ref_crs = None
    for gdf in gdfs:
        g = gdf.copy()
        # CRS 정리: 없으면 임의로 WGS84로 지정(필요 시 여기 조정)
        if g.crs is None:
            g.set_crs(target_epsg, inplace=True)
        if ref_crs is None:
            ref_crs = g.crs
        if g.crs != ref_crs:
            g = g.to_crs(ref_crs)
        fixed.append(g)
    # 서로 컬럼이 달라도 concat 가능(없으면 NaN)
    return pd.concat(fixed, ignore_index=True)


# ---------- 시각화 ----------
def plot_choropleth(
        gdf: gpd.GeoDataFrame,
        value_col: str | None = None,
        title: str | None = None,
        cmap: str = "YlOrRd",
        use_satellite: bool = True,   # True: Esri 위성 + 라벨, False: Carto 밝은 지도
        add_labels: bool = True,
        alpha_polys: float = 0.9,
        save_path: str | None = None,
):
    if value_col is None:
        value_col = guess_value_col(gdf)

    g = gdf.copy()
    g[value_col] = normalize_percent_series(g[value_col])

    # CRS → Web Mercator
    if g.crs is None:
        g.set_crs(4326, inplace=True)
    gm = g.to_crs(3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # extent
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # basemap
    if HAS_CTX:
        try:
            if use_satellite:
                cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=True)
                if add_labels:
                    try:
                        cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, attribution=False)
                    except Exception:
                        cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels, attribution=False)
            else:
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=True)
        except Exception as e:
            print(f"[WARN] 베이스맵 불가 → {e}")

    # layer
    gm.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        edgecolor="white",
        linewidth=0.15,
        alpha=alpha_polys,
        missing_kwds={"color": "#d9d9d9", "hatch": "///", "label": "No data", "alpha": 0.7},
    )

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(title or value_col, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] saved → {os.path.abspath(save_path)}")
    plt.show()


# ---------- 실행 예시 ----------
if __name__ == "__main__":
    # (예) 바깥 ZIP: 그 안에 구별 ZIP/레이어들이 들어있는 구조
    outer_zip = r"C:\Desktop\density_prediction\PopulationData\서울시 건폐율 데이터 100m.zip"

    gdfs = load_all_layers_from_nested_zip(outer_zip)
    print(f"[INFO] loaded layers: {len(gdfs)}")

    g_all = concat_and_unify(gdfs, target_epsg=4326)
    print("[INFO] concatenated rows:", len(g_all))
    print("columns:", list(g_all.columns))
    print("crs:", g_all.crs)

    # value_col=None → 자동 추정(맘에 안 들면 직접 지정)
    plot_choropleth(
        g_all,
        value_col=None,
        title="서울시 건폐율(100m) — 모든 구 레이어 병합",
        cmap="YlOrRd",
        use_satellite=True,
        add_labels=True,
        alpha_polys=0.85,
        save_path=None,   # "bcr_map.png"
    )
