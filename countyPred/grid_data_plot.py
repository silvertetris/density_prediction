# grid_choropleth_with_satellite.py
import os, re, glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 한글(Windows 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 1) TXT 로딩 (폴더/파일 모두 지원)
# -----------------------------
def _read_one_txt(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=r"\^", engine="python", header=None,
        names=["year", "cell_id", "metric", "value"],
        dtype={"year": int, "cell_id": str, "metric": str, "value": float},
        encoding=encoding,
    )
    # 깨진 제로폭문자/공백 정리
    df["cell_id"] = (df["cell_id"].astype(str)
                     .str.replace("\u200b", "", regex=False)
                     .str.strip())
    df["__source_path__"] = path
    return df

def load_txts(path_or_dir: str, encoding: str = "utf-8") -> pd.DataFrame:
    """폴더면 *.txt 전부 읽어 concat, 파일이면 그 파일만"""
    if os.path.isdir(path_or_dir):
        paths = sorted(glob.glob(os.path.join(path_or_dir, "*.txt")))
        if not paths:
            raise FileNotFoundError(f"폴더에 *.txt 없음: {path_or_dir}")
    else:
        if not os.path.exists(path_or_dir):
            raise FileNotFoundError(path_or_dir)
        paths = [path_or_dir]

    frames = []
    for p in paths:
        try:
            frames.append(_read_one_txt(p, encoding=encoding))
        except Exception as e:
            print(f"[WARN] TXT 로드 실패: {p} → {e}")
    if not frames:
        raise RuntimeError("읽을 수 있는 TXT가 없습니다.")
    return pd.concat(frames, ignore_index=True)

def overview(df: pd.DataFrame) -> None:
    print("== years =="); print(df["year"].value_counts().sort_index())
    print("\n== metrics =="); print(df["metric"].value_counts())
    print("\n== value describe =="); print(df["value"].describe())


# -----------------------------
# 2) 격자 ID 정규화(접두 + 숫자부 zfill)
# -----------------------------
def _split_prefix_digits(s: pd.Series) -> pd.DataFrame:
    s2 = s.astype(str).str.replace("\u200b", "", regex=False).str.strip()
    m = s2.str.extract(r"^(?P<prefix>\D*?)(?P<digits>\d+)$")
    m["raw"] = s2
    return m

def normalize_grid_keys(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """SHP키(left)와 TXT키(right)의 숫자부 자릿수를 맞춰 조인 키를 통일"""
    L = _split_prefix_digits(left)
    R = _split_prefix_digits(right)
    l_len = L["digits"].fillna("").str.len()
    r_len = R["digits"].fillna("").str.len()
    width = int(max(l_len.max(), r_len.max()))
    if width == 0:
        return (left.astype(str).str.strip(), right.astype(str).str.strip())
    l_key = (L["prefix"].fillna("") + L["digits"].fillna("").str.zfill(width))
    r_key = (R["prefix"].fillna("") + R["digits"].fillna("").str.zfill(width))
    return (l_key, r_key)


# -----------------------------
# 3) SHP 조인 (연/지표 자동 선택 가능)
# -----------------------------
def to_geopandas_auto(
        df_all: pd.DataFrame,
        shp_path: str,
        grid_id_col: str,
        year: int | None = None,           # None → 최신연도 자동
        metric: str | None = None,         # None → 해당 연도 최빈 metric 자동
        missing_policy: str = "gray",      # 'gray' | 'zero' | 'drop'
) -> tuple[gpd.GeoDataFrame, int, str]:
    years = sorted(df_all["year"].dropna().unique())
    if not years:
        raise ValueError("데이터에 year가 없습니다.")
    if year is None:
        year = max(years)
    sub = df_all[df_all["year"] == year]
    if sub.empty:
        raise ValueError(f"해당 연도 데이터가 없습니다: {year}")
    if metric is None:
        metric = sub["metric"].value_counts().idxmax()

    dsel = sub[sub["metric"] == metric].copy()
    if dsel.empty:
        raise ValueError(f"[ERR] year={year}, metric='{metric}' 데이터가 없습니다.")

    ggrid = gpd.read_file(shp_path)
    if grid_id_col not in ggrid.columns:
        raise KeyError(f"Shapefile에 '{grid_id_col}' 없음. gdf.columns={list(ggrid.columns)}")

    left_key, right_key = normalize_grid_keys(ggrid[grid_id_col], dsel["cell_id"])
    ggrid = ggrid.copy(); ggrid["_key"] = left_key
    dsel["_key"] = right_key

    # 커버리지 로그
    A = set(ggrid["_key"].astype(str)); B = set(dsel["_key"].astype(str))
    print(f"[DEBUG] SHP 격자 수: {len(A)}  / TXT 격자 수: {len(B)}  / 교집합: {len(A & B)}")

    merged = ggrid.merge(dsel[["_key", "value"]], on="_key", how="left")

    print(f"[INFO] year={year}, metric='{metric}'")
    print(f"[INFO] merged rows: {len(merged)}")
    print(f"[INFO] value non-null: {merged['value'].notna().sum()} (of {len(merged)})")

    if missing_policy == "zero":
        merged["value"] = merged["value"].fillna(0.0)
        print("[INFO] missing_policy='zero' → NaN→0")
    elif missing_policy == "drop":
        before = len(merged)
        merged = merged.dropna(subset=["value"]).reset_index(drop=True)
        print(f"[INFO] missing_policy='drop' → {before - len(merged)}개 격자 제거")
    else:
        missing = merged.loc[merged["value"].isna(), "_key"].unique().tolist()
        if missing:
            print(f"[WARN] 값 없는 격자: {len(missing)} (예: {missing[:10]})")

    merged = merged.set_crs(4326) if merged.crs is None else merged
    return merged, year, metric


# -----------------------------
# 4) 플롯: 위성사진/라벨 베이스맵 (contextily)
# -----------------------------
def plot_grid_with_basemap(
        gdf: gpd.GeoDataFrame,
        value_col: str = "value",
        title: str | None = None,
        cmap: str = "OrRd",
        figsize=(8, 8),
        basemap: str = "esri_satellite",   # 'esri_satellite' | 'carto_light' | 'none'
        add_labels: bool = True,           # 라벨 타일 오버레이
        alpha_polys: float = 0.85,
):
    """
    basemap:
      - 'esri_satellite' : Esri 위성사진
      - 'carto_light'    : CartoDB 밝은 바탕
      - 'none'           : 배경 없이 데이터만
    """
    gm = gdf.to_crs(3857)
    fig, ax = plt.subplots(figsize=figsize)

    # 영역 먼저 잡아두기(베이스맵 타일 범위 계산용)
    minx, miny, maxx, maxy = gm.total_bounds
    pad_x, pad_y = (maxx-minx)*0.05, (maxy-miny)*0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # 베이스맵
    if basemap != "none":
        try:
            import contextily as cx
            if basemap == "esri_satellite":
                cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, attribution=True)
            elif basemap == "carto_light":
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution=True)
            else:
                pass

            # 라벨 오버레이
            if add_labels:
                try:
                    cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, attribution=False)
                except Exception:
                    # Stamen 이슈 시 Carto 라벨 대체
                    cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels, attribution=False)
        except Exception as e:
            print(f"[WARN] basemap 불가 → {e}")

    # 데이터 레이어
    gm.plot(
        column=value_col, ax=ax, cmap=cmap, legend=True,
        edgecolor="white", linewidth=0.2, alpha=alpha_polys,
        missing_kwds={"color":"#cccccc","hatch":"///","label":"No data","alpha":0.6},
    )

    if title: ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5) 예시 실행
# -----------------------------
if __name__ == "__main__":
    # TXT가 여러 연도로 있는 폴더
    txt_dir_or_file = "../PopulationData/sgis"          # 예: 2015~2023 txt가 있는 폴더
    # 격자 SHP (동일 폴더의 .dbf/.shx/.prj 필요)
    shp_path = "../PopulationData/sgis/grid_다사_1K.shp"
    grid_id_col = "GRID_1K_CD"

    # 1) TXT 전체 로드 + 개요
    df_all = load_txts(txt_dir_or_file, encoding="utf-8")
    overview(df_all)

    # 2) 최신 연/최빈 지표 자동 선택 (원하면 year, metric 직접 지정)
    gmerged, year, metric = to_geopandas_auto(
        df_all,
        shp_path=shp_path,
        grid_id_col=grid_id_col,
        year=None,             # 최신 연도 자동
        metric=None,           # 해당 연도 최빈 metric 자동
        missing_policy="gray", # 'gray' | 'zero' | 'drop'
    )

    # 3) 위성사진 + 라벨 배경으로 시각화
    plot_grid_with_basemap(
        gmerged,
        value_col="value",
        title=f"{year} · {metric}",
        cmap="OrRd",
        basemap="esri_satellite",  # ← 위성사진(구글맵 느낌)
        add_labels=True,           # 도로/지명 라벨 오버레이
    )
