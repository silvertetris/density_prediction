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

#quick_plot(gdf, "quick")

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from countyPred.data_prex import data_prex
import json
def build_seoul_population_df(
    shape_path: str,
    county_col_in_shape: str = "sgg",   # GeoJSON에서 '구 이름' 컬럼명 (영문 가정)
    sido_col_in_shape: str = "sido",       # 시/도 컬럼명
    seoul_values=("Seoul", "서울특별시"),      # 서울 식별 값(영/한 둘 다 허용)
    data_path: str = "../PopulationData",
    value_at: pd.Timestamp | None = None    # 특정 시점 값을 쓰고 싶다면 지정(없으면 최신값)
) -> pd.DataFrame:
    gdf = gpd.read_file(shape_path)
    # 좌표계 정리(없거나 지역좌표면 4326으로 맞춤)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    # 서울만 필터
    seoul_gdf = gdf[gdf[sido_col_in_shape].isin(seoul_values)].copy()

    with open("county.json", "r", encoding="utf-8") as f:
        gu_names = json.load(f)
    rows = []
    for gu in gu_names:
        print(gu)
        try:
            #중복
            result, _, _ = data_prex(path=data_path, county=gu)
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


import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Tuple

def plot_seoul_choropleth(
    shape_path: str,
    pop_df: pd.DataFrame,
    merge_on: str,                                # 우선 사용하려는 키(예: 'sgg' 또는 'SIG_KOR_NM')
    value_col: str = "population",
    filter_col: Optional[str] = None,             # 서울만 그릴 때 GeoJSON 시/도 컬럼 (없으면 None)
    filter_values: Optional[Iterable[str]] = None,# ex) ("서울특별시","Seoul")
    cmap: str = "OrRd",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 7),
):
    # --- 입력 점검 ---
    if merge_on not in pop_df.columns:
        raise KeyError(f"pop_df에 '{merge_on}' 컬럼이 없습니다. pop_df.columns={list(pop_df.columns)}")
    if value_col not in pop_df.columns:
        raise KeyError(f"pop_df에 '{value_col}' 컬럼이 없습니다. pop_df.columns={list(pop_df.columns)}")

    # filter_values가 문자열 단일값이면 리스트로 보정
    if filter_values is not None and isinstance(filter_values, (str, bytes)):
        filter_values = [filter_values]

    # --- GeoJSON 로드 & CRS 정리 ---
    gdf = gpd.read_file(shape_path)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    # --- (선택) 권역 필터 ---
    if filter_col and filter_values:
        if filter_col not in gdf.columns:
            raise KeyError(f"GeoJSON에 '{filter_col}' 컬럼이 없습니다. gdf.columns={list(gdf.columns)}")
        gdf[filter_col] = gdf[filter_col].astype(str).str.strip()
        filt_vals = {str(v).strip() for v in filter_values}
        gdf_f = gdf[gdf[filter_col].isin(filt_vals)].copy()
        if gdf_f.empty:
            all_vals = gdf[filter_col].dropna().astype(str).str.strip().unique().tolist()
            raise ValueError(
                f"필터 결과가 비었습니다. filter_col='{filter_col}', filter_values={filter_values}. "
                f"가능한 값 예시: {all_vals[:10]}"
            )
        gdf = gdf_f

    # --- 키 정규화 ---
    pop_df = pop_df.copy()
    pop_df[merge_on] = pop_df[merge_on].astype(str).str.strip()

    # --- 1차: 사용자가 준 merge_on으로 병합 시도 ---
    can_merge_direct = merge_on in gdf.columns
    if can_merge_direct:
        gdf[merge_on] = gdf[merge_on].astype(str).str.strip()
        merged = gdf.merge(pop_df[[merge_on, value_col]], on=merge_on, how="left")
    else:
        # --- 2차: 자동 매핑 시도 (이름 → 코드) ---
        # gdf의 코드/이름 후보 컬럼 찾아보기
        code_cands = [c for c in ["SIG_CD", "sig_cd", "ADM_CD", "adm_cd", "CODE", "code"] if c in gdf.columns]
        name_cands = [c for c in [merge_on, "sgg", "SIG_KOR_NM", "C1_NM_KOR", "C1_NM_ENG"] if c in gdf.columns]

        if code_cands and name_cands:
            code_col = code_cands[0]
            name_col = name_cands[0]
            # gdf의 (이름→코드) 매핑 만들기
            tmp = (
                gdf[[name_col, code_col]]
                .dropna()
                .assign(
                    **{
                        name_col: lambda d: d[name_col].astype(str).str.strip(),
                        code_col: lambda d: d[code_col].astype(str).str.strip(),
                    }
                )
                .drop_duplicates(subset=[name_col])
            )
            name_to_code = dict(zip(tmp[name_col], tmp[code_col]))

            # pop_df의 merge_on(이름) → 코드로 변환
            pop_df[code_col] = pop_df[merge_on].map(name_to_code)

            # 매핑 안 된 항목 경고
            unm = pop_df[pop_df[code_col].isna()][merge_on].unique().tolist()
            if unm:
                print(f"[WARN] 코드 매핑 실패 {len(unm)}개: {unm[:10]} ... "
                      f"(gdf의 '{name_col}' 값과 pop_df['{merge_on}']를 일치시키세요)")

            # 코드 기준 병합
            gdf[code_col] = gdf[code_col].astype(str).str.strip()
            merged = gdf.merge(pop_df[[code_col, value_col]], on=code_col, how="left")
        else:
            raise KeyError(
                f"GeoJSON에 '{merge_on}' 키가 없고, 자동 매핑을 위한 이름/코드 후보도 찾지 못했습니다. "
                f"gdf.columns={list(gdf.columns)}"
            )

    # --- 병합 결과 점검 ---
    if merged.empty:
        raise ValueError("merged가 비었습니다. merge_on 키/필터 조건을 확인하세요.")
    if merged.geometry.isna().all():
        raise ValueError("merged.geometry가 전부 NaN입니다. 필터/키를 확인하세요.")

    tb = merged.total_bounds
    if not np.isfinite(tb).all():
        raise ValueError(f"merged.total_bounds가 NaN/무한입니다: {tb}. 필터/키를 확인하세요.")

    # 누락 경고(머지 불일치)
    missing = merged[merged[value_col].isna()]
    if not missing.empty:
        miss_keys = missing[(merge_on if can_merge_direct else code_col)].unique().tolist()
        print(f"[WARN] 값 누락 {len(missing)}개 (머지 불일치) 예시: {miss_keys[:10]}")

    # --- 투영 후 플롯 (aspect 에러 방지) ---
    merged = merged.to_crs(3857)  # Web Mercator

    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.5,
        legend=True,
        missing_kwds={"color": "lightgray", "hatch": "///", "label": "No data"},
    )
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")
    ax.set_aspect("equal")

    # 자동 줌
    minx, miny, maxx, maxy = merged.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    plt.tight_layout()
    plt.show()



shape_path = "../PopulationData/seoul_map.geojson"
seoul_pop_df = build_seoul_population_df(shape_path=shape_path)
#plot_seoul_choropleth(shape_path=shape_path, pop_df=seoul_pop_df, title="Seoul Population (latest)")
