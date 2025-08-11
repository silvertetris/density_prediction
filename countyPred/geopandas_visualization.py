import geopandas as gpd

def abstract_geojson(path: str, sample_n: int = 5):
    gdf = gpd.read_file(path)
    print("✅ Loaded:", path)
    print("- shape:", gdf.shape)                  # (rows, cols)
    print("- columns:", list(gdf.columns))        # 모든 컬럼명
    print("- dtypes:\n", gdf.dtypes)              # 컬럼 타입
    print("- crs:", gdf.crs)                      # 좌표계 (예: EPSG:4326, EPSG:5179 등)
    print("- geometry type:", gdf.geometry.geom_type.unique().tolist())
    print("- total_bounds [minx, miny, maxx, maxy]:", gdf.total_bounds)

    # 표본 행
    print(f"\n- sample({sample_n}):")
    display_cols = [c for c in gdf.columns if c != "geometry"][:10]  # geometry 제외, 앞 10개만
    print(gdf[display_cols].head(sample_n))

    return gdf

gdf = abstract_geojson("../PopulationData/seoul_map.geojson")

import matplotlib.pyplot as plt

def quick_plot(gdf, title="Quick view"):
    ax = gdf.to_crs(4326).plot(figsize=(6,6), edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.axis("off")
    plt.show()

quick_plot(gdf, "Korea boundary (quick view)")
