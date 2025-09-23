# tools/self_check.py
import sys, json, yaml
import geopandas as gpd
from shapely.geometry import Point
from core.analysis import run_analysis

def main():
    if len(sys.argv) < 5:
        print("Usage: python tools/self_check.py LAT LON DEM.tif AWC.tif CLC.(tif|gpkg) [SLOPE.tif]")
        return 1
    lat = float(sys.argv[1]); lon = float(sys.argv[2])
    dem = sys.argv[3]; awc = sys.argv[4]; clc = sys.argv[5]
    slope = sys.argv[6] if len(sys.argv) > 6 else None

    gdf = gpd.GeoDataFrame({"asset_id":["A1"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    thresholds = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}

    try:
        out = run_analysis(gdf, dem, awc, clc, thresholds, slope_file=slope)
        print(out.drop(columns=["geometry"], errors="ignore").to_string(index=False))
        return 0
    except Exception as e:
        print(f"FAILED {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
