import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_points(uploaded_file) -> gpd.GeoDataFrame:
    name = getattr(uploaded_file, "name", "upload")
    if name.lower().endswith((".geojson", ".json")):
        gdf = gpd.read_file(uploaded_file)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        if "asset_id" not in gdf.columns:
            gdf["asset_id"] = [f"site_{i+1}" for i in range(len(gdf))]
        return gdf.to_crs(4326)

    # CSV
    df = pd.read_csv(uploaded_file)
    lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("lon","lng","longitude")), None)
    if lat_col is None or lon_col is None:
        raise ValueError("CSV must include Latitude/Longitude (or lat/lon) columns.")
    if "asset_id" not in df.columns:
        df["asset_id"] = [f"site_{i+1}" for i in range(len(df))]
    geom = [Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    return gdf
