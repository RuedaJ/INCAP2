import geopandas as gpd
import numpy as np

# Try common CORINE code field names
CLC_CODE_FIELDS = [
    "CODE_18", "CLC_CODE", "CLC_CODE18", "code_18", "CODE", "CLC_CODE_18"
]

def load_clc_vector(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("CLC vector has no CRS; please set CRS on the source data.")
    # Find code field
    code_field = None
    for f in CLC_CODE_FIELDS:
        if f in gdf.columns:
            code_field = f
            break
    if code_field is None:
        # fallback: numeric-looking column?
        for c in gdf.columns:
            if gdf[c].dtype.kind in ("i", "u", "f"):
                code_field = c
                break
    if code_field is None:
        raise ValueError("Could not find a CORINE code field (e.g., CODE_18/CLC_CODE) in the GeoPackage.")

    # Keep only needed fields
    keep_cols = [code_field]
    gdf = gdf[keep_cols + ["geometry"]]
    gdf = gdf.rename(columns={code_field: "CLC_CODE"})
    return gdf

def assign_clc_code_to_points(points_wgs84: gpd.GeoDataFrame, clc_vector_path: str) -> list:
    """Returns a list of CLC codes (ints or np.nan) via spatial join."""
    clc = load_clc_vector(clc_vector_path)
    # Reproject CLC to points' CRS to avoid on-the-fly reprojection
    clc = clc.to_crs(points_wgs84.crs)
    # Spatial join (point-in-polygon)
    joined = gpd.sjoin(points_wgs84[["geometry"]].copy(), clc, how="left", predicate="intersects")
    codes = joined["CLC_CODE"].to_list()
    # normalize to numeric (or NaN)
    codes = [int(c) if c is not None and not (isinstance(c, float) and np.isnan(c)) else np.nan for c in codes]
    return codes
