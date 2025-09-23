# core/analysis.py
import os, pathlib
import numpy as np
import geopandas as gpd
from typing import Dict, Optional
from .raster_ops import open_reader_wgs84, batch_extract_elevation, batch_slope_percent_3x3, sample_raster_at_points
from .land_cover import CLC_NAMES, WATER_BODIES, WETLANDS
from .clc_vector import assign_clc_code_to_points

def classify_recharge(awc_mm, slope_percent, thr):
    hi, med = thr["recharge"]["high"], thr["recharge"]["medium"]
    if (awc_mm is not None and awc_mm > hi["awc_min"]) and (slope_percent is not None and slope_percent < hi["slope_max"]):
        return "High"
    if (awc_mm is not None and awc_mm >= med["awc_min"]) or (slope_percent is not None and slope_percent <= med["slope_max"]):
        return "Medium"
    return "Low"

def decode_clc(code):
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ("Unknown", False, False)
    c = int(code)
    return (CLC_NAMES.get(c, "Unknown"), c in WATER_BODIES, c in WETLANDS)

def run_analysis(points_gdf: gpd.GeoDataFrame,
                 dem_file: str,
                 awc_file: str,
                 clc_file: str,
                 thresholds: Dict,
                 slope_file: Optional[str] = None) -> gpd.GeoDataFrame:
    # Cap GDAL/threads to avoid OOM
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("GDAL_CACHEMAX", "128")  # MB
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

    gdf = points_gdf.to_crs(4326) if points_gdf.crs else points_gdf.set_crs(4326)
    coords = [(p.x, p.y) for p in gdf.geometry]

    # DEM-based slope & elevation
    try:
        src, reader = open_reader_wgs84(dem_file)
        slopes = sample_raster_at_points(gdf, slope_file) if slope_file else batch_slope_percent_3x3(reader, coords)
        elevs  = batch_extract_elevation(reader, coords)
        if reader is not src: reader.close()
        src.close()
    except Exception as e:
        raise RuntimeError(f"[stage:dem_slope_elev] {e}") from e

    # AWC sampling
    try:
        awc_vals = sample_raster_at_points(gdf, awc_file) if awc_file else [None]*len(gdf)
    except Exception as e:
        raise RuntimeError(f"[stage:awc_sample] {e}") from e

    # CLC raster/vector
    try:
        suf = pathlib.Path(clc_file).suffix.lower()
        if suf in [".tif", ".tiff"]:
            clc_vals = sample_raster_at_points(gdf, clc_file)
        else:
            clc_vals = assign_clc_code_to_points(gdf, clc_file)
    except Exception as e:
        raise RuntimeError(f"[stage:clc] {e}") from e

    # Assemble
    out = gdf.drop(columns=[c for c in ["geometry"] if c in gdf.columns]).copy()
    out["latitude"] = gdf.geometry.y
    out["longitude"] = gdf.geometry.x
    out["elevation_m"] = elevs
    out["slope_percent"] = slopes
    out["awc_mm"] = awc_vals
    out["land_cover_code"] = clc_vals
    dec = [decode_clc(v) for v in clc_vals]
    out["land_cover_name"] = [d[0] for d in dec]
    out["near_water"] = [d[1] for d in dec]
    out["near_wetland"] = [d[2] for d in dec]
    out["recharge_class"] = [classify_recharge(a, s, thresholds) for a, s in zip(awc_vals, slopes)]
    return out
