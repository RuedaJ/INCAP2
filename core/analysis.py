# core/analysis.py
import os, pathlib
import numpy as np
import geopandas as gpd
from typing import Dict, Optional

from .raster_ops import (
    open_reader_wgs84,
    batch_extract_elevation,
    batch_slope_percent_3x3,
    sample_raster_at_points,
)
from .land_cover import CLC_NAMES, WATER_BODIES, WETLANDS

# Cap resources early to avoid native crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

def _classify_recharge(awc_mm, slope_percent, thr):
    hi, med = thr["recharge"]["high"], thr["recharge"]["medium"]
    try:
        a = None if awc_mm is None else float(awc_mm)
    except Exception:
        a = None
    try:
        s = None if slope_percent is None else float(slope_percent)
    except Exception:
        s = None
    if a is None or s is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(s, float) and np.isnan(s)):
        return "Low"  # conservative when missing
    if (a > hi["awc_min"]) and (s < hi["slope_max"]):
        return "High"
    if (a >= med["awc_min"]) or (s <= med["slope_max"]):
        return "Medium"
    return "Low"

def _awc_category(awc_mm, thr):
    if awc_mm is None or (isinstance(awc_mm, float) and np.isnan(awc_mm)):
        return "Unknown"
    a = float(awc_mm)
    if a < thr["recharge"]["medium"]["awc_min"]:
        return "Low"
    if a < thr["recharge"]["high"]["awc_min"]:
        return "Medium"
    return "High"

def _recharge_confidence(awc_mm, slope_percent, thr):
    # Heuristic distance from thresholds
    try:
        a = float(awc_mm); s = float(slope_percent)
    except Exception:
        return "low"
    if np.isnan(a) or np.isnan(s):
        return "low"
    awc_edges = [thr["recharge"]["medium"]["awc_min"], thr["recharge"]["high"]["awc_min"]]
    slope_edges = [thr["recharge"]["high"]["slope_max"], thr["recharge"]["medium"]["slope_max"]]
    awc_margin = min(abs(a - e) for e in awc_edges)
    slope_margin = min(abs(s - e) for e in slope_edges)
    if awc_margin >= 30 and slope_margin >= 5:
        return "high"
    if awc_margin >= 10 and slope_margin >= 2:
        return "medium"
    return "low"

def _decode_clc(code):
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ("Unknown", False, False)
    try:
        c = int(code)
    except Exception:
        return ("Unknown", False, False)
    return (CLC_NAMES.get(c, "Unknown"), c in WATER_BODIES, c in WETLANDS)

def run_analysis(points_gdf: gpd.GeoDataFrame,
                 dem_file: str,
                 awc_file: str,
                 clc_file: str,
                 thresholds: Dict,
                 slope_file: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Phase I Lite analysis with water-focused enrichments and nodata handling.
    Outputs include:
      - elevation_m, slope_percent, awc_mm
      - land_cover_code, land_cover_name, near_water, near_wetland
      - recharge_class, awc_category, recharge_confidence
      - slope_quality_flag, water_stress_flag
      - dem_nodata_flag, awc_nodata_flag
    """
    # Normalize input CRS
    gdf = points_gdf.to_crs(4326) if points_gdf.crs else points_gdf.set_crs(4326)
    coords = [(p.x, p.y) for p in gdf.geometry]

    # 1) DEM slope/elevation (mask nodata)
    try:
        src, reader = open_reader_wgs84(dem_file)
        slopes = sample_raster_at_points(gdf, slope_file) if slope_file else batch_slope_percent_3x3(reader, coords, src_nodata=src.nodata)
        elevs  = batch_extract_elevation(reader, coords, src_nodata=src.nodata)
        if reader is not src: reader.close()
        src.close()
    except Exception as e:
        raise RuntimeError(f"[stage:dem_slope_elev] {e}") from e

    # 2) AWC sampling (mask nodata)
    try:
        awc_vals = sample_raster_at_points(gdf, awc_file) if awc_file else [None]*len(gdf)
    except Exception as e:
        raise RuntimeError(f"[stage:awc_sample] {e}") from e

    # 3) CLC raster/vector
    try:
        suf = pathlib.Path(clc_file).suffix.lower()
        if suf in [".tif", ".tiff"]:
            clc_vals = sample_raster_at_points(gdf, clc_file)
        else:
            from .clc_vector import assign_clc_code_to_points
            clc_vals = assign_clc_code_to_points(gdf, clc_file)
    except Exception as e:
        raise RuntimeError(f"[stage:clc] {e}") from e

    # 4) Assemble
    out = gdf.copy()
    out["latitude"] = out.geometry.y
    out["longitude"] = out.geometry.x
    out["elevation_m"] = elevs
    out["slope_percent"] = slopes
    out["awc_mm"] = awc_vals
    out["land_cover_code"] = clc_vals

    # Decode CLC & water flags
    dec = [_decode_clc(v) for v in clc_vals]
    out["land_cover_name"] = [d[0] for d in dec]
    out["near_water"] = [d[1] for d in dec]
    out["near_wetland"] = [d[2] for d in dec]

    # Recharge & enrichments
    out["recharge_class"] = [_classify_recharge(a, s, thresholds) for a, s in zip(awc_vals, slopes)]
    out["awc_category"] = [_awc_category(a, thresholds) for a in awc_vals]
    out["recharge_confidence"] = [_recharge_confidence(a, s, thresholds) for a, s in zip(awc_vals, slopes)]
    out["slope_quality_flag"] = "precomputed" if slope_file else "approx"

    # Water stress (screening-grade)
    if "water_use_m3y" in out.columns:
        try:
            out["water_stress_flag"] = (out["recharge_class"] == "Low") & out["water_use_m3y"].fillna(0).astype(float).gt(0)
        except Exception:
            out["water_stress_flag"] = (out["recharge_class"] == "Low")
    else:
        out["water_stress_flag"] = False

    # Nodata flags
    out["dem_nodata_flag"] = out["elevation_m"].isna()
    out["awc_nodata_flag"] = out["awc_mm"].isna()

    return out
