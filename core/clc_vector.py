# core/clc_vector.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import geopandas as gpd

try:
    import pyogrio  # noqa: F401
    HAS_PYOGRIO = True
except Exception:
    HAS_PYOGRIO = False

CLC_CODE_FIELDS: Sequence[str] = (
    "CODE_18","CLC_CODE","CLC_CODE18","code_18","CODE","CLC_CODE_18"
)

def _detect_layer(path: str) -> Optional[str]:
    try:
        if HAS_PYOGRIO:
            layers = gpd.io.file.fiona.drvsupport.supported_drivers  # dummy touch
            layers = __import__('pyogrio').list_layers(path)
            # Prefer polygon layers
            for name, geomtype, *_ in layers:
                if (geomtype or '').lower().find('polygon') >= 0:
                    return name
            return layers[0][0] if layers else None
        else:
            import fiona
            layers = fiona.listlayers(path)
            return layers[0] if layers else None
    except Exception:
        return None

def _read_vector_bbox(path: str, bbox: Tuple[float,float,float,float], layer: Optional[str]) -> gpd.GeoDataFrame:
    # Force pyogrio engine if available to avoid loading entire GPKG
    if HAS_PYOGRIO:
        return gpd.read_file(path, layer=layer, bbox=bbox, engine="pyogrio")
    return gpd.read_file(path, layer=layer, bbox=bbox)

def _pick_code_field(gdf: gpd.GeoDataFrame) -> Optional[str]:
    for f in CLC_CODE_FIELDS:
        if f in gdf.columns:
            return f
    for c in gdf.columns:
        if gdf[c].dtype.kind in ('i','u','f'):
            return c
    return None

def load_clc_vector_bbox(path: str, points_wgs84: gpd.GeoDataFrame, simplify_tol_m: float = 0.0) -> gpd.GeoDataFrame:
    # For single-site use, simplification set to 0 by default (no topology changes)
    if points_wgs84.crs is None:
        points_wgs84 = points_wgs84.set_crs(4326)
    elif points_wgs84.crs.to_epsg() != 4326:
        points_wgs84 = points_wgs84.to_crs(4326)
    # Tight bbox around point(s) with small padding to ensure intersection
    xmin, ymin, xmax, ymax = points_wgs84.total_bounds
    pad = 0.0005  # ~50 m at mid-lat
    bbox = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)

    layer = _detect_layer(path)
    gdf = _read_vector_bbox(path, bbox=bbox, layer=layer)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        raise ValueError("CLC vector has no CRS.")

    code_field = _pick_code_field(gdf)
    if code_field is None:
        raise ValueError("Cannot find CLC code field (e.g., CODE_18/CLC_CODE).")

    gdf = gdf[[code_field, "geometry"]].rename(columns={code_field: "CLC_CODE"})

    # Optional simplify in metric CRS for larger extents (off by default for single site)
    if simplify_tol_m and simplify_tol_m > 0:
        clc_3035 = gdf.to_crs(3035)
        clc_3035["geometry"] = clc_3035.geometry.simplify(simplify_tol_m, preserve_topology=True)
        clc_3035 = clc_3035[~clc_3035.geometry.is_empty & clc_3035.geometry.notnull()]
        gdf = clc_3035.to_crs(4326)

    def _norm(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return int(v)
        except Exception:
            return np.nan
    gdf["CLC_CODE"] = gdf["CLC_CODE"].map(_norm)
    return gdf

def assign_clc_code_to_points(points_wgs84: gpd.GeoDataFrame, clc_path: str) -> list:
    clc = load_clc_vector_bbox(clc_path, points_wgs84, simplify_tol_m=0.0)
    if clc.empty:
        return [np.nan] * len(points_wgs84)
    # Spatial index may be unavailable; handle gracefully
    try:
        _ = clc.sindex
    except Exception:
        pass
    joined = gpd.sjoin(points_wgs84[["geometry"]].copy(),
                       clc[["CLC_CODE", "geometry"]],
                       how="left", predicate="intersects")
    return joined["CLC_CODE"].to_list()
