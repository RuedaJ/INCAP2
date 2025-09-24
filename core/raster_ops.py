# core/raster_ops.py
from typing import List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# --- Open a WGS84 reader with sane defaults for DEM sampling ---
def open_reader_wgs84(path: str):
    """
    Returns (src, reader) where src is the original dataset handle and
    reader is either src or a WarpedVRT to EPSG:4326.
    Uses bilinear resampling and carries src_nodata.
    """
    src = rasterio.open(path)
    if src.crs and src.crs.to_string() != "EPSG:4326":
        vrt = WarpedVRT(
            src,
            crs="EPSG:4326",
            resampling=Resampling.bilinear,  # continuous DEM → bilinear
            src_nodata=src.nodata,
        )
        return src, vrt
    return src, src

def _mask_nodata(val: Optional[float], nodata: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if nodata is None:
        return v
    if np.isnan(v):
        return None
    if v == nodata:
        return None
    return v

# --- Elevation sampling (returns list of floats or None where nodata) ---
def batch_extract_elevation(reader, coords_lonlat: List[Tuple[float, float]], src_nodata: Optional[float]=None) -> List[Optional[float]]:
    vals = []
    for v in reader.sample(coords_lonlat):
        raw = float(v[0]) if v is not None else None
        vals.append(_mask_nodata(raw, src_nodata))
    return vals

# --- 3x3 Horn slope in %-units, on the provided reader (WGS84) ---
def batch_slope_percent_3x3(reader, coords_lonlat: List[Tuple[float, float]], src_nodata: Optional[float]=None) -> List[Optional[float]]:
    from rasterio.windows import Window
    out = []

    def m_per_deg(lat):
        mlat = 111320.0
        mlon = 111320.0 * np.cos(np.deg2rad(lat))
        return mlat, mlon

    for lon, lat in coords_lonlat:
        try:
            row, col = reader.index(lon, lat)
            r0, c0 = row - 1, col - 1
            if r0 < 0 or c0 < 0 or r0 + 3 > reader.height or c0 + 3 > reader.width:
                out.append(None); continue
            win = Window(c0, r0, 3, 3)
            z = reader.read(1, window=win).astype(float)
            # If any 3x3 value is nodata, return None
            if src_nodata is not None and np.any(z == src_nodata):
                out.append(None); continue
            transform = reader.window_transform(win)
            dx_deg, dy_deg = transform.a, -transform.e
            mlat, mlon = m_per_deg(lat)
            dx_m = dx_deg * mlon
            dy_m = dy_deg * mlat
            if dx_m == 0 or dy_m == 0:
                out.append(None); continue
            dzdx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0]))/(8*dx_m)
            dzdy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2]))/(8*dy_m)
            slope_pct = (dzdx**2 + dzdy**2) ** 0.5 * 100.0
            out.append(float(slope_pct))
        except Exception:
            out.append(None)
    return out

# --- Generic single-band sampling with nodata masking (nearest by default) ---
def sample_raster_at_points(gdf, raster_path: str) -> List[Optional[float]]:
    vals = []
    with rasterio.open(raster_path) as src:
        reader = (
            WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=src.nodata)
            if src.crs and src.crs.to_string() != "EPSG:4326" else src
        )
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        for v in reader.sample(coords):
            raw = float(v[0]) if v is not None else None
            vals.append(_mask_nodata(raw, src.nodata))
        if reader is not src:
            reader.close()
    return vals

# --- Coverage preflight for DEM (bounds-only quick check) ---
def coverage_report_for_dem(points_wgs84_gdf, dem_path: str) -> dict:
    """
    Returns how many points fall within the DEM tile bounds (fast heuristic).
    """
    with rasterio.open(dem_path) as src:
        pts = points_wgs84_gdf.to_crs(src.crs) if src.crs else points_wgs84_gdf
        xmin, ymin, xmax, ymax = src.bounds
        xs, ys = pts.geometry.x.values, pts.geometry.y.values
        inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
        n_inside = int(inside.sum())
    return {"n_total": len(pts), "n_inside_bounds": n_inside}
