# core/raster_ops.py
from typing import List, Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT

def open_reader_wgs84(path: str):
    """Open raster and return a reader in WGS84 (WarpedVRT if needed)."""
    src = rasterio.open(path)
    if src.crs and src.crs.to_string() != "EPSG:4326":
        vrt = WarpedVRT(src, crs="EPSG:4326")
        return src, vrt  # caller must close both
    return src, src     # same handle (close once)

def batch_extract_elevation(reader, coords_lonlat: List[Tuple[float, float]]) -> List[float]:
    vals = []
    for v in reader.sample(coords_lonlat):
        vals.append(float(v[0]) if v is not None else None)
    return vals

def batch_slope_percent_3x3(reader, coords_lonlat: List[Tuple[float, float]]) -> List[float]:
    """Compute simple Horn 3×3 slope (%) per coord using a single open reader."""
    out = []
    # estimate meters per degree by latitude to turn degrees/pixel into meters
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
            transform = reader.window_transform(win)
            dx_deg, dy_deg = transform.a, -transform.e
            mlat, mlon = m_per_deg(lat)
            dx_m = dx_deg * mlon
            dy_m = dy_deg * mlat
            if dx_m == 0 or dy_m == 0:
                out.append(0.0); continue
            dzdx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0]))/(8*dx_m)
            dzdy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2]))/(8*dy_m)
            slope_pct = np.tan(np.arctan(np.sqrt(dzdx**2 + dzdy**2))) * 100.0
            out.append(float(slope_pct))
        except Exception:
            out.append(None)
    return out

def sample_raster_at_points(gdf, raster_path):
    """(Keep your existing implementation if it already opens VRT once).
    Consider refactoring to open_reader_wgs84 for consistency."""
    values = []
    with rasterio.open(raster_path) as src:
        reader = WarpedVRT(src, crs="EPSG:4326") if src.crs and src.crs.to_string() != "EPSG:4326" else src
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        for val in reader.sample(coords):
            values.append(float(val[0]) if val is not None else None)
        if isinstance(reader, WarpedVRT):
            reader.close()
    return values
