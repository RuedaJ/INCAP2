from typing import Tuple
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT

def sample_raster_at_points(gdf, raster_path):
    """Sample raster values at point coords (auto-reproject to WGS84)."""
    values = []
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            reader = WarpedVRT(src, crs="EPSG:4326")
        else:
            reader = src
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        for val in reader.sample(coords):
            values.append(float(val[0]) if val is not None else None)
        if isinstance(reader, WarpedVRT):
            reader.close()
    return values

def slope_percent_3x3(dem_path, lon, lat):
    """Compute slope (%) via Horn 3x3 window around a WGS84 point (screening)."""
    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            reader = WarpedVRT(src, crs="EPSG:4326")
        else:
            reader = src

        row, col = reader.index(lon, lat)
        r0, c0 = row-1, col-1
        if r0 < 0 or c0 < 0 or r0+3 > reader.height or c0+3 > reader.width:
            if isinstance(reader, WarpedVRT): reader.close()
            return None
        window = Window(c0, r0, 3, 3)
        z = reader.read(1, window=window).astype(float)

        transform = reader.window_transform(window)
        dx_deg = transform.a
        dy_deg = -transform.e
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))
        dx_m = dx_deg * meters_per_deg_lon
        dy_m = dy_deg * meters_per_deg_lat

        dzdx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0]))/(8*dx_m) if dx_m != 0 else 0.0
        dzdy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2]))/(8*dy_m) if dy_m != 0 else 0.0
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope_pct = np.tan(slope_rad) * 100.0

        if isinstance(reader, WarpedVRT): reader.close()
        return float(slope_pct)

def extract_elevation(dem_path, lon, lat):
    with rasterio.open(dem_path) as src:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            reader = WarpedVRT(src, crs="EPSG:4326")
        else:
            reader = src
        val = list(reader.sample([(lon, lat)]))[0][0]
        if isinstance(reader, WarpedVRT): reader.close()
        return float(val)
