# Slope Preprocessing with GDAL/QGIS

This optional step creates an accurate **slope raster** from your DEM for the MVP to sample,
instead of computing an approximate 3×3 slope per point.

## Option A — GDAL (CLI)

```bash
# 1) Create slope raster in degrees (or percent) from DEM
gdaldem slope dem.tif slope_deg.tif -compute_edges

# (Optional) Convert degrees -> percent
# slope_% = tan(deg2rad(slope_deg)) * 100
# You can do this with gdal_calc.py:
gdal_calc.py -A slope_deg.tif --outfile=slope_pct.tif --calc="tan(A*3.1415926535/180.0)*100.0" --NoDataValue=0

# 2) Reproject slope raster to WGS84 for consistent on-the-fly sampling in app (optional)
gdalwarp -t_srs EPSG:4326 -r bilinear slope_pct.tif slope_pct_wgs84.tif
```

**Notes**
- Keep **units** consistent with the app (percent). Bilinear resampling is recommended for continuous rasters.
- You may also keep the raster in its native CRS; the app can sample via VRT, but pre-warping can be faster for large rasters.

## Option B — QGIS GUI

1. **Raster ► Analysis ► Slope** → Input DEM, Output slope (degrees).  
2. (Optional) **Raster Calculator**: `tan("slope@1" * pi()/180) * 100` → slope in percent.  
3. (Optional) **Reproject** with **Raster ► Projections ► Warp (reproject)** to EPSG:4326, resampling = bilinear.

## Using in the App

On **Upload** or **Analysis** page, provide the optional **Slope raster (.tif)**.  
If provided, the app will sample slope from this raster and **skip** the approximate 3×3 computation.
