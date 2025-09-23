# GDAL / QGIS Slope Preprocessing

## GDAL (CLI)
```bash
# Degrees
gdaldem slope DEM.tif slope_deg.tif

# Percent rise (recommended)
gdaldem slope DEM.tif slope_pct.tif -p
```

## QGIS GUI
Raster ► Analysis ► **Slope**
- Input: DEM
- Output measurement: **Percent**
- Save as: `slope_pct.tif`

Use the generated `slope_pct.tif` in the app's Analysis page as the optional **Slope raster** input.
