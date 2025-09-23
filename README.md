# Water Screening Lite — Phase I MVP (DEM + AWC + CLC2018)

Minimal Streamlit tool for **watershed characterization & recharge potential screening** using **DEM**, **AWC**, and **CLC2018** rasters.

## Scope (Lite)
- Inputs: portfolio points (CSV/GeoJSON) + DEM + AWC + CLC2018 rasters
- Outputs: elevation, slope, AWC, recharge class, land cover context (code/name), near_water, near_wetland
- Out of scope: ET/water balance, water stress, complex hydrology

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
# or run the memo dashboard:
streamlit run memo_app.py
```

## Accurate Slope via GDAL/QGIS (Optional)
Precompute slope once and upload it as an optional raster:
```bash
# Percent rise (recommended)
gdaldem slope DEM.tif slope_pct.tif -p
```
Then upload `slope_pct.tif` on the **Analysis** page. The app will sample that instead of using the built-in 3×3 slope.

See `tools/README_GDAL_QGIS.md` for details.


---

## Optional: Accurate Slope via GDAL/QGIS
See `scripts/SLOPE_PREPROCESS_GDAL_QGIS.md` or run:
```bash
python scripts/precompute_slope_gdal.py /path/to/dem.tif --out_dir outputs --reproject_to EPSG:4326
```
Then upload the produced **slope raster (percent)** in the app (optional field).

## Memo-Style Dashboard (Single Page)
Alongside `app.py`, a **memo dashboard** is available as `memo_app.py` presenting an executive-summary briefing with charts and prioritization. Run:
```bash
streamlit run memo_app.py
```
