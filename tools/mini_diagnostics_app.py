import os
# Cap resources to avoid native crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
import pathlib, tempfile

# Import repo helpers if available; otherwise define minimal fallbacks
try:
    from core.raster_ops import open_reader_wgs84, batch_extract_elevation, batch_slope_percent_3x3, sample_raster_at_points
    from core.clc_vector import assign_clc_code_to_points
except Exception as e:
    st.stop()

st.set_page_config(page_title="Mini Diagnostics — DEM/AWC/CLC", layout="centered")

st.title("🧪 Mini Diagnostics — DEM / AWC / CLC")
st.caption("Runs three independent checks to reveal which stage fails.")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=41.750139, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=-0.687249, format="%.6f")

    dem_up   = st.file_uploader("DEM (.tif)", type=["tif","tiff"], key="dem")
    slope_up = st.file_uploader("Slope raster (percent, optional, .tif)", type=["tif","tiff"], key="slope")
    awc_up   = st.file_uploader("AWC (.tif)", type=["tif","tiff"], key="awc")
    clc_up   = st.file_uploader("CLC2018 (raster .tif OR vector .gpkg/.geojson/.shp)",
                                type=["tif","tiff","gpkg","geojson","json","shp"], key="clc")
    submitted = st.form_submit_button("Run Diagnostics")

def _save_uploaded(tmpdir, uploaded_file, target_basename):
    if uploaded_file is None:
        return None
    suffix = pathlib.Path(uploaded_file.name).suffix.lower() or ".bin"
    path = pathlib.Path(tmpdir) / f"{target_basename}{suffix}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)

if submitted:
    # Build single-point GeoDataFrame
    gdf = gpd.GeoDataFrame({"id":[1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path   = _save_uploaded(tmpdir, dem_up,   "dem")
        slope_path = _save_uploaded(tmpdir, slope_up, "slope_pct") if slope_up else None
        awc_path   = _save_uploaded(tmpdir, awc_up,   "awc")
        clc_path   = _save_uploaded(tmpdir, clc_up,   "clc")

        if not dem_path or not awc_path or not clc_path:
            st.error("Please upload DEM, AWC, and CLC inputs (slope optional).")
            st.stop()

        # Stage 1: DEM slope/elevation
        st.subheader("Stage 1 — DEM slope/elevation")
        try:
            src, rdr = open_reader_wgs84(dem_path)
            if slope_path:
                st.write("Sampling **precomputed slope** raster (percent).")
                slope_vals = sample_raster_at_points(gdf, slope_path)
            else:
                st.write("Computing **3×3 slope** on DEM (screening-grade).")
                coords = [(lon, lat)]
                slope_vals = batch_slope_percent_3x3(rdr, coords)
            elev_vals = batch_extract_elevation(rdr, [(lon, lat)])
            if rdr is not src: rdr.close()
            src.close()
            st.success(f"[stage:dem_slope_elev] OK — elev={elev_vals[0]}, slope={slope_vals[0]}")
        except Exception as e:
            st.error(f"[stage:dem_slope_elev] FAILED: {e}")
            st.stop()

        # Stage 2: AWC sample
        st.subheader("Stage 2 — AWC sample")
        try:
            awc_vals = sample_raster_at_points(gdf, awc_path)
            st.success(f"[stage:awc_sample] OK — awc={awc_vals[0]}")
        except Exception as e:
            st.error(f"[stage:awc_sample] FAILED: {e}")
            st.stop()

        # Stage 3: CLC (raster/vector)
        st.subheader("Stage 3 — CLC sample/join")
        try:
            if pathlib.Path(clc_path).suffix.lower() in [".tif", ".tiff"]:
                clc_vals = sample_raster_at_points(gdf, clc_path)
            else:
                clc_vals = assign_clc_code_to_points(gdf, clc_path)
            st.success(f"[stage:clc] OK — code={clc_vals[0]}")
        except Exception as e:
            st.error(f"[stage:clc] FAILED: {e}")
            st.stop()

        st.balloons()
        st.info("✅ All stages passed. If the full app still crashes, the issue is likely outside these core reads (e.g., report generation).")
