import os
# Cap threads/cache early to avoid native crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
import tempfile, pathlib, os as _os

from core.io_utils import load_points
from core.land_cover import CLC_NAMES, WATER_BODIES, WETLANDS
from core.analysis import run_analysis
from core.raster_ops import open_reader_wgs84, batch_extract_elevation, batch_slope_percent_3x3, sample_raster_at_points
from core.clc_vector import assign_clc_code_to_points

st.set_page_config(page_title="Water Screening Lite", layout="wide")

# ---------------- Session init ----------------
for key, default in {"points_gdf": None, "results_gdf": None, "dem": None, "awc": None, "clc": None, "slope": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Helpers ----------------
def _save_uploaded(tmpdir, uploaded_file, target_basename):
    """Save an UploadedFile to disk, preserving extension; return path or None."""
    if uploaded_file is None:
        return None
    suffix = pathlib.Path(uploaded_file.name).suffix.lower() or ".bin"
    path = _os.path.join(tmpdir, target_basename + suffix)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ---------------- UI ----------------
page = st.sidebar.radio("Navigation", ["1) Upload", "2) Analysis", "3) Results"])

# Page 1 — Upload
if page.startswith("1"):
    st.title("Water Screening Lite — MVP")
    st.write("Upload portfolio and rasters (DEM, AWC, CLC2018). Optional: precomputed SLOPE raster (percent).")

    up = st.file_uploader("Upload sites (CSV or GeoJSON)", type=["csv","geojson"])

    c1, c2, c3 = st.columns(3)
    with c1:
        dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"], key="dem_upl")
    with c2:
        awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"], key="awc_upl")
    with c3:
        clc_up = st.file_uploader(
            "CLC2018 (raster .tif OR vector .gpkg/.geojson/.shp)",
            type=["tif","tiff","gpkg","geojson","json","shp"],
            key="clc_upl"
        )
    st.markdown("**Optional:** Upload a precomputed SLOPE raster (percent)")
    slope_up = st.file_uploader("Slope raster (.tif)", type=["tif","tiff"], key="slope_upl")

    # Cache uploads in session
    if dem_up:   st.session_state["dem"] = dem_up
    if awc_up:   st.session_state["awc"] = awc_up
    if clc_up:   st.session_state["clc"] = clc_up
    if slope_up: st.session_state["slope"] = slope_up

    if up:
        try:
            gdf = load_points(up)
            st.session_state["points_gdf"] = gdf
            st.success(f"Loaded {len(gdf)} point(s).")
            st.map(gdf.to_crs(4326))
        except Exception as e:
            st.error(f"Failed to load points: {e}")

# Page 2 — Analysis
elif page.startswith("2"):
    st.title("Run Analysis")
    if st.session_state["points_gdf"] is None:
        st.warning("Please upload a portfolio first (Page 1).")
    else:
        thr_file = "config/thresholds.yaml"
        try:
            thresholds = yaml.safe_load(pathlib.Path(thr_file).read_text())
        except Exception:
            thresholds = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}

        dem_up = st.session_state.get("dem")
        awc_up = st.session_state.get("awc")
        clc_up = st.session_state.get("clc")
        slope_up = st.session_state.get("slope")

        if not dem_up or not awc_up or not clc_up:
            st.info("Please (re)upload rasters / CLC here if needed:")
            c1, c2, c3 = st.columns(3)
            with c1:
                dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"])
            with c2:
                awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"])
            with c3:
                clc_up = st.file_uploader(
                    "CLC2018 (raster .tif OR vector .gpkg/.geojson/.shp)",
                    type=["tif","tiff","gpkg","geojson","json","shp"]
                )
            slope_up = st.file_uploader("Slope raster (percent, .tif) — optional", type=["tif","tiff"])

        # Paths
        dem_path = awc_path = clc_path = slope_path = None

        colA, colB = st.columns([1,1])
        with colA:
            if dem_up and awc_up and clc_up:
                if st.button("🚀 Run Screening", type="primary"):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        dem_path   = _save_uploaded(tmpdir, dem_up,  "dem")
                        awc_path   = _save_uploaded(tmpdir, awc_up,  "awc")
                        clc_path   = _save_uploaded(tmpdir, clc_up,  "clc")
                        slope_path = _save_uploaded(tmpdir, slope_up, "slope_pct") if slope_up else None
                        with st.spinner("Analyzing points..."):
                            try:
                                out = run_analysis(
                                    st.session_state["points_gdf"],
                                    dem_path, awc_path, clc_path, thresholds,
                                    slope_file=slope_path
                                )
                                st.session_state["results_gdf"] = out
                                st.success("Done. See Results page.")
                            except Exception as e:
                                st.error(f"Analysis failed {e}")
            else:
                st.warning("Upload DEM, AWC, and CLC to proceed.")

        with colB:
            st.subheader("🔍 Diagnostics (single site)")
            st.caption("Run each stage separately to reveal the failing step without crashing the app.")
            if st.button("Run diagnostics"):
                try:
                    # Use temporary persisted files if not created yet
                    with tempfile.TemporaryDirectory() as tmpdir:
                        if dem_path is None and dem_up:   dem_path   = _save_uploaded(tmpdir, dem_up,  "dem")
                        if awc_path is None and awc_up:   awc_path   = _save_uploaded(tmpdir, awc_up,  "awc")
                        if clc_path is None and clc_up:   clc_path   = _save_uploaded(tmpdir, clc_up,  "clc")
                        if slope_path is None and slope_up: slope_path = _save_uploaded(tmpdir, slope_up, "slope_pct")

                        gdf = st.session_state["points_gdf"].to_crs(4326)
                        assert len(gdf) >= 1, "No points loaded"
                        p = gdf.geometry.iloc[0]
                        coords = [(p.x, p.y)]

                        st.info("Stage: DEM open & slope/elevation")
                        src, rdr = open_reader_wgs84(dem_path)
                        if slope_path:
                            st.info("Sampling precomputed slope raster")
                            slope_vals = sample_raster_at_points(gdf.iloc[[0]], slope_path)
                        else:
                            st.info("Computing 3×3 slope on DEM")
                            slope_vals = batch_slope_percent_3x3(rdr, coords)
                        elev_vals = batch_extract_elevation(rdr, coords)
                        if rdr is not src: rdr.close()
                        src.close()
                        st.success(f"DEM OK — elev={elev_vals[0]}, slope={slope_vals[0]}")

                        st.info("Stage: AWC sample")
                        awc_vals = sample_raster_at_points(gdf.iloc[[0]], awc_path)
                        st.success(f"AWC OK — {awc_vals[0]}")

                        st.info("Stage: CLC (raster/vector)")
                        if pathlib.Path(clc_path).suffix.lower() in [".tif", ".tiff"]:
                            clc_vals = sample_raster_at_points(gdf.iloc[[0]], clc_path)
                        else:
                            clc_vals = assign_clc_code_to_points(gdf.iloc[[0]], clc_path)
                        st.success(f"CLC OK — code={clc_vals[0]}")
                except Exception as e:
                    st.exception(e)

# Page 3 — Results
else:
    st.title("Results")
    out = st.session_state.get("results_gdf")
    if out is None:
        st.info("No results yet. Run the analysis on Page 2.")
    else:
        st.dataframe(out)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name="results.csv", mime="text/csv")
        try:
            gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["longitude"], out["latitude"]), crs="EPSG:4326")
            geojson_bytes = gdf.to_json().encode("utf-8")
            st.download_button("Download GeoJSON", geojson_bytes, file_name="results.geojson", mime="application/geo+json")
        except Exception as e:
            st.error(f"GeoJSON export failed: {e}")
