import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
import tempfile, os, pathlib
from shapely.geometry import Point

# Core helpers from repo
from core.io_utils import load_points
from core.land_cover import CLC_NAMES, WATER_BODIES, WETLANDS
from core.analysis import run_analysis  # shared, stage-tagged, memory-safe

st.set_page_config(page_title="Water Screening Lite", layout="wide")

# ---------------- Session init ----------------
for key, default in {"points_gdf": None, "results_gdf": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- Small helpers ----------------
def classify_recharge(awc_mm, slope_percent, thresholds):
    hi = thresholds["recharge"]["high"]
    med = thresholds["recharge"]["medium"]
    if (awc_mm is not None and awc_mm > hi["awc_min"]) and (slope_percent is not None and slope_percent < hi["slope_max"]):
        return "High"
    if (awc_mm is not None and awc_mm >= med["awc_min"]) or (slope_percent is not None and slope_percent <= med["slope_max"]):
        return "Medium"
    return "Low"

def decode_clc(code):
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Unknown", False, False
    c = int(code)
    return CLC_NAMES.get(c, "Unknown"), c in WATER_BODIES, c in WETLANDS

def _save_uploaded(tmpdir, uploaded_file, target_basename):
    """Save an UploadedFile to disk, preserving extension; return path or None."""
    if uploaded_file is None:
        return None
    suffix = pathlib.Path(uploaded_file.name).suffix.lower() or ".bin"
    path = os.path.join(tmpdir, target_basename + suffix)
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
        dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"], key="dem")
    with c2:
        awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"], key="awc")
    with c3:
        clc_up = st.file_uploader(
            "CLC2018 (raster .tif OR vector .gpkg/.geojson/.shp)",
            type=["tif","tiff","gpkg","geojson","json","shp"],
            key="clc"
        )
    st.markdown("**Optional:** Upload a precomputed SLOPE raster (percent)")
    slope_up = st.file_uploader("Slope raster (.tif)", type=["tif","tiff"], key="slope")

    if up:
        try:
            gdf = load_points(up)
            st.session_state["points_gdf"] = gdf
            st.success(f"Loaded {len(gdf)} points.")
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
            thresholds = yaml.safe_load(Path(thr_file).read_text())
        except Exception:
            thresholds = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}

        # Retrieve uploads from session (may be None)
        dem_up = st.session_state.get("dem")
        awc_up = st.session_state.get("awc")
        clc_up = st.session_state.get("clc")
        slope_up = st.session_state.get("slope")

        # If not present, ask again
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

        if dem_up and awc_up and clc_up:
            if st.button("🚀 Run Screening", type="primary"):
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    dem_path   = _save_uploaded(tmpdir, dem_up,  "dem")
                    awc_path   = _save_uploaded(tmpdir, awc_up,  "awc")
                    clc_path   = _save_uploaded(tmpdir, clc_up,  "clc")
                    slope_path = _save_uploaded(tmpdir, slope_up, "slope_pct") if slope_up else None
                    with st.spinner("Analyzing points..."):
                        try:
                            out = run_analysis(st.session_state["points_gdf"], dem_path, awc_path, clc_path, thresholds, slope_file=slope_path)
                            st.session_state["results_gdf"] = out
                            st.success("Done. See Results page.")
                        except Exception as e:
                            st.error(f"Analysis failed {e}")
        else:
            st.warning("Upload DEM, AWC, and CLC to proceed.")

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
