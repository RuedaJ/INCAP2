import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import yaml
import io
import tempfile, os
from core.io_utils import load_points
from core.raster_ops import sample_raster_at_points, extract_elevation, slope_percent_3x3
from core.land_cover import CLC_NAMES, WATER_BODIES, WETLANDS

st.set_page_config(page_title="Water Screening Lite", layout="wide")

for key, default in {"points_gdf": None, "results_gdf": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

page = st.sidebar.radio("Navigation", ["1) Upload", "2) Analysis", "3) Results"])

def classify_recharge(awc_mm, slope_percent, thresholds):
    hi = thresholds["recharge"]["high"]
    med = thresholds["recharge"]["medium"]
    if (awc_mm is not None and awc_mm > hi["awc_min"]) and (slope_percent is not None and slope_percent < hi["slope_max"]):
        return "High"
    if (awc_mm is not None and awc_mm >= med["awc_min"]) or (slope_percent is not None and slope_percent <= med["slope_max"]):
        return "Medium"
    return "Low"

def decode_clc(code):
    name = CLC_NAMES.get(int(code), "Unknown") if code is not None and not np.isnan(code) else "Unknown"
    near_water = int(code) in WATER_BODIES if code is not None and not np.isnan(code) else False
    near_wetland = int(code) in WETLANDS if code is not None and not np.isnan(code) else False
    return name, near_water, near_wetland

def run_analysis(points_gdf, dem_file, awc_file, clc_file, thresholds, slope_path=None):
    gdf = points_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    # Elevation
    elevs = []
    for geom in gdf.geometry:
        lon, lat = geom.x, geom.y
        try:
            elev = extract_elevation(dem_file, lon, lat)
        except Exception:
            elev = None
        elevs.append(elev)

    # Slope: prefer uploaded slope raster (% rise), else compute 3x3
    if slope_path:
        slopes = sample_raster_at_points(gdf, slope_path)
    else:
        slopes = []
        for geom in gdf.geometry:
            lon, lat = geom.x, geom.y
            try:
                slope = slope_percent_3x3(dem_file, lon, lat)
            except Exception:
                slope = None
            slopes.append(slope)

    # AWC & CLC
    awc_vals = sample_raster_at_points(gdf, awc_file) if awc_file else [None]*len(gdf)
    clc_vals = sample_raster_at_points(gdf, clc_file) if clc_file else [None]*len(gdf)

    out = gdf.drop(columns=[c for c in ["geometry"] if c in gdf.columns]).copy()
    out["latitude"] = gdf.geometry.y
    out["longitude"] = gdf.geometry.x
    out["elevation_m"] = elevs
    out["slope_percent"] = slopes
    out["awc_mm"] = awc_vals
    out["land_cover_code"] = clc_vals
    decoded = [decode_clc(v) for v in clc_vals]
    out["land_cover_name"] = [d[0] for d in decoded]
    out["near_water"] = [d[1] for d in decoded]
    out["near_wetland"] = [d[2] for d in decoded]
    out["recharge_class"] = [classify_recharge(a, s, thresholds) for a, s in zip(awc_vals, slopes)]
    return out

# Page 1
if page.startswith("1"):
    st.title("Water Screening Lite — MVP")
    st.write("Upload portfolio and rasters (DEM, AWC, CLC2018).")

    up = st.file_uploader("Upload sites (CSV or GeoJSON)", type=["csv", "geojson"])
    c1, c2, c3 = st.columns(3)
    with c1:
        dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"], key="dem")
    with c2:
        awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"], key="awc")
    with c3:
        clc_up = st.file_uploader("CLC2018 (.tif)", type=["tif","tiff"], key="clc")
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

# Page 2
elif page.startswith("2"):
    st.title("Run Analysis")
    if st.session_state["points_gdf"] is None:
        st.warning("Please upload a portfolio first (Page 1).")
    else:
        try:
            thresholds = yaml.safe_load(Path("config/thresholds.yaml").read_text())
        except Exception:
            thresholds = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}

        st.info("Upload rasters if not already uploaded on Page 1.")
        dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"])
        awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"])
        clc_up = st.file_uploader("CLC2018 (.tif)", type=["tif","tiff"])
        slope_up = st.file_uploader("Optional: Slope raster (% rise, .tif)", type=["tif","tiff"])

        if dem_up and awc_up and clc_up and st.button("🚀 Run Screening", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                dem_path = os.path.join(tmpdir, "dem.tif")
                awc_path = os.path.join(tmpdir, "awc.tif")
                clc_path = os.path.join(tmpdir, "clc.tif")
                slope_path = os.path.join(tmpdir, "slope.tif") if slope_up else None
                files = [(dem_path, dem_up), (awc_path, awc_up), (clc_path, clc_up)]
                if slope_up:
                    files.append((slope_path, slope_up))
                for p, upf in files:
                    with open(p, "wb") as f: f.write(upf.getbuffer())
                with st.spinner("Analyzing points..."):
                    out = run_analysis(st.session_state["points_gdf"], dem_path, awc_path, clc_path, thresholds, slope_path=slope_path)
                    st.session_state["results_gdf"] = out
                    st.success("Done. See Results page.")
        else:
            st.warning("Upload DEM, AWC, and CLC rasters to proceed.")

# Page 3
else:
    st.title("Results")
    out = st.session_state.get("results_gdf")
    if out is None:
        st.info("No results yet. Run the analysis on Page 2.")
    else:
        st.dataframe(out)
        st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"), file_name="results.csv", mime="text/csv")
        try:
            gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["longitude"], out["latitude"]), crs="EPSG:4326")
            st.download_button("Download GeoJSON", gdf.to_json().encode("utf-8"), file_name="results.geojson", mime="application/geo+json")
        except Exception as e:
            st.error(f"GeoJSON export failed: {e}")
