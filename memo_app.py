import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from core.io_utils import load_points
from core.raster_ops import sample_raster_at_points, extract_elevation, slope_percent_3x3
from core.land_cover import CLC_NAMES, WATER_BODIES, WETLANDS
import tempfile, os, yaml

st.set_page_config(page_title="Water Risk Memo", layout="wide")

st.title("🌊 Water Risk Screening Memo — Lite MVP")

# Uploads
with st.expander("📤 Upload Data", expanded=True):
    sites_up = st.file_uploader("Sites (CSV/GeoJSON)", type=["csv","geojson"])
    c1,c2,c3,c4 = st.columns(4)
    with c1:
       dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"])
    with c2:
       awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"])
    with c3:
       clc_up = st.file_uploader("CLC2018 (.tif)", type=["tif","tiff"])
    with c4:
       slope_up = st.file_uploader("Slope raster (percent, .tif, optional)", type=["tif","tiff"])
    thr = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}
    st.caption("Recharge thresholds (editable in config/thresholds.yaml in full app).")

def classify_recharge(awc_mm, slope_percent, thr):
    hi = thr["recharge"]["high"]; med = thr["recharge"]["medium"]
    if (awc_mm is not None and awc_mm > hi["awc_min"]) and (slope_percent is not None and slope_percent < hi["slope_max"]):
        return "High"
    if (awc_mm is not None and awc_mm >= med["awc_min"]) or (slope_percent is not None and slope_percent <= med["slope_max"]):
        return "Medium"
    return "Low"

def analyze(gdf, dem, awc, clc, slope=None):
    gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    # slope
    if slope:
        slope_vals = sample_raster_at_points(gdf, slope)
    else:
        slope_vals = [slope_percent_3x3(dem, x.y, x.x) if False else None for x in gdf.geometry]  # placeholder; we compute below properly
        # compute approximate per point
        slope_vals = []
        for geom in gdf.geometry:
            s = slope_percent_3x3(dem, geom.x, geom.y)
            slope_vals.append(s)
    # elevation
    elevs = []
    for geom in gdf.geometry:
        elevs.append(extract_elevation(dem, geom.x, geom.y))
    # awc/clc
    awc_vals = sample_raster_at_points(gdf, awc)
    clc_vals = sample_raster_at_points(gdf, clc)
    # assemble
    out = gdf.copy()
    out["latitude"] = out.geometry.y
    out["longitude"] = out.geometry.x
    out["elevation_m"] = elevs
    out["slope_percent"] = slope_vals
    out["awc_mm"] = awc_vals
    out["land_cover_code"] = clc_vals
    out["land_cover_name"] = [CLC_NAMES.get(int(c), "Unknown") if c is not None and not np.isnan(c) else "Unknown" for c in clc_vals]
    out["near_water"] = [int(c) in WATER_BODIES if c is not None and not np.isnan(c) else False for c in clc_vals]
    out["near_wetland"] = [int(c) in WETLANDS if c is not None and not np.isnan(c) else False for c in clc_vals]
    out["recharge_class"] = [classify_recharge(a, s, thr) for a,s in zip(awc_vals, slope_vals)]
    return out

if st.button("🚀 Generate Analysis", type="primary") and sites_up and dem_up and awc_up and clc_up:
    with tempfile.TemporaryDirectory() as tmpdir:
        dem_path = os.path.join(tmpdir, "dem.tif"); open(dem_path,"wb").write(dem_up.getbuffer())
        awc_path = os.path.join(tmpdir, "awc.tif"); open(awc_path,"wb").write(awc_up.getbuffer())
        clc_path = os.path.join(tmpdir, "clc.tif"); open(clc_path,"wb").write(clc_up.getbuffer())
        slope_path = None
        if slope_up:
            slope_path = os.path.join(tmpdir, "slope_pct.tif"); open(slope_path,"wb").write(slope_up.getbuffer())
        gdf = load_points(sites_up)
        with st.spinner("Analyzing..."):
            results = analyze(gdf, dem_path, awc_path, clc_path, slope=slope_path)
        st.success("Analysis complete.")

        # Executive summary
        st.header("Executive Summary")
        n = len(results)
        pct_low = (results["recharge_class"].eq("Low").mean()*100.0) if n else 0
        dominant_lc = results["land_cover_name"].mode().iat[0] if n and not results["land_cover_name"].isna().all() else "Unknown"
        pct_near_water = (results["near_water"].mean()*100.0) if n else 0
        st.info(f"""**Portfolio Water Risk Summary**
- {n} site(s) analyzed
- {pct_low:.1f}% classified as **Low recharge**
- {pct_near_water:.1f}% located near **water bodies** (CLC classes)
- Dominant land cover: **{dominant_lc}**
""")

        # Analytics: simple charts using st.bar_chart / st.pyplot
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Recharge Class Distribution")
            st.bar_chart(results["recharge_class"].value_counts())
        with col2:
            st.subheader("Slope (%) — Distribution")
            st.line_chart(results["slope_percent"].dropna().sort_values().reset_index(drop=True))
        with col3:
            st.subheader("AWC (mm) — Distribution")
            st.line_chart(pd.Series(results["awc_mm"]).dropna().sort_values().reset_index(drop=True))

        # Detailed table and downloads
        st.header("Detailed Results")
        st.dataframe(results.drop(columns=["geometry"]))
        st.download_button("⬇️ Download CSV", results.drop(columns=["geometry"]).to_csv(index=False).encode("utf-8"), "results_memo.csv", "text/csv")
        try:
            gjson = gpd.GeoDataFrame(results, geometry=results.geometry, crs="EPSG:4326").to_json().encode("utf-8")
            st.download_button("⬇️ Download GeoJSON", gjson, "results_memo.geojson", "application/geo+json")
        except Exception as e:
            st.warning(f"GeoJSON export warning: {e}")

st.caption("Note: Slope from precomputed raster (percent) is preferred for accuracy. The fallback 3×3 computation is an approximation appropriate for screening only.")
