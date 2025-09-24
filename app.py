import os, sys, pathlib
# Robust import path so 'core/' is always found
APP_DIR = pathlib.Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Cap resources
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import tempfile

from core.io_utils import load_points
from core.analysis import run_analysis
from core.raster_ops import coverage_report_for_dem  # NEW

st.set_page_config(page_title="Water Screening Lite — Enhanced", layout="wide")

# Sidebar nav + diagnostics link
page = st.sidebar.radio("Navigation", ["1) Upload", "2) Analysis", "3) Results"])
try:
    st.sidebar.page_link("pages/01_Mini_Diagnostics.py", label="🧪 Mini Diagnostics")
    st.sidebar.page_link("pages/02_DEM_Probe.py", label="🧪 DEM Probe")
except Exception:
    st.sidebar.caption("Run diagnostics pages from the /pages folder.")

# Session state (avoid widget key collisions)
for k in ["points_gdf","results_gdf","dem_file","awc_file","clc_file","slope_file"]:
    if k not in st.session_state:
        st.session_state[k] = None

def _save_uploaded(tmpdir, uploaded_file, target_basename):
    if uploaded_file is None: return None
    suffix = pathlib.Path(uploaded_file.name).suffix.lower() or ".bin"
    path = pathlib.Path(tmpdir)/f"{target_basename}{suffix}"
    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
    return str(path)

def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None

def water_exec_summary(df: pd.DataFrame) -> str:
    n = len(df)
    pct_low  = (df["recharge_class"].eq("Low").mean()*100) if n else 0
    pct_med  = (df["recharge_class"].eq("Medium").mean()*100) if n else 0
    pct_high = (df["recharge_class"].eq("High").mean()*100) if n else 0
    stress   = int(df.get("water_stress_flag", pd.Series([False]*n)).sum()) if n else 0
    near_w   = int(df.get("near_water", pd.Series([False]*n)).sum()) if n else 0
    slope_q  = "precomputed" if (df.get("slope_quality_flag", pd.Series(["approx"]*n)).eq("precomputed").any()) else "approx"
    dom_lc = "Unknown"
    if "land_cover_name" in df.columns and n:
        dom_lc = df["land_cover_name"].fillna("Unknown").mode().iloc[0]
    return (
        f"**Portfolio Water Summary**\n"
        f"- Sites analyzed: **{n}**\n"
        f"- Recharge classes: **{pct_low:.0f}% Low**, **{pct_med:.0f}% Medium**, **{pct_high:.0f}% High**\n"
        f"- Water stress flags: **{stress}** site(s)\n"
        f"- Near water (CLC water): **{near_w}** site(s)\n"
        f"- Dominant land cover: **{dom_lc}**\n"
        f"- Slope quality: **{slope_q}**\n"
    )

def render_charts(df: pd.DataFrame):
    plt = _try_import_matplotlib()
    if plt is None:
        st.info("Charts skipped: `matplotlib` not installed. Add `matplotlib>=3.7` to requirements.txt.")
        return
    if "awc_mm" in df.columns:
        st.subheader("AWC Distribution")
        fig1 = plt.figure()
        df["awc_mm"].dropna().astype(float).plot(kind="hist", bins=20)
        plt.xlabel("AWC (mm)"); plt.ylabel("Count")
        st.pyplot(fig1)
    if {"slope_percent","awc_mm","recharge_class"}.issubset(df.columns):
        st.subheader("Slope vs AWC (colored by recharge class)")
        colors = df["recharge_class"].map({"Low":"tab:red","Medium":"tab:orange","High":"tab:green"}).fillna("tab:gray")
        fig2 = plt.figure()
        plt.scatter(df["awc_mm"], df["slope_percent"], s=40, c=colors)
        plt.xlabel("AWC (mm)"); plt.ylabel("Slope (%)")
        st.pyplot(fig2)
    if "recharge_class" in df.columns:
        st.subheader("Recharge Class Breakdown")
        counts = df["recharge_class"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig3 = plt.figure()
        plt.pie(counts.values, labels=counts.index, autopct="%1.0f%%")
        st.pyplot(fig3)

def build_html_memo(df: pd.DataFrame) -> str:
    n = len(df)
    pct_low  = (df["recharge_class"].eq("Low").mean()*100) if n else 0
    pct_med  = (df["recharge_class"].eq("Medium").mean()*100) if n else 0
    pct_high = (df["recharge_class"].eq("High").mean()*100) if n else 0
    stress   = int(df.get("water_stress_flag", pd.Series([False]*n)).sum()) if n else 0
    near_w   = int(df.get("near_water", pd.Series([False]*n)).sum()) if n else 0
    slope_q  = "precomputed" if (df.get("slope_quality_flag", pd.Series(["approx"]*n)).eq("precomputed").any()) else "approx"
    dom_lc = "Unknown"
    if "land_cover_name" in df.columns and n:
        dom_lc = df["land_cover_name"].fillna("Unknown").mode().iloc[0]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Water Screening Memo</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:8px; background:#eef; margin-right:6px; }}
.section {{ margin-top: 18px; }}
ul {{ margin-top: 6px; }}
</style></head>
<body>
<h1>Water Screening Memo</h1>
<div class="section">
  <div class="badge">Sites: {n}</div>
  <div class="badge">Low: {pct_low:.0f}%</div>
  <div class="badge">Medium: {pct_med:.0f}%</div>
  <div class="badge">High: {pct_high:.0f}%</div>
  <div class="badge">Stress flags: {stress}</div>
  <div class="badge">Near water: {near_w}</div>
  <div class="badge">Slope: {slope_q}</div>
</div>
<div class="section">
  <h2>Executive Summary</h2>
  <ul>
    <li>Dominant land cover: <b>{dom_lc}</b></li>
    <li>{pct_low:.0f}% of sites classify as <b>Low recharge</b> (screening-grade)</li>
    <li>{stress} site(s) flagged for potential <b>water stress</b> (high use on low recharge)</li>
  </ul>
</div>
<div class="section">
  <h2>Data Lineage & Assumptions</h2>
  <ul>
    <li><b>Slope:</b> {"precomputed slope raster (preferred)" if slope_q=="precomputed" else "3×3 DEM window (approx)"}.</li>
    <li><b>AWC & CLC sampling:</b> nearest pixel; screening-grade only.</li>
    <li><b>Scope:</b> No watershed delineation / ET / water balance in Phase I Lite.</li>
  </ul>
</div>
</body></html>"""

# ---------------------- PAGES ----------------------

if page.startswith("1"):
    st.title("Water Screening Lite — Enhanced")
    up = st.file_uploader("Upload sites (CSV or GeoJSON)", type=["csv","geojson"])
    c1,c2,c3 = st.columns(3)
    with c1:
        dem_up = st.file_uploader("DEM (.tif)", type=["tif","tiff"], key="dem_widget")
    with c2:
        awc_up = st.file_uploader("AWC (.tif)", type=["tif","tiff"], key="awc_widget")
    with c3:
        clc_up = st.file_uploader(
            "CLC2018 (raster .tif OR vector .gpkg/.geojson/.shp)",
            type=["tif","tiff","gpkg","geojson","json","shp"],
            key="clc_widget",
        )
    slope_up = st.file_uploader("Slope raster (percent, optional, .tif)", type=["tif","tiff"], key="slope_widget")

    # Store in our own session keys (distinct from widget keys)
    if dem_up:   st.session_state["dem_file"] = dem_up
    if awc_up:   st.session_state["awc_file"] = awc_up
    if clc_up:   st.session_state["clc_file"] = clc_up
    if slope_up: st.session_state["slope_file"] = slope_up

    if up:
        try:
            gdf = load_points(up)
            st.session_state["points_gdf"] = gdf
            st.success(f"Loaded {len(gdf)} point(s).")
            st.map(gdf.to_crs(4326))
        except Exception as e:
            st.error(f"Failed to load points: {e}")

elif page.startswith("2"):
    st.title("Run Analysis")
    if st.session_state["points_gdf"] is None:
        st.warning("Please upload a portfolio first (Page 1).")
    else:
        # thresholds
        try:
            thresholds = yaml.safe_load((APP_DIR/'config'/'thresholds.yaml').read_text())
        except Exception:
            thresholds = {"recharge":{"high":{"awc_min":150,"slope_max":5},"medium":{"awc_min":50,"slope_max":15}}}

        dem_up   = st.session_state.get("dem_file")
        awc_up   = st.session_state.get("awc_file")
        clc_up   = st.session_state.get("clc_file")
        slope_up = st.session_state.get("slope_file")

        if dem_up and awc_up and clc_up:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Persist files
                dem_path   = _save_uploaded(tmpdir, dem_up,  "dem")
                awc_path   = _save_uploaded(tmpdir, awc_up,  "awc")
                clc_path   = _save_uploaded(tmpdir, clc_up,  "clc")
                slope_path = _save_uploaded(tmpdir, slope_up, "slope_pct") if slope_up else None

                # Preflight coverage (bounds only)
                cov = coverage_report_for_dem(st.session_state["points_gdf"], dem_path)
                st.caption(f"DEM coverage: {cov['n_inside_bounds']}/{cov['n_total']} points inside tile bounds.")
                if cov["n_inside_bounds"] < cov["n_total"] and cov["n_total"] > 0:
                    st.warning("Some points lie outside the DEM tile bounds. They will return no elevation (nodata). "
                               "Use a larger DEM mosaic/VRT or reduce test radius.")

                if st.button("🚀 Run Screening", type="primary"):
                    with st.spinner("Analyzing point(s)..."):
                        try:
                            out = run_analysis(
                                st.session_state["points_gdf"],
                                dem_path, awc_path, clc_path, thresholds,
                                slope_file=slope_path
                            )
                            # Store a plain DataFrame for display/export
                            df_out = pd.DataFrame(out.drop(columns=["geometry"], errors="ignore"))
                            st.session_state["results_gdf"] = df_out
                            # Nodata warnings
                            if "dem_nodata_flag" in df_out.columns:
                                n_dem_no = int(df_out["dem_nodata_flag"].sum())
                                if n_dem_no:
                                    st.warning(f"{n_dem_no} site(s) have no elevation (DEM nodata/out of bounds).")
                            if "awc_nodata_flag" in df_out.columns:
                                n_awc_no = int(df_out["awc_nodata_flag"].sum())
                                if n_awc_no:
                                    st.warning(f"{n_awc_no} site(s) have no AWC (nodata/out of bounds).")
                            st.success("Done. See Results page.")
                        except Exception as e:
                            st.error(f"Analysis failed {e}")
        else:
            st.info("Upload DEM, AWC, and CLC (slope optional) then click Run.")

else:
    st.title("Results")
    out_df = st.session_state.get("results_gdf")
    if out_df is None or len(out_df) == 0:
        st.info("No results yet. Run the analysis on Page 2.")
    else:
        # Executive Summary
        st.header("Executive Summary (Water)")
        st.info(water_exec_summary(out_df))

        # Charts
        with st.expander("Show charts (AWC, Slope vs AWC, Recharge breakdown)", expanded=True):
            render_charts(out_df)

        # Results table + downloads
        st.header("Detailed Results")
        st.dataframe(out_df)
        st.download_button("Download CSV", out_df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")

        # GeoJSON export if coords exist
        if {"longitude","latitude"}.issubset(out_df.columns):
            try:
                gdf = gpd.GeoDataFrame(out_df, geometry=gpd.points_from_xy(out_df['longitude'], out_df['latitude']), crs="EPSG:4326")
                st.download_button("Download GeoJSON", gdf.to_json().encode("utf-8"), "results.geojson", "application/geo+json")
            except Exception as e:
                st.error(f"GeoJSON export failed: {e}")

        # HTML memo download
        memo_html = build_html_memo(out_df)
        st.download_button(
            "Download Water Memo (HTML)",
            memo_html.encode("utf-8"),
            file_name="water_memo.html",
            mime="text/html"
        )
