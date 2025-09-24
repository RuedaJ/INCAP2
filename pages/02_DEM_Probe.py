# pages/02_DEM_Probe.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.vrt,.gpkg")

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import io
import json

st.set_page_config(page_title="🧪 DEM Sampling Probe", layout="centered")
st.title("🧪 DEM Sampling Probe")

st.markdown(
    "Upload a **sites CSV** (`latitude, longitude`) and a **DEM GeoTIFF** to compare "
    "**direct source-CRS sampling** vs **WGS84 VRT sampling**. "
    "This helps diagnose “all points have same elevation” issues."
)

sites_up = st.file_uploader("Sites CSV", type=["csv"])
dem_up   = st.file_uploader("DEM (.tif)", type=["tif","tiff"])

run = st.button("Run probe", type="primary", disabled=not (sites_up and dem_up))

def _as_float_list(samples):
    out = []
    for v in samples:
        try:
            out.append(float(v[0]) if v is not None else None)
        except Exception:
            out.append(None)
    return out

if run:
    try:
        # ---- Load sites
        df = pd.read_csv(sites_up)
        assert {"latitude","longitude"}.issubset(df.columns), "CSV must have 'latitude' and 'longitude'."
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        lonlats = list(zip(df["longitude"].tolist(), df["latitude"].tolist()))

        # ---- DEM metadata + sampling
        with rasterio.open(dem_up) as src:
            meta = {
                "src_crs": str(src.crs),
                "src_res": src.res,
                "src_width": src.width,
                "src_height": src.height,
                "src_nodata": src.nodata,
                "src_transform": tuple(src.transform),
            }
            st.subheader("DEM metadata")
            st.json(meta)

            # A) Direct sample in SOURCE CRS (avoid VRT)
            pts_src = gdf.to_crs(src.crs) if src.crs else gdf
            coords_src = [(p.x, p.y) for p in pts_src.geometry]
            vals_src = _as_float_list(src.sample(coords_src))
            st.subheader("Direct Sample (Source CRS)")
            st.write("First 10 values:", vals_src[:10])
            st.write("Unique values (source CRS):", len(set([v for v in vals_src if v is not None])))

            # B) Sample through WGS84 VRT (like main app)
            with WarpedVRT(
                src,
                crs="EPSG:4326",
                resampling=Resampling.bilinear,   # DEM is continuous → bilinear
                src_nodata=src.nodata,
            ) as vrt:
                rows_cols = [vrt.index(lon, lat) for lon, lat in lonlats]
                vals_vrt = _as_float_list(vrt.sample(lonlats))
                st.subheader("VRT Sample (WGS84)")
                st.write("First 10 values:", vals_vrt[:10])
                st.write("Unique values (VRT):", len(set([v for v in vals_vrt if v is not None])))
                st.write("Unique (row,col) hits:", len(set(rows_cols)), "of", len(rows_cols))
                st.write("Example (row,col):", rows_cols[:10])

        # ---- Quick interpretation
        st.subheader("Interpretation guide")
        st.info(
            "- If **Unique (row,col) = 1** in VRT but **>1** in Source CRS → VRT collapse ⇒ "
            "fix `open_reader_wgs84` (set resampling & use source-like resolution) or sample in source CRS.\n"
            "- If **both** have 1 unique value → DEM constant/nodata (or input points duplicated).\n"
            "- If rows/cols vary but values identical → DEM band is flat here or nodata fill being returned."
        )

        # ---- Download a JSON probe report
        report = {
            "dem_meta": meta,
            "direct_sample_first10": vals_src[:10],
            "vrt_sample_first10": vals_vrt[:10],
            "vrt_unique_rowcol": len(set(rows_cols)),
            "n_points": len(lonlats),
        }
        buf = io.BytesIO(json.dumps(report, indent=2).encode("utf-8"))
        st.download_button("Download probe report (JSON)", data=buf, file_name="dem_probe_report.json", mime="application/json")

    except Exception as e:
        st.error(f"Probe failed: {e}")

# Sidebar link back to main pages (optional)
with st.sidebar:
    st.markdown("**Tips**")
    st.caption("• Prefer precomputed slope in projected CRS.\n• Ensure points are truly EPSG:4326 before VRT sampling.\n• Consider sampling in source CRS for a handful of sites.")
