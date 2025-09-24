# tools/dem_probe.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "128")

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

st.set_page_config(page_title="DEM Probe", layout="centered")
st.title("🧪 DEM Sampling Probe")

sites_up = st.file_uploader("Upload sites (CSV with latitude, longitude)", type=["csv"])
dem_up   = st.file_uploader("Upload DEM (.tif)", type=["tif","tiff"])

if sites_up and dem_up and st.button("Run probe"):
    df = pd.read_csv(sites_up)
    assert {"latitude","longitude"}.issubset(df.columns), "CSV must have 'latitude' and 'longitude'."
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    lonlats = list(zip(df["longitude"].tolist(), df["latitude"].tolist()))

    st.subheader("DEM metadata")
    with rasterio.open(dem_up) as src:
        st.write({
            "src_crs": str(src.crs),
            "src_res": src.res,
            "src_width": src.width,
            "src_height": src.height,
            "src_nodata": src.nodata,
            "src_transform": tuple(src.transform)
        })

        # A) SAMPLE DIRECTLY IN SOURCE CRS (no VRT)
        pts_src = gdf.to_crs(src.crs) if src.crs else gdf
        coords_src = [(p.x, p.y) for p in pts_src.geometry]
        vals_src = [float(v[0]) if v is not None else None for v in src.sample(coords_src)]
        st.write("Direct sample (source CRS) — first 10:", vals_src[:10])
        st.write("Unique values (source CRS):", len(set(vals_src)))

        # B) SAMPLE THROUGH WGS84 VRT (current code path)
        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear, src_nodata=src.nodata) as vrt:
            rows_cols = [vrt.index(lon, lat) for lon, lat in lonlats]
            vals_vrt = [float(v[0]) if v is not None else None for v in vrt.sample(lonlats)]
            st.write("VRT sample (WGS84) — first 10:", vals_vrt[:10])
            st.write("Unique values (VRT):", len(set(vals_vrt)))
            st.write("Unique (row,col) hits:", len(set(rows_cols)), "of", len(rows_cols))
            st.write("Example (row,col) pairs:", rows_cols[:10])

    st.info("Interpretation:\n"
            "- If 'Unique (row,col)' is 1 in VRT but >1 in Source CRS → VRT collapse → patch VRT open.\n"
            "- If both show 1 unique value → DEM constant/nodata or points duplicated.\n"
            "- If both show many uniques but same *value* → DEM is flat in this area (unlikely) or nodata fill.")
