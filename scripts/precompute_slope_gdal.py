#!/usr/bin/env python3
"""Helper to preprocess a DEM into a slope raster (percent) using GDAL tools.
Requires: gdal (gdaldem, gdal_calc.py, gdalwarp) in PATH.
"""
import subprocess as sp
import argparse
from pathlib import Path
import sys

def run(cmd):
    print("+", " ".join(cmd)); sys.stdout.flush()
    sp.check_call(cmd)

def main():
    ap = argparse.ArgumentParser(description="Create slope raster (percent) from DEM using GDAL.")
    ap.add_argument("dem", help="Path to DEM GeoTIFF")
    ap.add_argument("--out_dir", default=".", help="Output directory")
    ap.add_argument("--reproject_to", default=None, help="Target CRS (e.g., EPSG:4326). Optional.")
    args = ap.parse_args()

    dem = Path(args.dem)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    slope_deg = out_dir / (dem.stem + "_slope_deg.tif")
    slope_pct = out_dir / (dem.stem + "_slope_pct.tif")

    # 1) slope in degrees
    run(["gdaldem", "slope", str(dem), str(slope_deg), "-compute_edges"])

    # 2) convert to percent with gdal_calc
    run([
        "gdal_calc.py",
        "-A", str(slope_deg),
        "--outfile=" + str(slope_pct),
        "--calc=tan(A*3.1415926535/180.0)*100.0",
        "--NoDataValue=0"
    ])

    # 3) optional reprojection
    if args.reproject_to:
        slope_reproj = out_dir / (dem.stem + f"_slope_pct_{args.reproject_to.replace(':','_')}.tif")
        run([
            "gdalwarp",
            "-t_srs", args.reproject_to,
            "-r", "bilinear",
            str(slope_pct),
            str(slope_reproj),
        ])
        print("Slope raster (percent, reprojected):", slope_reproj)
    else:
        print("Slope raster (percent):", slope_pct)

if __name__ == "__main__":
    main()
