# -*- coding: utf-8 -*-
"""
Keogram of dTEC IPPs (Time × Latitude) aggregating ALL stations into a
single global canvas using datashader + dask.

Reads the per-station IPP parquets produced by `visualize_ipp.py` and
generates:
  - a NetCDF file with the aggregated grid (for re-plotting later)
  - a PNG image of the keogram

@author: hisas
"""

import argparse
from pathlib import Path

import numpy as np
import dask.dataframe as dd
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr

# Import custom modules
from shared.config import dtec_ipp_dir


# ==========================================
# 1. Core: build aggregated grid
# ==========================================
def build_keogram_grid(
    parquet_files,
    plot_width: int = 5760,
    plot_height: int = 1200,
    y_range: tuple = (-60.0, 60.0),
    dtec_col: str = "dtec_30min",
    lat_col: str = "lts",
):
    """
    Aggregates IPP points from many parquets into a single (Time × Lat) grid.

    Parameters
    ----------
    parquet_files : list of Path
        Per-station IPP parquet files (must contain `horario` index + lat/dtec columns).
    plot_width : int
        Pixels along the time axis (default 5760 → 1 px per 15s over 24h).
    plot_height : int
        Pixels along the latitude axis (1200 = 0.1°/px, 2400 = 0.05°/px).
    y_range : (float, float)
        Latitude range covered by the canvas.
    dtec_col, lat_col : str
        Column names for dTEC and latitude.

    Returns
    -------
    xarray.DataArray
        Mean dTEC per pixel (NaN where no points fell).
    """
    if not parquet_files:
        raise FileNotFoundError("No IPP parquet files provided.")

    # Lazy read all files at once
    df_all = dd.read_parquet(
        parquet_files,
        columns=[lat_col, dtec_col],   # `horario` comes in as the index
    )
    df_all = df_all.dropna(subset=[lat_col, dtec_col])

    # Convert datetime index → decimal hours (vectorized via dask)
    df_all = df_all.reset_index()
    df_all["t_hours"] = (
        df_all["horario"].dt.hour
        + df_all["horario"].dt.minute / 60
        + df_all["horario"].dt.second / 3600
    )

    # Canvas covering the full day
    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(0.0, 24.0),
        y_range=y_range,
    )

    # Mean dTEC per pixel — empty pixels stay NaN
    agg = canvas.points(df_all, "t_hours", lat_col, agg=ds.mean(dtec_col))
    return agg


# ==========================================
# 2. Save grid (NetCDF only)
# ==========================================
def save_grid(agg, nc_path: Path):
    """
    Saves the aggregated DataArray to NetCDF.

    Parameters
    ----------
    agg : xarray.DataArray
        Output of build_keogram_grid().
    nc_path : Path
        Output .nc file path.
    """
    nc_path = Path(nc_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_netcdf(nc_path)
    print(f"Grid saved: {nc_path}")


# ==========================================
# 3. Plot from saved grid
# ==========================================
def plot_keogram(nc_path: Path, trh: float = 0.5, cmap: str = "RdBu_r"):
    """
    Loads a saved NetCDF grid and exports a shaded PNG image.
    Independent from the build step — can be called multiple times
    with different cmap/trh without re-aggregating.

    Parameters
    ----------
    nc_path : Path
        Path to the .nc file produced by save_grid().
    trh : float
        Symmetric color limits [-trh, +trh] in TECU.
    cmap : str
        Matplotlib colormap name.
    """
    nc_path  = Path(nc_path)
    agg      = xr.open_dataarray(nc_path)
    img      = tf.shade(agg, cmap=cmap, span=[-trh, trh], how="linear")
    img_path = nc_path.with_suffix("")   # same name, no extension (datashader adds .png)
    tf.export_image(img, str(img_path))
    print(f"Image saved: {img_path}.png")


# ==========================================
# 4. CLI parsing
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build a keogram (Time × Lat) from IPP parquets via datashader."
    )
    parser.add_argument("--year",       type=int, required=True)
    parser.add_argument("--doy",        type=int, required=True)
    parser.add_argument("--resolution", choices=["low", "high"], default="low",
                        help="low = 0.1°/px (1200px), high = 0.05°/px (2400px).")
    parser.add_argument("--trh",        type=float, default=0.5,
                        help="Symmetric dTEC color limits [-trh, +trh].")
    parser.add_argument("--cmap",       type=str,   default="RdBu_r")
    return parser.parse_args()


# ==========================================
# 5. Entry points
# ==========================================
def main_build(year, doy, resolution="low"):
    """
    Stage 1 — aggregate all station parquets into a single NetCDF grid.
    Run once per day/resolution. Output: keogram_{year}_{doy}_{resolution}.nc
    """
    plot_height   = 1200 if resolution == "low" else 2400
    ipp_folder    = dtec_ipp_dir(year, doy)
    parquet_files = sorted(Path(ipp_folder).glob(f"*_dtecIPP_{year}{doy}.parquet"))

    if not parquet_files:
        print(f"No IPP files found in {ipp_folder}")
        return

    print(f"Found {len(parquet_files)} station files. Aggregating...")
    agg     = build_keogram_grid(parquet_files, plot_width=5760, plot_height=plot_height)
    nc_path = Path(ipp_folder) / f"keogram_{year}_{doy:03d}_{resolution}.nc"
    save_grid(agg, nc_path)


def main_plot(year, doy, resolution="low", trh=0.5, cmap="RdBu_r"):
    """
    Stage 2 — load saved grid and export PNG.
    Can be called many times with different trh/cmap without re-aggregating.
    """
    ipp_folder = dtec_ipp_dir(year, doy)
    nc_path    = Path(ipp_folder) / f"keogram_{year}_{doy:03d}_{resolution}.nc"

    if not nc_path.exists():
        print(f"Grid not found: {nc_path}. Run main_build() first.")
        return

    plot_keogram(nc_path, trh=trh, cmap=cmap)


# ==========================================
# 6. IDE / CLI entry point
# ==========================================
if __name__ == "__main__":
    # =====================================================
    # SPYDER / IDE MANUAL OVERRIDE
    # =====================================================
    RUN_MANUALLY = True

    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")

        # Stage 1 — run once
        main_build(year=2025, doy=316, resolution="low")

        # Stage 2 — run as many times as needed
        main_plot(year=2025, doy=316, resolution="low", trh=1.0, cmap="RdBu_r")

    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        main_build(year=args.year, doy=args.doy, resolution=args.resolution)
        main_plot( year=args.year, doy=args.doy, resolution=args.resolution,
                   trh=args.trh,  cmap=args.cmap)