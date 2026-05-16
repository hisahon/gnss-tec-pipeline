# -*- coding: utf-8 -*-
"""
Keograms of dTEC IPPs aggregating ALL stations into single global canvases
using datashader + dask. Generates TWO independent keograms per day:

  - Time × Latitude   (y_range = -60° to +60°)
  - Time × Longitude  (y_range = -120° to -20°, Americas sector)

Reads the per-station IPP parquets produced by `visualize_ipp.py` and
generates one .nc grid + one .png image per axis.

@author: hisas
"""

import argparse
from pathlib import Path

import numpy as np
import dask.dataframe as dd
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr
import colorcet as cc
import matplotlib.pyplot as plt

# Import custom modules
from shared.config import dtec_ipp_dir, dtec_keogram_dir


# Default ranges per axis ----------------------------------------------------
LAT_RANGE = (-60.0,  60.0)    # 120° span
LON_RANGE = (-120.0, -20.0)   # 100° span (Americas)


# ==========================================
# 1. Core: build aggregated grid
# ==========================================
def build_keogram_grid(
    parquet_files,
    y_col: str,
    y_range: tuple,
    plot_width: int = 5760,
    plot_height: int = 1200,
    dtec_col: str = "dtec_30min",
):
    """
    Aggregates IPP points from many parquets into a single (Time × y_col) grid.

    Parameters
    ----------
    parquet_files : list of Path
        Per-station IPP parquet files.
    y_col : str
        Column to put on the Y axis ('lts' for latitude, 'lns' for longitude).
    y_range : (float, float)
        Coordinate range covered by the canvas on the Y axis.
    plot_width : int
        Pixels along the time axis (default 5760 → 1 px per 15s over 24h).
    plot_height : int
        Pixels along the Y axis.
    dtec_col : str
        Column name for dTEC values.

    Returns
    -------
    xarray.DataArray
        Mean dTEC per pixel (NaN where no points fell).
    """
    if not parquet_files:
        raise FileNotFoundError("No IPP parquet files provided.")

    df_all = dd.read_parquet(parquet_files, columns=[y_col, dtec_col])
    df_all = df_all.dropna(subset=[y_col, dtec_col])

    # Datetime index → decimal hours
    df_all = df_all.reset_index()
    df_all["t_hours"] = (
        df_all["horario"].dt.hour
        + df_all["horario"].dt.minute / 60
        + df_all["horario"].dt.second / 3600
    )

    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(0.0, 24.0),
        y_range=y_range,
    )

    agg = canvas.points(df_all, "t_hours", y_col, agg=ds.mean(dtec_col))
    return agg


# ==========================================
# 2. Save grid
# ==========================================
def save_grid(agg, nc_path: Path):
    nc_path = Path(nc_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_netcdf(str(nc_path))
    print(f"Grid saved: {nc_path}")


# ==========================================
# 3. Plot from saved grid
# ==========================================
def plot_keogram(nc_path: Path, trh: float = 0.5, cmap=cc.coolwarm):
    nc_path  = Path(nc_path)
    agg      = xr.open_dataarray(nc_path)
    img      = tf.shade(agg, cmap=cmap, span=[-trh, trh], how="linear")
    img_path = nc_path.with_suffix("")
    img.to_pil().convert("RGB").save(str(img_path) + ".png")
    print(f"Image saved: {img_path}.png")


# ==========================================
# 3b. Plot both keograms in a single figure (matplotlib, two panels)
# ==========================================
def plot_keograms_mpl(nc_path_lat: Path, nc_path_lon: Path,
                      trh: float = 0.5, cmap: str = "RdBu_r",
                      dpi: int = 150, out_path: Path = None):
    """
    Loads both lat and lon NetCDF grids and renders them as a two-panel
    figure (Time × Lat on top, Time × Lon on bottom) sharing the X axis.

    Parameters
    ----------
    nc_path_lat, nc_path_lon : Path
        Paths to the .nc grids produced by save_grid().
    trh : float
        Symmetric color limits [-trh, +trh] in TECU.
    cmap : str
        Matplotlib colormap name.
    dpi : int
        Output resolution.
    out_path : Path, optional
        Output PNG path. Defaults to keograms_combined_*.png next to the .nc.
    """
    agg_lat = xr.open_dataarray(Path(nc_path_lat))
    agg_lon = xr.open_dataarray(Path(nc_path_lon))

    extent_lat = [
        float(agg_lat["t_hours"].min()), float(agg_lat["t_hours"].max()),
        float(agg_lat["lts"].min()),     float(agg_lat["lts"].max()),
    ]
    extent_lon = [
        float(agg_lon["t_hours"].min()), float(agg_lon["t_hours"].max()),
        float(agg_lon["lns"].min()),     float(agg_lon["lns"].max()),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             constrained_layout=True, sharex=True)

    # --- Top: Latitude ---
    im_lat = axes[0].imshow(
        agg_lat.values, extent=extent_lat, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh, interpolation="nearest",
    )
    axes[0].set_ylabel("Latitude (°)")
    axes[0].set_title(f"dTEC Keogram — {year} DOY {doy:03d}  |  30-min moving average  |  resolution: {resolution}")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    plt.colorbar(im_lat, ax=axes[0]).set_label("dTEC (TECU)")

    # --- Bottom: Longitude ---
    im_lon = axes[1].imshow(
        agg_lon.values, extent=extent_lon, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh, interpolation="nearest",
    )
    axes[1].set_ylabel("Longitude (°)")
    axes[1].set_xlabel("Time (UTC, hours)")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(np.arange(0, 25, 4))
    axes[1].grid(True, linestyle=":", alpha=0.4)
    plt.colorbar(im_lon, ax=axes[1]).set_label("dTEC (TECU)")

    if out_path is None:
        out_path = Path(nc_path_lat).parent / f"keogram_{year}{doy}_{resolution}.mpl.png"
    
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Image saved: {out_path}")


# ==========================================
# 4. CLI parsing
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build lat and lon keograms from IPP parquets via datashader."
    )
    parser.add_argument("--year",       type=int, required=True)
    parser.add_argument("--doy",        type=int, required=True)
    parser.add_argument("--resolution", choices=["low", "high"], default="low",
                        help="low = 0.1°/px, high = 0.05°/px.")
    parser.add_argument("--trh",        type=float, default=0.5)
    return parser.parse_args()


# ==========================================
# 5. Entry points
# ==========================================
def main_build(year, doy, resolution="low"):
    """
    Stage 1 — aggregate all station parquets into TWO NetCDF grids
    (one for latitude, one for longitude).
    """
    # Resolution in degrees per pixel
    deg_per_px      = 0.1 if resolution == "low" else 0.03
    plot_height_lat = int((LAT_RANGE[1] - LAT_RANGE[0]) / deg_per_px)
    plot_height_lon = int((LON_RANGE[1] - LON_RANGE[0]) / deg_per_px)

    ipp_folder      = dtec_ipp_dir(year, doy)
    keograms_folder = dtec_keogram_dir(year, doy)
    parquet_files   = sorted(Path(ipp_folder).glob(f"*_dtecIPP_{year}{doy}.parquet"))

    if not parquet_files:
        print(f"No IPP files found in {ipp_folder}")
        return

    print(f"Found {len(parquet_files)} station files. Aggregating...")

    # --- Latitude keogram ---
    print("Building lat keogram...")
    agg_lat = build_keogram_grid(
        parquet_files, y_col="lts", y_range=LAT_RANGE,
        plot_width=5760, plot_height=plot_height_lat,
    )
    save_grid(agg_lat, keograms_folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc")

    # --- Longitude keogram ---
    print("Building lon keogram...")
    agg_lon = build_keogram_grid(
        parquet_files, y_col="lns", y_range=LON_RANGE,
        plot_width=5760, plot_height=plot_height_lon,
    )
    save_grid(agg_lon, keograms_folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc")


def main_plot(year, doy, resolution="low", trh=0.5, cmap=cc.coolwarm,
              use_matplotlib: bool = False):
    """
    Stage 2 — load both saved grids and export their PNGs.

    Parameters
    ----------
    use_matplotlib : bool
        If True, combines lat and lon into a single two-panel figure
        (with axes, colorbar, labels). If False, exports two separate
        raw datashader images (faster, no axes).
    """
    keogram_folder = dtec_keogram_dir(year, doy)
    nc_lat = keogram_folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc"
    nc_lon = keogram_folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc"

    if not nc_lat.exists() or not nc_lon.exists():
        print(f"Grids not found. Run main_build() first.")
        return

    if use_matplotlib:
        mpl_cmap = "rainbow" if cmap is cc.coolwarm else cmap
        plot_keograms_mpl(nc_lat, nc_lon, trh=trh, cmap=mpl_cmap)
    else:
        plot_keogram(nc_lat, trh=trh, cmap=cmap)
        plot_keogram(nc_lon, trh=trh, cmap=cmap)


# ==========================================
# 6. IDE / CLI entry point
# ==========================================
if __name__ == "__main__":
    RUN_MANUALLY = True

    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")
        year, doy = 2025, 316
        resolution = 'high'
        # Stage 1 — run once
        main_build(year=year, doy=doy, resolution=resolution)

        # # Stage 2 — pick ONE of the two below:
        # # datashader raw image (fast, no axes)
        # main_plot(year=2025, doy=316, resolution="low", trh=1.0,
        #           use_matplotlib=False)

        # matplotlib version (with axes, colorbar, title)
        main_plot(year=year, doy=doy, resolution=resolution, trh=1.0,
                  use_matplotlib=True)

    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        main_build(year=args.year, doy=args.doy, resolution=args.resolution)
        main_plot( year=args.year, doy=args.doy, resolution=args.resolution,
                   trh=args.trh)