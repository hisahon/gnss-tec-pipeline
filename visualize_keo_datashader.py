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
import re
from pathlib import Path

import numpy as np
import dask.dataframe as dd
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Importe o módulo ticker

from scipy.ndimage import uniform_filter1d

import matplotlib as mpl
mpl.use("Agg")
import pandas as pd

# Import custom modules
from shared.config import dtec_ipp_dir, dtec_keogram_dir, dtec_keogram_figs_dir, storms_SouthAmerica





# ==========================================
# 1. Core: build aggregated grid
# ==========================================

# import numpy as np


def smooth_vertical_nan(arr, size, axis=0, treat_zero_as_missing=True):
    """Vertical moving average (degree axis) ignoring NaN and, optionally,
    zeros (empty bins). size=None or <=1 returns the array untouched."""
    if size is None or size <= 1:
        return arr

    arr = arr.astype(float)
    valid = np.isfinite(arr)
    if treat_zero_as_missing:
        valid &= (arr != 0.0)

    filled = np.where(valid, arr, 0.0)
    num = uniform_filter1d(filled, size=size, axis=axis, mode="nearest")
    den = uniform_filter1d(valid.astype(float), size=size, axis=axis, mode="nearest")

    return np.divide(num, den, out=np.full_like(arr, np.nan), where=den > 0)




def build_keogram_grid(
    parquet_files,
    y_col: str,
    y_range: tuple,
    plot_width: int = 5760,
    plot_height: int = 1200,
    dtec_col: str = "dtec_30min",
):
    if not parquet_files:
        raise FileNotFoundError("No IPP parquet files provided.")

    # 1. Leia apenas as colunas que não são índice
    df_all = dd.read_parquet(parquet_files, columns=[y_col, dtec_col])
    df_all = df_all.dropna(subset=[y_col, dtec_col])
    
    # 2. O reset_index() transforma o índice original (horario) em uma coluna
    df_all = df_all.reset_index()

    # 3. Agora a coluna "horario" existe para o cálculo dos segundos
    df_all["t_seconds"] = (
        df_all["horario"].dt.hour * 3600
        + df_all["horario"].dt.minute * 60
        + df_all["horario"].dt.second
    ).astype("int32")

    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(0, 86400),   
        y_range=y_range,
    )

    # 1. Mapeia os extremos superiores (picos) e inferiores (depleções)
    agg_max = canvas.points(df_all, "t_seconds", y_col, agg=ds.max(dtec_col))
    agg_min = canvas.points(df_all, "t_seconds", y_col, agg=ds.min(dtec_col))

    # 2. Preenche NaNs temporariamente com 0 apenas para a condição booleana
    # Isso evita avisos de erro (RuntimeWarning) ao comparar NaNs no numpy
    max_mag = np.abs(agg_max.fillna(0))
    min_mag = np.abs(agg_min.fillna(0))

    # 3. Combina as matrizes: Retorna agg_max se a magnitude dele for maior, senão agg_min
    # Como as matrizes originais (agg_max e agg_min) contêm os NaNs, os "buracos" da grade são preservados
    agg_abs_max = xr.where(max_mag >= min_mag, agg_max, agg_min)

    return agg_abs_max


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



def plot_keograms_storm_mpl(nc_path_lat: Path, nc_path_lon: Path,
                            trh: float = 0.5, cmap: str = "RdBu_r",
                            smooth_size: int = None,
                            plot_storms: bool = False,
                            dpi: int = 300, out_path: Path = None):

    # --- Storm overlay styling (fixed, not exposed in the signature) -
    storm_temp_col  = "temp_min_C"
    storm_cmap      = "rainbow"
    storm_vmin      = -90.0
    storm_vmax      = -65.0
    storm_linewidth = 6.0
    storm_alpha     = 0.25

    resolutions = {
        "low": 0.1,
        "high": 0.05,
        "veryhigh": 0.025,
        "ultra": 0.005,
    }

    agg_lat = xr.open_dataarray(Path(nc_path_lat))
    agg_lon = xr.open_dataarray(Path(nc_path_lon))

    # --- Optional vertical smoothing (degree axis only) -----------------
    # NaN- and zero-aware moving average along the latitude/longitude axis.
    # smooth_size None/<=1 keeps the original data untouched.
    lat_vals = agg_lat.values
    lon_vals = agg_lon.values
    if smooth_size:
        lat_vals = smooth_vertical_nan(lat_vals, smooth_size)
        lon_vals = smooth_vertical_nan(lon_vals, smooth_size)

        # degrees-per-pixel derived from each grid's own coordinates
        deg_per_px_lat = abs(float(agg_lat["lts"][1] - agg_lat["lts"][0]))
        deg_per_px_lon = abs(float(agg_lon["lns"][1] - agg_lon["lns"][0]))
        print(f"[smoothing] ON — vertical window = {smooth_size} px "
              f"(≈ {smooth_size * deg_per_px_lat:.3f}° lat, "
              f"{smooth_size * deg_per_px_lon:.3f}° lon). "
              f"To turn OFF, call with smooth_size=None (or omit it).")
    else:
        print("[smoothing] OFF — raw data. "
              "To turn ON, pass smooth_size=<n_pixels> (e.g. smooth_size=5).")

    extent_lat = [
        float(agg_lat["t_seconds"].min())/3600, float(agg_lat["t_seconds"].max())/3600,
        float(agg_lat["lts"].min()),     float(agg_lat["lts"].max()),
    ]
    extent_lon = [
        float(agg_lon["t_seconds"].min())/3600, float(agg_lon["t_seconds"].max())/3600,
        float(agg_lon["lns"].min()),     float(agg_lon["lns"].max()),
    ]

    # --- Storms overlay: pre-compute X (hours), Y spans and colors ---
    # Loaded once and reused on both panels. If plot_storms is False,
    # nothing is overlaid — original behavior is preserved.
    storms_data = None
    if plot_storms:
        print("[storms] ON — overlay enabled. "
              "To turn OFF, call with plot_storms=False.")
        df_storms = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")
        # equivalent radius in degrees from area (km²)
        df_storms['radius_deg'] = np.sqrt(df_storms['area_km2'] / np.pi) / 110.0

        # clip to the target day (same module-level year/doy used in the title)
        t0 = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        t1 = t0 + pd.Timedelta(days=1)
        df_storms = df_storms[
            (df_storms['datetime'] >= t0) & (df_storms['datetime'] < t1)
        ].copy()

        if not df_storms.empty:
            dt = pd.to_datetime(df_storms['datetime'])
            if dt.dt.tz is not None:
                dt = dt.dt.tz_convert(None)
            t_hours = (dt - t0).dt.total_seconds().to_numpy() / 3600.0

            norm_s = mpl.colors.Normalize(vmin=storm_vmin, vmax=storm_vmax)
            sm_s   = mpl.cm.ScalarMappable(norm=norm_s, cmap=storm_cmap)
            sm_s.set_array([])
            storm_colors = sm_s.to_rgba(df_storms[storm_temp_col].to_numpy())

            lon_c = df_storms['centroid_lon'].to_numpy()
            lat_c = df_storms['centroid_lat'].to_numpy()
            rad2  = df_storms['radius_deg'].to_numpy() / 2.0

            storms_data = dict(
                t=t_hours, colors=storm_colors, sm=sm_s,
                lon_lo=lon_c - rad2, lon_hi=lon_c + rad2,
                lat_lo=lat_c - rad2, lat_hi=lat_c + rad2,
            )
    else:
        print("[storms] OFF — no overlay. "
              "To turn ON, call with plot_storms=True.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             constrained_layout=True, sharex=True)
    # --- Top: Longitud ---

    im_lon = axes[0].imshow(
        lon_vals, extent=extent_lon, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh,
        interpolation="nearest",
    )

    axes[0].set_title(
        f"dTEC Keogram — {year} DOY {doy:03d}  |  {movavg_srt}-min moving average  |  "
        f"resolution: {resolutions.get(resolution, 0.1)} deg x 15 sec"
    )
    axes[0].set_ylabel("Longitud, deg")
    # axes[0].set_ylim(LON_RANGE)
    axes[0].set_ylim(extent_lon[2], extent_lon[3])

    axes[0].grid(True, linestyle=":", alpha=0.4)
    plt.colorbar(im_lon, ax=axes[0]).set_label("dTEC (TECU)")

    # Storm overlay (top panel: longitude span)
    if storms_data is not None:
        axes[0].vlines(storms_data["t"],
                       storms_data["lon_lo"], storms_data["lon_hi"],
                       colors=storms_data["colors"],
                       linewidth=storm_linewidth, alpha=storm_alpha,
                       zorder=3, label='Storms')

    # for y in (-50, -60):
    # for y in (-43.1788,): #copacabana
    for y in (-53.57,): #RS
        axes[0].axhline(y, color="black", linewidth=1., linestyle="--", zorder=4)
    # --- Bottom: Latitud ---

    im_lat = axes[1].imshow(
        lat_vals, extent=extent_lat, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh, interpolation="nearest",
    )
    axes[1].set_ylabel("Latitud, deg")

    axes[1].set_xlim(0, 24)
    # axes[1].set_ylim(LAT_RANGE)
    axes[1].set_ylim(extent_lat[2], extent_lat[3])
    axes[1].set_xticks(np.arange(0, 25, 3))
    axes[1].set_xlabel("Time (UTC, hours)")
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].grid(True, which='major', linestyle=":", alpha=0.4)
    axes[1].grid(True, which='minor', linestyle=":", alpha=0.2)

    plt.colorbar(im_lat, ax=axes[1]).set_label("dTEC (TECU)")

    # Storm overlay (bottom panel: latitude span)
    if storms_data is not None:
        axes[1].vlines(storms_data["t"],
                       storms_data["lat_lo"], storms_data["lat_hi"],
                       colors=storms_data["colors"],
                       linewidth=storm_linewidth, alpha=storm_alpha,
                       zorder=3, label='Storms')
        # Optional shared colorbar for storm temperature (kept commented
        # out to preserve the original two-colorbar layout).
        # cb_s = fig.colorbar(storms_data["sm"], ax=axes,
        #                     location='right', shrink=0.6, pad=0.02)
        # cb_s.set_label("min cloud-top T (°C)")
    
    # for y in (-20, -30):
    # for y in (-22.9678,): #copacabana
    for y in (-30.23,): #RS
        axes[1].axhline(y, color="black", linewidth=1.2, linestyle="--", zorder=4)
    
    fig.text(0.9, -0.01, 'Script: gnss-tec-pipeline/visualize_keo.py\nFunction: plot_keograms_storm_mpl\nStorage: database/ionosphere/keograms_figs/', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=10, color='red', alpha=1.)
    
    if out_path is None:
        out_path = dtec_keogram_figs_dir() / f"keogram_{year}{doy:03d}_{resolution}.mpl.png"

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
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



def main_build(year, doy, resolution="low", dtec_col="dtec_10min", overwrite=False, lat_range=None, lon_range=None):
    """
    Stage 1 — aggregate all station parquets into TWO NetCDF grids
    (one for latitude, one for longitude).
    """
    # Fallback para as variáveis globais caso não sejam passadas diretamente
    if lat_range is None: lat_range = globals().get('LAT_RANGE')
    if lon_range is None: lon_range = globals().get('LON_RANGE')

    resolutions = {
        "low": 0.1,
        "high": 0.05,
        "veryhigh": 0.025,
        "ultra": 0.005
    }
    deg_per_px = resolutions.get(resolution, 0.1)

    ipp_folder      = dtec_ipp_dir(year, doy)
    keograms_folder = dtec_keogram_dir(year, doy)

    # --- Output paths ---
    lat_path = keograms_folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc"
    lon_path = keograms_folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc"

    lat_exists = lat_path.exists()
    lon_exists = lon_path.exists()

    if lat_exists and lon_exists and not overwrite:
        print(f"Both grids already exist for {year} DOY {doy:03d} ({resolution}). Skipping build.")
        return

    parquet_files = sorted(Path(ipp_folder).glob(f"*_dtecIPP_{year}{doy}.parquet"))
    if not parquet_files:
        print(f"No IPP files found in {ipp_folder}")
        return
    print(f"Found {len(parquet_files)} station files. Aggregating...")

    # ==========================================
    # EXTRAÇÃO DINÂMICA DE LIMITES (MIN/MAX)
    # ==========================================
    if lat_range is None or lon_range is None:
        import dask
        print("Limites não definidos. Extraindo lat/lon min e max dos dados via Dask...")
        # Lemos apenas as duas colunas necessárias para não sobrecarregar a memória
        df_bounds = dd.read_parquet(parquet_files, columns=["lts", "lns"])
        
        if lat_range is None:
            lts_min, lts_max = dask.compute(df_bounds["lts"].min(), df_bounds["lts"].max())
            lat_range = (float(lts_min), float(lts_max))
            print(f" -> LAT_RANGE dinâmico: {lat_range}")
            
        if lon_range is None:
            lns_min, lns_max = dask.compute(df_bounds["lns"].min(), df_bounds["lns"].max())
            lon_range = (float(lns_min), float(lns_max))
            print(f" -> LON_RANGE dinâmico: {lon_range}")
    # ==========================================

    plot_height_lat = int((lat_range[1] - lat_range[0]) / deg_per_px)
    plot_height_lon = int((lon_range[1] - lon_range[0]) / deg_per_px)

    # --- Latitude keogram ---
    if lat_exists and not overwrite:
        print(f"Lat grid exists, skipping: {lat_path.name}")
    else:
        print("Building lat keogram...")
        agg_lat = build_keogram_grid(
            parquet_files, y_col="lts", y_range=lat_range,
            plot_width=5760, plot_height=plot_height_lat, dtec_col=dtec_col
        )
        save_grid(agg_lat, lat_path)

    # --- Longitude keogram ---
    if lon_exists and not overwrite:
        print(f"Lon grid exists, skipping: {lon_path.name}")
    else:
        print("Building lon keogram...")
        agg_lon = build_keogram_grid(
            parquet_files, y_col="lns", y_range=lon_range,
            plot_width=5760, plot_height=plot_height_lon, dtec_col=dtec_col
        )
        save_grid(agg_lon, lon_path)


def main_plot(year, doy, resolution="low", trh=0.5, cmap=cc.coolwarm,
              use_matplotlib: bool = False, smooth_size: int = None):
    """
    Stage 2 — load both saved grids and export their PNGs.

    Parameters
    ----------
    use_matplotlib : bool
        If True, combines lat and lon into a single two-panel figure
        (with axes, colorbar, labels). If False, exports two separate
        raw datashader images (faster, no axes).
    smooth_size : int, optional
        Vertical moving-average window (in pixels) applied to each grid
        before plotting. None (default) disables smoothing.
    """
    keogram_folder = dtec_keogram_dir(year, doy)
    nc_lat = keogram_folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc"
    nc_lon = keogram_folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc"
    if not nc_lat.exists() or not nc_lon.exists():
        print(f"Grids not found. Run main_build() first.")
        return
    if use_matplotlib:
        mpl_cmap = "seismic_r" if cmap is cc.coolwarm else cmap
        plot_keograms_storm_mpl(nc_lat, nc_lon, trh=trh, cmap=mpl_cmap,
                                plot_storms=True, smooth_size=smooth_size)
    else:
        plot_keogram(nc_lat, trh=trh, cmap=cmap)
        plot_keogram(nc_lon, trh=trh, cmap=cmap)


# ==========================================
# 6. IDE / CLI entry point
# ==========================================
if __name__ == "__main__":

    # Default ranges per axis ----------------------------------------------------
    LAT_RANGE = (-60.01,  60.01)    # 120° span
    LON_RANGE = (-120.01, -19.99)   # 100° span (Americas)

    LAT_RANGE = (-32.01,  -16.99)    # 120° span
    LON_RANGE = (-65.01, -39.99)   # 100° span (Americas)
    
    LAT_RANGE = (-43.01,  -9.99)    # 120° span
    LON_RANGE = (-60.01, -39.99)   # 100° span (Americas)
    
    # LAT_RANGE = None
    # LON_RANGE = None
    
    # TIME_RANGE = (0, 24)

    # LAT_RANGE = (-25.0,  -20.0)    # 120° span
    # LON_RANGE = (-60, -50.0)   # 100° span (Americas)
    DTEC_COL = "dtec_15min"
    movavg_srt = int(re.findall(r'\d+', DTEC_COL)[0])

    RUN_MANUALLY = True
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")
        year, doy = 2021, 296
        
        # resolution = 'low'
        resolution = 'high'
        resolution = 'ultra'

        # Vertical smoothing window, in pixels (None = off)
        # smooth_size = None
        smooth_size = 20

        # Stage 1 — run once
        main_build(year=year, doy=doy, resolution=resolution, dtec_col=DTEC_COL)
        # # Stage 2 — pick ONE of the two below:
        # # datashader raw image (fast, no axes)
        # main_plot(year=2025, doy=316, resolution="low", trh=1.0,
        #           use_matplotlib=False)
        # matplotlib version (with axes, colorbar, title)
        main_plot(year=year, doy=doy, resolution=resolution,
                  trh=0.1501,
                  # trh=1.01,
                  smooth_size=smooth_size,
                  use_matplotlib=True)
    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        main_build(year=args.year, doy=args.doy, resolution=args.resolution)
        main_plot( year=args.year, doy=args.doy, resolution=args.resolution,
                   trh=args.trh)