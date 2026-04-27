# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
Created by Rogério Hisashi Honda on Fri Nov  7 10:44:04 2025
hisashi.brase@gmail.com

----------------------------------------------------------------------------
"""

# -*- coding: utf-8 -*-
# =====================================================================
# RASTHER_DTEC - dTEC processing and visualization pipeline
# =====================================================================
# This script reads .parquet files containing dTEC (variation of Total
# Electron Content) time series per GNSS station, computes the
# Ionospheric Pierce Point (IPP - latitude/longitude on a thin-shell
# ionospheric layer) for each station/PRN pair, and generates
# diagnostic figures of Time x Latitude and Time x Longitude.
#
# Version history (kept here for traceability):
# - v1..v3: pcolormesh via Datashader (kept as a utility in
#   `trajectories_to_pcolormesh`).
# - v4    : pcolormesh replaced by scatter (matplotlib) with
#   overlay of convective storms (vlines colored by minimum cloud
#   top temperature) - function `plot_ipp_trajectory`.
# - v5 (this version):
#     * comments added in every routine to make their purpose
#       easy to identify;
#     * existing routines preserved (Datashader + matplotlib with
#       storms overlay);
#     * new routine `plot_lat_lon_scatter_mpl` added: for EACH
#       .parquet file (each PRN), produces a simple figure with
#       two stacked panels (lat x time and lon x time) using
#       ONLY a matplotlib scatter, with no overlays;
#     * `run_pipeline_and_plot` was extended with the
#       `make_simple_scatter` flag to generate this new figure
#       inside the loop.
# =====================================================================

from __future__ import annotations

# --- Core scientific libraries --------------------------------------
import numpy as np                # Vector/numeric ops (ECEF, masks, etc.)
import pandas as pd               # Tabular handling of parquets and time indices
from pathlib import Path          # OS-independent path manipulation


# --- Datashader (kept from previous version, used by
#     `trajectories_to_pcolormesh` for fast rasterization of
#     dense trajectories onto a regular grid) ------------------------
import datashader as ds                       # Canvas / rasterizer
# import datashader.transfer_functions as tf  # not used, but useful for extensions
import datashader.reductions as rd            # count / sum / mean per cell

from shared.geometry import ipp_calc
from shared.figures_rasther import plot_ipp_trajectory, plot_lat_lon_scatter_mpl

from shared.config import DB_ROOT, station_positions_dir, dtec_pc_dir, storms_SouthAmerica

# ---------------------------------------------------------------------
# FILE UTILITIES
# ---------------------------------------------------------------------
def read_files(folder: Path | str, extension: str) -> list[Path]:
    """
    Recursively scans `folder` and returns a sorted list of files
    whose extension matches `extension` (case-insensitive).

    Parameters
    ----------
    folder : Path | str
        Root directory of the search.
    extension : str
        Desired extension, with or without the dot (e.g. 'parquet'
        or '.parquet').

    Returns
    -------
    list[Path]
        Alphabetically sorted absolute/relative paths.
    """
    folder = Path(folder)
    # Normalize extension: ensure leading dot and lowercase
    ext = extension if extension.startswith('.') else f'.{extension}'
    ext = ext.lower()
    # rglob('*') performs a recursive search; keep only files with the target extension
    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == ext]
    files.sort()  # deterministic ordering for reproducibility
    return files


# ---------------------------------------------------------------------
# TIME NORMALIZATION
# ---------------------------------------------------------------------
def infer_freq_from_index(dt_index: pd.DatetimeIndex) -> str | None:
    """
    Tries to infer the DOMINANT frequency (mode of the deltas in
    seconds) of a DatetimeIndex and returns a pandas-compatible
    string (e.g. '15S').

    Useful when the nominal frequency was not declared and there
    may be small losses/duplicates in the index. Using the mode
    (instead of the mean) is robust against outliers (gaps).

    Returns None if there are fewer than 2 valid samples.
    """
    if len(dt_index) < 2:
        return None
    # Consecutive differences in seconds
    deltas = (dt_index[1:] - dt_index[:-1]).to_series().dt.total_seconds()
    # Keep only positive deltas (avoid zeros from duplicates) and round to int
    vals = deltas[deltas > 0].round().astype('Int64').dropna()
    if vals.empty:
        return None
    mode = int(vals.mode().iloc[0])  # most common frequency
    if mode <= 0:
        return None
    return f"{mode}S"


def normalize_time_grid(
    df: pd.DataFrame,
    time_col: str = "horario",
    freq: str | None = "15s",
    pos_cols=("x_km","y_km","z_km")
) -> pd.DataFrame:
    """
    Normalizes a DataFrame onto a continuous time grid.

    STRICT sequence applied (in order):
      1) Convert 'horario' to datetime and sort chronologically.
      2) Drop duplicates (keep the first occurrence).
      3) Infer/use the frequency (default '15s').
      4) Reindex onto a continuous grid (gaps become NaN).
      5) (Optional/disabled) would interpolate ONLY the PRN
         position columns - kept commented out to preserve the
         current behavior (no interpolation).
      6) Return the DataFrame with a continuous time index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with the time column `time_col`.
    time_col : str
        Name of the time column.
    freq : str | None
        Target frequency (e.g. '15s'). If None, will be inferred.
    pos_cols : tuple
        Satellite ECEF position columns (used if interpolation
        is re-enabled).
    """
    df = df.copy()  # never modifies the input DataFrame

    # 1) datetime + ordering -----------------------------------------
    # Explicit conversion is commented out because parquets already
    # come with datetime64. Re-enable if the source differs.
    # if not np.issubdtype(df[time_col].dtype, np.datetime64):
    #     df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    # 2) deduplication -----------------------------------------------
    # `keep="first"` is a conservative choice; could be replaced by
    # an aggregation (mean/last) if needed in the future.
    df = df.drop_duplicates(subset=[time_col], keep="first").set_index(time_col)

    # 3) frequency ---------------------------------------------------
    # If the user does not provide it, try to infer; if that also
    # fails, fall back to a default of 15 seconds.
    if freq is None:
        freq = infer_freq_from_index(df.index)
        if freq is None:
            freq = "15s"

    # 4) continuous grid --------------------------------------------
    # Build a regular time index from start to end and reindex.
    # Gaps become NaN (important to preserve the topology of the
    # passes - do not forward-fill).
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)

    # 5) OPTIONAL interpolation of PRN positions --------------------
    # Kept commented out to reproduce current behavior.
    # cols_exist = [c for c in pos_cols if c in df.columns]
    # if cols_exist:
    #     df[cols_exist] = df[cols_exist].interpolate(method="time", limit_direction="both")

    # 6) return ------------------------------------------------------
    return df




# ---------------------------------------------------------------------
# TRAJECTORY RASTERIZATION FOR PCOLORMESH (Datashader)
# ---------------------------------------------------------------------
# NOTE: the two functions below are KEPT from the previous
# implementation (v3). They use Datashader to reduce dense
# trajectories into a Z matrix (height x width) that can be drawn
# with `ax.pcolormesh`. They are not called by the current pipeline
# (which uses scatter), but remain available for diagnostics or for
# very large datasets (>1e6 points) where a scatter is not viable.
# ---------------------------------------------------------------------
def _ensure_2d_time_matrix(t_like, n_cols: int) -> np.ndarray:
    """
    Ensures the time axis is a 2D matrix (N x n_cols) compatible
    with Y_df.

    - If the input is 1D (a vector of timestamps), tiles it
      horizontally one column per trajectory.
    - If it is already 2D, validates the number of columns.
    """
    # Accepts Series/Index/DatetimeIndex without extra conversion
    if isinstance(t_like, (pd.Series, pd.DatetimeIndex, pd.Index)):
        t_arr = t_like.to_numpy()
    else:
        t_arr = np.asarray(t_like)

    if t_arr.ndim == 1:
        # Tile the time vector into n_cols columns (manual broadcast)
        return np.tile(t_arr[:, None], (1, n_cols))
    elif t_arr.ndim == 2:
        if t_arr.shape[1] != n_cols:
            raise ValueError("When X_df is 2D, the number of columns must equal Y_df.")
        return t_arr
    else:
        raise ValueError("Invalid X_df: must be 1D (time) or 2D (time matrix).")


def trajectories_to_pcolormesh(
    X_df,                     # 1D (time) or 2D (replicated time) matching Y_df
    Y_df: pd.DataFrame,       # 2D, rows = timestamps, cols = trajectories
    W_df: pd.DataFrame | None = None,  # weights/attribute for sum/mean (same shape as Y_df)
    plot_width: int = 1600,
    plot_height: int = 1600,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
    agg: str = "count"        # "count", "sum", "mean"
):
    """
    Rasterizes a set of (x,y) trajectories into a regular grid ready
    for `ax.pcolormesh`, using Datashader.

    Supported aggregation modes:
      - "count": number of points per cell (density);
      - "sum"  : sum of weight `w` per cell;
      - "mean" : mean of weight `w` per cell.
    For "sum"/"mean", `W_df` is REQUIRED and must have the same
    shape as `Y_df`.

    Returns
    -------
    x_edges, y_edges, Z, cbar_label
        Pixel edges (compatible with pcolormesh), aggregated value
        matrix and a suggested colorbar label.
    """
    # Basic data-shape validation ------------------------------------
    if not isinstance(Y_df, pd.DataFrame):
        raise TypeError("Y_df must be a pandas.DataFrame.")

    N, M = Y_df.shape
    X_mat = _ensure_2d_time_matrix(X_df, M)

    if X_mat.shape != Y_df.shape:
        raise ValueError("X (time) and Y (trajectories) must have the same 2D shape after expansion.")

    # Flatten into the long format expected by Datashader -----------
    x_time = X_mat.ravel()                         # dtype datetime64[ns]
    y_vals = Y_df.to_numpy().ravel()

    df_long = pd.DataFrame({"x": x_time, "y": y_vals})

    # For sum/mean, attach the weight column and drop NaNs ----------
    if agg in ("sum", "mean"):
        if W_df is None:
            raise ValueError('For agg="sum" or "mean", W_df must be provided with the same shape as Y_df.')
        if W_df.shape != Y_df.shape:
            raise ValueError("W_df must have the same shape as Y_df.")
        df_long["w"] = W_df.to_numpy().ravel()
        df_long = df_long.dropna(subset=["w"])

    df_long = df_long.dropna(subset=["x", "y"])

    # Ranges: if not provided, use observed min/max ------------------
    if x_range is None:
        x_range = (df_long["x"].min(), df_long["x"].max())
    else:
        x_range = (pd.to_datetime(x_range[0]), pd.to_datetime(x_range[1]))
    if y_range is None:
        y_range = (df_long["y"].min(), df_long["y"].max())

    # Datashader canvas ---------------------------------------------
    cvs = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=x_range,
        y_range=y_range
    )

    # Aggregator selection ------------------------------------------
    if agg == "count":
        agg_da = cvs.points(df_long, "x", "y", rd.count())
        cbar_label = "Count per cell"
    elif agg == "sum":
        agg_da = cvs.points(df_long, "x", "y", rd.sum("w"))
        cbar_label = "Sum per cell (w)"
    elif agg == "mean":
        agg_da = cvs.points(df_long, "x", "y", rd.mean("w"))
        cbar_label = "Mean per cell (w)"
    else:
        raise ValueError('agg must be "count", "sum" or "mean".')

    # Z matrix (plot_height x plot_width)
    Z = np.asarray(agg_da)

    # Pixel edges - `pcolormesh` requires edges (not centers) -------
    # For time (datetime), Matplotlib accepts datetime64 arrays.
    # x_edges = pd.date_range(x_range[0], x_range[1], periods=plot_width + 1).to_numpy()
    x_edges = np.linspace(x_range[0], x_range[1], plot_width + 1)
    y_edges = np.linspace(y_range[0], y_range[1], plot_height + 1)

    return x_edges, y_edges, Z, cbar_label


# ---------------------------------------------------------------------
# STATION POSITIONS LOADING
# ---------------------------------------------------------------------
def load_and_filter_stations(station_file: Path | str) -> pd.DataFrame:
    """
    Reads the GNSS station positions parquet and applies a fixed
    geographic clipping (southeast/south Brazil - South America).

    The returned DataFrame is indexed by `name` (station code) and
    contains at least the columns: x, y, z, ln_deg, lt_deg, h.
    """
    # Read with fastparquet (engine chosen for compatibility with
    # the dumps produced by the upstream preprocessing pipeline)
    df_sp = pd.read_parquet(station_file, engine="fastparquet").set_index('name')

    # Geographic clipping - South America/Brazil:
    #   longitude in [-60, -45)
    #   latitude  in [-38, -20)
    stations_slice = df_sp[
        (df_sp['ln_deg'] >= -60) & (df_sp['ln_deg'] < -45) &
        (df_sp['lt_deg'] >= -38) & (df_sp['lt_deg'] < -20)
    ]
    return stations_slice


# ---------------------------------------------------------------------
# PRN FILE PROCESSING (per-station IPP computation)
# ---------------------------------------------------------------------
def process_prn_file(
    file_path: Path,
    stations_slice: pd.DataFrame,
    time_freq: str,
    shell_height_km: float
    ):
    """
    Computation pipeline for ONE .parquet file (corresponding to a
    specific PRN, on a specific day):

      1) Read the parquet (`fastparquet`).
      2) Clip to the day window (UTC) extracted from the file name.
      3) Normalize the time grid (15s by default).
      4) For each station inside the geographic mask:
            - get the station ECEF + lon/lat coordinates;
            - compute IPP (lat, lon) and geometric elevation (El) at
              the receiver for all timestamps;
            - accumulate into dictionaries.
      5) Stack everything into side-by-side DataFrames (rows = time,
         cols = stations) and return together with the original dTEC.

    Output
    ------
    dict | None
        None if there is no valid data. Otherwise, a dictionary
        with: prn, df_lt, df_ln, df_el, df_dtec, time_index,
        date_string, t_start, t_end.
    """
    # Identify the PRN from the file name
    prn = file_path.stem.split('_')[-1]

    # --- 1. Time window from the file name --------------------------
    # Expected pattern: <something>_YYYY-DDD_<...>.parquet, e.g. 2021-296
    date_str = file_path.stem.split('_')[1]
    t_start = pd.to_datetime(date_str, format='%Y-%j', utc=True)
    t_end = t_start + pd.Timedelta(days=1)

    # --- 1.1. Read the parquet --------------------------------------
    df = pd.read_parquet(file_path, engine="fastparquet")

    # Keep only observations inside the target day. (Kept as a 24h
    # window; could become an argument for finer slicing.)
    df = df[(df['horario'] >= t_start) & (df['horario'] < t_end)]

    if df.empty:
        return None

    # --- 2. Time normalization (continuous 15s grid) ----------------
    df = normalize_time_grid(df, time_col="horario", freq=time_freq, pos_cols=("x_km","y_km","z_km"))

    # --- 3. PRN ECEF coordinates over time --------------------------
    prn_coordinates = (
        df['x_km'].to_numpy(),
        df['y_km'].to_numpy(),
        df['z_km'].to_numpy()
    )

    stations_in_region = stations_slice.index.to_numpy()

    # Accumulators: dictionaries station -> time series
    lts, lns, els = {}, {}, {}
    valid_stations = []

    # --- 4. Main loop: per-station IPP ------------------------------
    for st in stations_in_region:
        # The station only matters if there is a dTEC column for it in this file
        if st not in df.columns:
            continue

        # Station coordinates: [x, y, z, lon_deg, lat_deg, h]
        station_coords = stations_slice.loc[st].values
        # Sanity check: coordinates all near zero indicate an invalid/
        # placeholder record - skip
        if np.any(np.abs(station_coords) < 0.001):
            continue

        # IPP computation (thin-shell) for all timestamps
        lt, ln, el = ipp_calc(prn_coordinates, station_coords, H_km=shell_height_km)

        lts[st] = lt
        lns[st] = ln
        els[st] = el
        valid_stations.append(st)

    if not lts:
        return None

    # --- 5. Stack into DataFrames -----------------------------------
    idx_time = df.index
    df_lt = pd.DataFrame(lts, index=idx_time)
    df_ln = pd.DataFrame(lns, index=idx_time)
    df_el = pd.DataFrame(els, index=idx_time)
    # `df_dtec`: take only the dTEC columns of the valid stations,
    # aligned on the SAME time index as df_lt/df_ln/df_el
    df_dtec = df[valid_stations]

    return {
        "prn": prn,
        "df_lt": df_lt,
        "df_ln": df_ln,
        "df_el": df_el,
        "df_dtec": df_dtec,
        "time_index": idx_time,
        "date_string": date_str,
        "t_start": t_start,
        "t_end": t_end
        }




# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def run_pipeline_and_plot(
    folder_base: Path | str,
    station_positions_file: Path | str,
    shell_height_km: float = 300.0,
    parquet_extension: str = "parquet",
    time_freq: str | None = "15s",
    cmap: str = "turbo",
    trh: float = 0.5,
    # --- v5: controls for the two figures -------------------------
    make_storm_figure: bool = True,        # v4 figure (with storms)
    make_simple_scatter: bool = True,      # v5 figure (simple scatter)
    storms_path: Path | str | None = None  # storms parquet (CPTEC c13)
    ):
    """
    Main pipeline:
      1) List all .parquet files inside `folder_base`.
      2) Load station positions filtered by the geographic mask.
      3) Load the storm catalog (only if the "rich" figure is enabled).
      4) For each file:
            - compute IPPs (`process_prn_file`);
            - if `make_storm_figure`: produce the v4 figure with storms;
            - if `make_simple_scatter`: produce the v5 figure (simple
              scatter with only lat/lon vs time in matplotlib).

    Relevant parameters added in v5
    -------------------------------
    make_storm_figure : bool
        Toggles `plot_ipp_trajectory` (figure with storms overlay).
        Default True to preserve previous behavior.
    make_simple_scatter : bool
        Toggles the NEW `plot_lat_lon_scatter_mpl` (simple
        matplotlib-only scatter figure).
    storms_path : Path | str | None
        Path of the storms parquet. If None, falls back to the
        legacy path (local Dropbox). Only read when
        `make_storm_figure` is active.
    """

    # --- 1. Files to process ----------------------------------------
    files = read_files(folder_base, parquet_extension)
    if not files:
        raise FileNotFoundError(f"No files found in {folder_base}")

    # --- 2. Stations ------------------------------------------------
    stations_slice = load_and_filter_stations(station_positions_file)

    # --- 3. Storms (only if needed) ---------------------------------
    df_storms = None
    if make_storm_figure:
        # Default path (kept for compatibility); can be replaced via
        # the `storms_path` parameter.
        if storms_path is None:
            storms_path = storms_SouthAmerica()
        df_storms = pd.read_parquet(storms_path, engine="fastparquet")
        # EQUIVALENT radius in DEGREES, derived from the area (km^2):
        # assumes a circular storm and converts using ~110 km per degree.
        df_storms['radius_deg'] = np.sqrt(df_storms['area_km2']/np.pi)/110

    # --- 4. Loop over files -----------------------------------------
    for file in files:
        # 4.1 - Compute per-station IPPs for this PRN
        data_pack = process_prn_file(
            file,
            stations_slice,
            time_freq,
            shell_height_km
            )

        if data_pack is None:
            continue

        # 4.2 - v5 figure (NEW): matplotlib-only simple scatter.
        # Does not depend on storms, so it can run first.
        if make_simple_scatter:
            plot_lat_lon_scatter_mpl(
                data_pack,
                cmap=cmap,
                trh=trh,
            )

        # 4.3 - v4 figure (with storms), if enabled
        if make_storm_figure:
            # Clip storms to the day (file time interval)
            t_start = data_pack["t_start"].tz_localize(None)
            t_end = data_pack["t_end"].tz_localize(None)

            df_stormsS = df_storms[
                # (Optional geographic filters can go here)
                # (df_storms['centroid_lon'] >= -60) & (df_storms['centroid_lon'] < -45) &
                # (df_storms['centroid_lat'] >= -38) & (df_storms['centroid_lat'] < -20) &
                (df_storms['datetime'] >= pd.to_datetime(t_start)) &
                (df_storms['datetime'] < pd.to_datetime(t_end))
            ]

            data_pack['df_storms'] = df_stormsS

            plot_ipp_trajectory(data_pack, cmap=cmap, trh=trh)


# ---------------------------------------------------------------------
# USAGE EXAMPLE (UPDATE THE PATHS BEFORE RUNNING)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example paths. REPLACE with your own.
    # Both Linux/Windows assignments are kept - the last one wins.
    # ------------------------------------------------------------------
    # BASE_DB = Path("/home/hisashi/database/gnss/")
    # BASE_DB = Path("D:/database")
    
    year, doy = 2021, 296
    # Parquet with station positions (ECEF + lat/lon/h)
    STATION_POSITIONS_PARQUET = station_positions_dir() / "SP_2021-296.parquet"
    FOLDER_PARQUETS = dtec_pc_dir(year, doy) 

    # ------------------------------------------------------------------
    # Pipeline call:
    #   - make_simple_scatter=True  -> generates the NEW (v5) figure
    #     with a pure matplotlib scatter (lat vs time and lon vs time).
    #   - make_storm_figure=True    -> also generates the v4 figure
    #     with storms overlay (set to False if you only want the new
    #     figure or if you do not have the storms parquet yet).
    # ------------------------------------------------------------------
    run_pipeline_and_plot(
        folder_base=FOLDER_PARQUETS,
        station_positions_file=STATION_POSITIONS_PARQUET,
        shell_height_km=300,
        parquet_extension="parquet",
        time_freq="15s",
        cmap="turbo",
        # cmap="seismic",
        trh=.5,
        make_simple_scatter=True,   # enable the new (v5) figure
        make_storm_figure=True,     # enable the v4 figure (with storms)
        # storms_path=Path('C:/Users/hisas/Dropbox/python/cptec/merged/c13_2018-2025_SouthAmerica.parquet'),
    )

    # ALTERNATIVE USAGE EXAMPLES:
    #
    # 1) Only the new figure (simple scatter), skipping the storms:
    # run_pipeline_and_plot(
    #     folder_base=FOLDER_PARQUETS,
    #     station_positions_file=STATION_POSITIONS_PARQUET,
    #     make_simple_scatter=True,
    #     make_storm_figure=False,
    # )
    #
    # 2) Only the v4 figure (with storms):
    # run_pipeline_and_plot(
    #     folder_base=FOLDER_PARQUETS,
    #     station_positions_file=STATION_POSITIONS_PARQUET,
    #     make_simple_scatter=False,
    #     make_storm_figure=True,
    # )