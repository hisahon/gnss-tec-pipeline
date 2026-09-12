# -*- coding: utf-8 -*-
"""
Keogram EXTREME-PROFILE comparator (latitude axis).

Reads keogram .nc grids ALREADY SAVED on disk (produced by
visualize_keo_datashader.py -> main_build) and, for each one, extracts
the ABSOLUTE EXTREME value of each column: for every time step,
compares the maximum and minimum dTEC across all latitude bins of
that grid and keeps whichever has the LARGER MAGNITUDE, returned as a
positive value (sign is dropped). This results in a SINGLE line per
keogram that already captures both positive and negative peaks as one
magnitude -- useful for spotting oscillations without needing to look
at two separate curves or worry about sign.

Produces TWO figures, always for the latitude axis only, with each
date's curve vertically shifted (offset) to make comparison easier
without overlap:
  1) Full-day profile (0h-24h) for each date in DATES_WINDOWS.
  2) Profile cropped to each date's time window, in the format
     'YYYYMMDD-HHHH' or 'YYYYMMDD-HHMMHHMM' (start/end hour, or
     start/end hour+minute; e.g. '20211023-1519' = Oct 23, 2021 from
     15:00 to 19:00; or '20211023-15301900' = from 15:30 to 19:00).

This routine NEVER rebuilds grids: if a date's .nc file is not found
on disk, that date is simply skipped (with a console warning).

Optionally, a latitude range can be set via LAT_BOUNDARIES =
(lat_min, lat_max) to restrict the degree axis before extracting the
max/min per column. If LAT_BOUNDARIES is left as None, the full
latitude range available in the grid is used.

Built to run directly in Spyder: edit DATES_WINDOWS, RESOLUTION,
OFFSET_TECU and LAT_BOUNDARIES in the "if __name__" block and run the
script (F5).

@author: hisas
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Project configuration module (same folders as the original script)
from shared.config import dtec_keogram_dir, dtec_keogram_figs_dir


# ==========================================
# 1. Helpers
# ==========================================
def date_to_year_doy(date_str: str):
    """Convert 'YYYYMMDD' -> (year, day-of-year)."""
    dt = pd.to_datetime(str(date_str), format="%Y%m%d")
    return dt.year, dt.dayofyear


def format_date_label(date_str: str) -> str:
    """Format 'YYYYMMDD' as 'YYYY-MM-DD' for display in plots/legends."""
    return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"


def grid_paths(year: int, doy: int, resolution: str):
    """Build the expected lat/lon .nc paths for one day."""
    folder = dtec_keogram_dir(year, doy)
    lat_path = folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc"
    lon_path = folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc"
    return lat_path, lon_path


def parse_date_window(entry: str):
    """
    Parse the window suffix after 'YYYYMMDD-'. Accepts two formats,
    auto-detected by digit count (tolerant of an extra dash, e.g.
    '20260723-16-18'):
      - HHHH       (4 digits)  -> start hour and end hour (no minutes)
      - HHMMHHMM   (8 digits)  -> start hour+minute and end hour+minute

    Returns (date_str, start_hour, end_hour), where start_hour/end_hour
    are fractional hours (e.g. 15:30 -> 15.5).
    """
    parts = entry.split("-")
    date_str = parts[0]
    digits = "".join(ch for ch in "".join(parts[1:]) if ch.isdigit())

    if len(digits) == 4:
        start_hour = int(digits[0:2]) + 0 / 60.0
        end_hour = int(digits[2:4]) + 0 / 60.0
    elif len(digits) == 8:
        start_hour = int(digits[0:2]) + int(digits[2:4]) / 60.0
        end_hour = int(digits[4:6]) + int(digits[6:8]) / 60.0
    else:
        raise ValueError(
            f"Could not parse the time window in '{entry}' "
            f"(expected HHHH or HHMMHHMM, e.g. '1519' or '15191900')."
        )
    return date_str, start_hour, end_hour


def format_hour(frac_hour: float) -> str:
    """Format a fractional hour (e.g. 15.5) as 'HH:MM'."""
    h = int(frac_hour)
    m = int(round((frac_hour - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h:02d}:{m:02d}"


def _minor_time_formatter(x, pos):
    """Show 'HH:MM' labels only at the 20- and 40-minute minor ticks."""
    total_minutes = round(x * 60)
    minute_of_hour = total_minutes % 60
    if minute_of_hour in (20, 40):
        hour = int(total_minutes // 60)
        return f"{hour:02d}:{minute_of_hour:02d}"
    return ""


def merge_extreme(max_vals: np.ndarray, min_vals: np.ndarray) -> np.ndarray:
    """
    Combine the max and min (per column) into a single ABSOLUTE value:
    keeps whichever has the LARGER MAGNITUDE between the two, always
    returned as a positive value (sign is dropped). Handles columns
    where only one of the two exists (NaN in the other).
    """
    max_vals = np.abs(np.asarray(max_vals, dtype=float))
    min_vals = np.abs(np.asarray(min_vals, dtype=float))
    merged = np.full_like(max_vals, np.nan)

    both_valid = ~np.isnan(max_vals) & ~np.isnan(min_vals)
    merged[both_valid] = np.maximum(max_vals[both_valid], min_vals[both_valid])

    only_max = np.isnan(min_vals) & ~np.isnan(max_vals)
    merged[only_max] = max_vals[only_max]

    only_min = np.isnan(max_vals) & ~np.isnan(min_vals)
    merged[only_min] = min_vals[only_min]

    return merged


def extreme_profile(nc_path: Path, lat_boundaries: tuple = None):
    """
    Open a saved grid and collapse it into a single 1-D profile: for
    each time column, take the max and the min along the degree axis
    (lat), ignoring NaNs, and merge the two into the value of largest
    magnitude (see merge_extreme).

    Parameters
    ----------
    nc_path : Path to the saved .nc grid.
    lat_boundaries : optional (lat_min, lat_max) tuple. Restricts the
        degree axis to this range before taking the max/min. If None,
        the full latitude range available in the grid is used.

    Returns
    -------
    hours  : np.ndarray  (t_seconds / 3600)
    values : np.ndarray  (absolute extreme value per column, always >= 0)
    """
    agg = xr.open_dataarray(nc_path)
    deg_dim = [d for d in agg.dims if d != "t_seconds"][0]

    if lat_boundaries is not None:
        lat_min, lat_max = lat_boundaries
        coord_vals = agg[deg_dim].values
        mask = (coord_vals >= lat_min) & (coord_vals <= lat_max)
        agg = agg.isel({deg_dim: mask})

    max_da = agg.max(dim=deg_dim, skipna=True)
    min_da = agg.min(dim=deg_dim, skipna=True)
    hours = max_da["t_seconds"].values / 3600.0
    values = merge_extreme(max_da.values, min_da.values)
    return hours, values


def crop_window(hours: np.ndarray, values: np.ndarray, start_hour: float, end_hour: float):
    """Crop (hours, values) to the [start_hour, end_hour] interval."""
    mask = (hours >= start_hour) & (hours <= end_hour)
    return hours[mask], values[mask]


def coord_extent(nc_path: Path, boundaries: tuple = None):
    """
    Return the (min, max) of the degree-axis coordinate in a saved
    grid, optionally restricted to `boundaries` -- used to report the
    actual range that contributed to the plot.
    """
    agg = xr.open_dataarray(nc_path)
    deg_dim = [d for d in agg.dims if d != "t_seconds"][0]
    coord_vals = agg[deg_dim].values
    if boundaries is not None:
        lo, hi = boundaries
        coord_vals = coord_vals[(coord_vals >= lo) & (coord_vals <= hi)]
    if coord_vals.size == 0:
        return None
    return float(np.nanmin(coord_vals)), float(np.nanmax(coord_vals))


def compute_lat_extent(dates, resolution: str = "ultra", lat_boundaries: tuple = None):
    """
    For a list of 'YYYYMMDD' dates, read each date's latitude grid
    (when available) and combine their coordinate extents. Reflects
    `lat_boundaries` when given (i.e. what was actually used to build
    the profile).

    Returns (lat_min, lat_max); either value is None if no grid
    provided data.
    """
    lat_min = lat_max = None
    for date_str in dates:
        year, doy = date_to_year_doy(date_str)
        lat_path, _ = grid_paths(year, doy, resolution)

        if lat_path.exists():
            ext = coord_extent(lat_path, lat_boundaries)
            if ext is not None:
                lo, hi = ext
                lat_min = lo if lat_min is None else min(lat_min, lo)
                lat_max = hi if lat_max is None else max(lat_max, hi)

    return lat_min, lat_max


# ==========================================
# 2. Profile collection (latitude only)
# ==========================================
def collect_extreme_profiles(dates, resolution: str = "ultra", lat_boundaries: tuple = None):
    """
    FULL-DAY profile (0h-24h) for each date in `dates` (list of
    'YYYYMMDD'). Dates with no available grid are skipped.

    lat_boundaries : optional (lat_min, lat_max) tuple; if None, the
        full latitude range available in the grid is used.

    Returns: {date_str: (hours, values)}   -- latitude axis only
    """
    results = {}
    for date_str in dates:
        year, doy = date_to_year_doy(date_str)
        lat_path, _ = grid_paths(year, doy, resolution)

        if not lat_path.exists():
            print(f"[skip] {date_str}: grid not found -> {lat_path.name}")
            continue

        h_lat, vals_lat = extreme_profile(lat_path, lat_boundaries=lat_boundaries)
        results[date_str] = (h_lat, vals_lat)
        print(f"[ok]   {date_str}: profile extracted.")
    return results


def collect_extreme_profiles_cropped(date_windows, resolution: str = "ultra", lat_boundaries: tuple = None):
    """
    Profile CROPPED to a time window per date.

    Parameters
    ----------
    date_windows : list of 'YYYYMMDD-HHHH' or 'YYYYMMDD-HHMMHHMM' strings
    lat_boundaries : optional (lat_min, lat_max) tuple; if None, the
        full latitude range available in the grid is used.

    Returns: {date_str: {"lat": (hours, values), "window": (start_hour, end_hour)}}
    """
    results = {}
    for entry in date_windows:
        date_str, start_hour, end_hour = parse_date_window(entry)
        year, doy = date_to_year_doy(date_str)
        lat_path, _ = grid_paths(year, doy, resolution)

        if not lat_path.exists():
            print(f"[skip] {entry}: grid not found -> {lat_path.name}")
            continue

        h_lat, vals_lat = extreme_profile(lat_path, lat_boundaries=lat_boundaries)
        h_lat, vals_lat = crop_window(h_lat, vals_lat, start_hour, end_hour)

        results[date_str] = {
            "lat": (h_lat, vals_lat),
            "window": (start_hour, end_hour),
        }
        print(f"[ok]   {entry}: profile cropped ({format_hour(start_hour)}-{format_hour(end_hour)}).")
    return results


# ==========================================
# 3. Plot
# ==========================================
def _add_extent_annotation(ax, lat_min, lat_max):
    """Print the covered latitude range in the top-left corner of the axes."""
    if lat_min is None:
        return
    ax.text(0.01, 0.98, f"Lat: {lat_min:.2f}\u00b0 to {lat_max:.2f}\u00b0",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))


def _add_footer(fig, storage_subfolder: str):
    fig.text(0.9, -0.06,
              'Script: gnss-tec-pipeline/visualize_max_profiles.py\n'
              'Function: plot_extreme_profiles\n'
              f'Storage: database/ionosphere/{storage_subfolder}/',
              horizontalalignment='right', verticalalignment='bottom',
              fontsize=7, color='red', alpha=0.5)


def plot_extreme_profiles(results: dict, resolution: str = "ultra",
                           out_path: Path = None, offset_tecu: float = 1.0,
                           lat_boundaries: tuple = None):
    """
    Figure 1: full-day profile (0h-24h), latitude axis only.
    One line per date, vertically shifted by `offset_tecu` TECU
    relative to the previous one (the first date has no offset).
    """
    if not results:
        print("Nothing to plot (no date with an available grid).")
        return

    dates = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dates), 1)))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for i, (date_str, color) in enumerate(zip(dates, colors)):
        offset = i * offset_tecu
        hours, values = results[date_str]
        ax.plot(hours, values + offset, color=color, linewidth=1.2,
                 label=format_date_label(date_str))
        ax.fill_between(hours, offset, values + offset, color=color, alpha=0.25, linewidth=0)
        ax.axhline(offset, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    lat_tag = f" — lat: {lat_boundaries[0]:g}° to {lat_boundaries[1]:g}°" if lat_boundaries else ""
    ax.set_ylabel(f"|dTEC| max (TECU) — offset {offset_tecu:g} TECU/date\n(latitude axis)")
    ax.set_xlabel("Time (UTC, hours)")
    ax.set_title(f"Absolute dTEC extreme profile per column (latitude axis) — resolution: {resolution}{lat_tag}")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, len(dates) * offset_tecu+1)

    ax.set_xticks(np.arange(0, 25, 3))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which="major", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend(ncol=min(len(dates), 6), fontsize=8, title="Date")

    lat_min, lat_max = compute_lat_extent(dates, resolution=resolution,
                                           lat_boundaries=lat_boundaries)
    _add_extent_annotation(ax, lat_min, lat_max)

    _add_footer(fig, "keograms_dtecs")

    if out_path is None:
        tag = "_".join(dates)
        out_dir = dtec_keogram_figs_dir().parent / "keograms_dtecs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"keogram_extremeprofile_lat_{resolution}_{tag}.png"

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.show()


def plot_extreme_profiles_cropped(results: dict, resolution: str = "ultra",
                                   out_path: Path = None, offset_tecu: float = 1.0,
                                   lat_boundaries: tuple = None):
    """
    Figure 2: profile cropped to each date's time window, latitude
    axis only. One line per date, vertically shifted by `offset_tecu`
    TECU relative to the previous one, labeled with the time window
    used, and shaded between the curve and its baseline.
    """
    if not results:
        print("Nothing to plot (no date with an available grid/window).")
        return

    dates = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dates), 1)))

    starts = [results[d]["window"][0] for d in dates]
    ends = [results[d]["window"][1] for d in dates]
    xmin, xmax = min(starts), max(ends)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    for i, (date_str, color) in enumerate(zip(dates, colors)):
        offset = i * offset_tecu
        start_hour, end_hour = results[date_str]["window"]
        label = f"{format_date_label(date_str)} ({format_hour(start_hour)}-{format_hour(end_hour)})"
        hours, values = results[date_str]["lat"]
        ax.plot(hours, values + offset, color=color, linewidth=1.2, label=label)
        ax.fill_between(hours, offset, values + offset, color=color, alpha=0.25, linewidth=0)
        ax.axhline(offset, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    lat_tag = f" — lat: {lat_boundaries[0]:g}° to {lat_boundaries[1]:g}°" if lat_boundaries else ""
    ax.set_ylabel(f"|dTEC| max (TECU) — offset {offset_tecu:g} TECU/date\n(latitude axis)")
    ax.set_xlabel("Time (UTC, hours)")
    ax.set_title(f"Absolute dTEC extreme profile per column (time-cropped, latitude axis) — resolution: {resolution}{lat_tag}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, len(dates) * offset_tecu+1)
    step = 1 if (xmax - xmin) <= 6 else (2 if (xmax - xmin) <= 12 else 3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 60))  # 10-minute minor ticks
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(_minor_time_formatter))
    ax.tick_params(axis="x", which="minor", labelsize=7)
    ax.grid(True, which="major", axis="x", linestyle="-", linewidth=0.9, alpha=0.7)
    ax.grid(True, which="major", axis="y", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.15)
    ax.legend(ncol=min(len(dates), 4), fontsize=8, title="Date (window)")

    lat_min, lat_max = compute_lat_extent(dates, resolution=resolution,
                                           lat_boundaries=lat_boundaries)
    _add_extent_annotation(ax, lat_min, lat_max)

    _add_footer(fig, "keograms_dtecs")

    if out_path is None:
        tag = "_".join(dates)
        out_dir = dtec_keogram_figs_dir().parent / "keograms_dtecs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"keogram_extremeprofile_lat_cropped_{resolution}_{tag}.png"

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.show()


# ==========================================
# 4. Entry point (Spyder / IDE)
# ==========================================
if __name__ == "__main__":

    # Dates + time window (used by both figures: full day and cropped)
    # format: 'YYYYMMDD-HHHH' (hour) or 'YYYYMMDD-HHMMHHMM' (hour+minute)
    DATES_WINDOWS = [
        "20211023-15301800",
        "20260722-15301800",
        "20260723-15301800",
    ]

    # Must match the resolution used when the grids were built
    RESOLUTION = "ultra"

    # Vertical offset between curves, in TECU
    OFFSET_TECU = 2.0

    # Optional latitude range restriction: (lat_min, lat_max).
    # Leave as None to use the full latitude range available in the grid.
    LAT_BOUNDARIES = None
    # Example: LAT_BOUNDARIES = (-30.0, -10.0)

    # Figure 1: full day (uses only the date part of each entry, ignores the window)
    DATES = list(dict.fromkeys(parse_date_window(entry)[0] for entry in DATES_WINDOWS))
    results_full = collect_extreme_profiles(DATES, resolution=RESOLUTION, lat_boundaries=LAT_BOUNDARIES)
    plot_extreme_profiles(results_full, resolution=RESOLUTION, offset_tecu=OFFSET_TECU,
                           lat_boundaries=LAT_BOUNDARIES)

    # Figure 2: time-cropped
    results_cropped = collect_extreme_profiles_cropped(DATES_WINDOWS, resolution=RESOLUTION, lat_boundaries=LAT_BOUNDARIES)
    plot_extreme_profiles_cropped(results_cropped, resolution=RESOLUTION, offset_tecu=OFFSET_TECU,
                                   lat_boundaries=LAT_BOUNDARIES)