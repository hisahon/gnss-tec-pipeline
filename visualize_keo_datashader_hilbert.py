# -*- coding: utf-8 -*-
"""
Keogram HILBERT-ENVELOPE viewer (longitude on top, latitude on bottom).

Reads keogram .nc grids ALREADY SAVED on disk (produced by
visualize_keo_datashader.py -> main_build). Each grid is (degree x time):
the ROWS are the lat (or lon) bins and the COLUMNS are the time steps.

DTEC_THRESHOLD only sets the upper limit (vmax) of the color scale;
the grid values are NOT modified by it.

The processing direction is chosen with PROCESS_AXIS:
  "row"    -> each ROW (one lat/lon bin) is a signal ALONG TIME;
              scales are periods in minutes (PERIOD_*_MIN, MAX_GAP_MIN);
  "column" -> each COLUMN (one time step) is a signal ALONG LAT/LON;
              scales are wavelengths in degrees (WAVELENGTH_*_DEG,
              MAX_GAP_DEG).
For every signal:
  1) short gaps (<= MAX_GAP) are linearly interpolated; longer gaps
     are kept as gaps;
  2) signals with too little data (< MIN_VALID_FRACTION) are
     discarded;
  3) the mean is NOT removed and NO filter is applied; remaining gaps
     are filled with 0 (the neutral dTEC value) only so the transform
     can run;
  4) the pure envelope |hilbert(x)| is computed along the same axis
     (with optional reflective padding at the edges, EDGE_PAD_*);
  5) bins that were gaps are masked back to NaN.
The result is a new grid with the same shape as the keogram, where each
bin holds the Hilbert envelope of the dTEC signal.

All processing is done on the FULL DAY before cropping to the time
window.

Processes ONE date per call (DATE_WINDOW) and produces ONE figure with
two panels sharing the time axis:
  top    -> envelope of the LONGITUDE keogram
  bottom -> envelope of the LATITUDE keogram
DATE_WINDOW may be 'YYYYMMDD' (full day), 'YYYYMMDD-HHHH' or
'YYYYMMDD-HHMMHHMM' (time window, e.g. '20211023-15301800').

This routine NEVER rebuilds keogram grids: if a date's .nc file is not
found on disk, that panel is left empty (with a console warning). If
SAVE_GRIDS is True, the envelope grids are written next to the original
ones as keogram_{lat|lon}_{year}_{doy}_{resolution}_hilbert_{row|column}.nc.

Built to run directly in Spyder: edit the parameters in the
"if __name__" block and run the script (F5).

@author: hisas
"""

import inspect
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
mpl.use("Agg")                    # non-interactive backend: save figures only
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import hilbert

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
    """Format 'YYYYMMDD' as 'YYYY-MM-DD' for display in plots."""
    return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"


def grid_paths(year: int, doy: int, resolution: str):
    """Build the expected lat/lon .nc paths for one day."""
    folder = dtec_keogram_dir(year, doy)
    lat_path = folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc"
    lon_path = folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc"
    return lat_path, lon_path


def parse_date_window(entry: str):
    """
    Parse 'YYYYMMDD', 'YYYYMMDD-HHHH' or 'YYYYMMDD-HHMMHHMM'.
    A bare 'YYYYMMDD' means the full day (0h-24h).

    Returns (date_str, start_hour, end_hour) with fractional hours.
    """
    parts = entry.split("-")
    date_str = parts[0]
    digits = "".join(ch for ch in "".join(parts[1:]) if ch.isdigit())

    if len(digits) == 0:
        return date_str, 0.0, 24.0
    if len(digits) == 4:
        return date_str, float(int(digits[0:2])), float(int(digits[2:4]))
    if len(digits) == 8:
        start_hour = int(digits[0:2]) + int(digits[2:4]) / 60.0
        end_hour = int(digits[4:6]) + int(digits[6:8]) / 60.0
        return date_str, start_hour, end_hour
    raise ValueError(
        f"Could not parse the time window in '{entry}' "
        f"(expected YYYYMMDD, YYYYMMDD-HHHH or YYYYMMDD-HHMMHHMM)."
    )


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


def next_numbered_path(path: Path) -> Path:
    """
    Return '<stem>_000X<suffix>' with X = (highest existing number for
    that stem in the same folder) + 1, so figures are never overwritten.
    E.g. keogram_..._0001.png, keogram_..._0002.png, ...
    """
    path = Path(path)
    pattern = re.compile(rf"^{re.escape(path.stem)}_(\d{{4,}}){re.escape(path.suffix)}$")
    numbers = [int(m.group(1)) for f in path.parent.glob(f"{path.stem}_*{path.suffix}")
               if (m := pattern.match(f.name))]
    n = max(numbers, default=0) + 1
    return path.with_name(f"{path.stem}_{n:04d}{path.suffix}")


def add_provenance_footer(fig, storage_path, x=0.99, y=0.0, **kwargs):
    """
    Adiciona rodapé com Script / Function / Storage no canto inferior
    direito da figura, descobrindo automaticamente (via inspect) o arquivo
    e a função que chamaram esta rotina — evita rodapés dessincronizados
    ao copiar/colar entre scripts.

    O texto é ancorado pelo TOPO logo abaixo da borda inferior da figura
    (y=0, va='top'); como o savefig usa bbox_inches="tight", ele entra na
    imagem salva sem sobrepor o rótulo do eixo x.
    """
    caller = inspect.stack()[1]
    try:
        script_name = os.path.relpath(caller.filename)
    except ValueError:
        # Windows: relpath falha se o arquivo estiver em outro drive
        script_name = caller.filename
    function_name = caller.function
    storage_text = str(storage_path)

    text = (
        f"Script: {script_name}\n"
        f"Function: {function_name}\n"
        f"Storage: {storage_text}"
    )

    defaults = dict(
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=7,
        color='black',
        alpha=0.7,
    )
    defaults.update(kwargs)

    fig.text(x, y, text, **defaults)


# ==========================================
# 2. Envelope along time (per row)
# ==========================================
def fill_short_gaps(row: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Linearly interpolate (in time) NaN runs of length <= max_gap
    samples that are bounded by valid data on both sides. Longer gaps
    and gaps touching the start/end of the day stay NaN.
    """
    row = row.copy()
    isnan = np.isnan(row)
    if not isnan.any() or isnan.all() or max_gap <= 0:
        return row

    # Start/end indices of each NaN run
    edges = np.diff(np.concatenate(([0], isnan.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]          # exclusive

    idx = np.arange(row.size)
    valid_idx = idx[~isnan]
    for a, b in zip(starts, ends):
        if a == 0 or b == row.size or (b - a) > max_gap:
            continue
        row[a:b] = np.interp(idx[a:b], valid_idx, row[valid_idx])
    return row


def envelope_along_last_axis(data: np.ndarray, step: float,
                             pad_length: float = 60.0, max_gap: float = 5.0,
                             min_valid_fraction: float = 0.1):
    """
    Pure Hilbert envelope |hilbert(x)| along the LAST axis of a 2-D
    array: each row of `data` is one 1-D signal. No filtering and no
    mean removal.

    step       : sampling interval along that axis (seconds or degrees).
    pad_length : length of the reflective padding added at each edge
                 before the transform (same unit as `step`; 0 disables).
    max_gap    : longest gap (same unit as `step`) to interpolate.

    Returns the envelope, NaN where the original bin was an unfilled
    gap or the signal was discarded.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[1]

    # 1) interpolate short gaps, signal by signal
    max_gap_n = int(round(max_gap / step))
    filled = np.vstack([fill_short_gaps(r, max_gap_n) for r in data])
    valid = ~np.isnan(filled)

    # 2) discard signals with too little data
    keep = valid.mean(axis=1) >= min_valid_fraction
    valid &= keep[:, None]

    # 3) remaining gaps -> 0 (neutral dTEC value); mean NOT removed
    x = np.where(valid, filled, 0.0)

    # 4) Hilbert, with optional reflective padding at the edges
    pad = max(0, min(n - 1, int(round(pad_length / step))))
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    envelope = np.abs(hilbert(padded, axis=1))[:, pad:pad + n]

    # 5) restore gaps
    envelope[~valid] = np.nan
    return envelope


def keogram_envelope(values: np.ndarray, process_axis: str, **kw):
    """
    Apply envelope_along_last_axis to a (degree x time) grid.

    process_axis : "row"    -> along TIME, one signal per lat/lon bin
                   "column" -> along LAT/LON, one signal per time step
    Output keeps the (degree x time) orientation.
    """
    if process_axis == "row":
        return envelope_along_last_axis(values, **kw)
    if process_axis == "column":
        return envelope_along_last_axis(values.T, **kw).T
    raise ValueError(f"PROCESS_AXIS must be 'row' or 'column', got {process_axis!r}.")


def load_keogram(nc_path: Path, boundaries: tuple = None):
    """
    Open a saved keogram grid, oriented as (degree x time).

    boundaries : optional (deg_min, deg_max) restricting the degree axis.

    Returns (deg_dim, deg_values, t_seconds, values[deg, time]).
    """
    agg = xr.open_dataarray(nc_path)
    deg_dim = [d for d in agg.dims if d != "t_seconds"][0]
    agg = agg.transpose(deg_dim, "t_seconds")

    deg = np.asarray(agg[deg_dim].values, dtype=float)
    t_seconds = np.asarray(agg["t_seconds"].values, dtype=float)
    values = np.asarray(agg.values, dtype=float)
    agg.close()

    if boundaries is not None:
        lo, hi = boundaries
        mask = (deg >= lo) & (deg <= hi)
        deg, values = deg[mask], values[mask, :]

    return deg_dim, deg, t_seconds, values


def axis_step(coord: np.ndarray, name: str) -> float:
    """Sampling interval of a coordinate; warns if it is not uniform."""
    steps = np.abs(np.diff(coord))
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=1e-3):
        print(f"[warn] non-uniform {name} axis; using median step = {step:g}")
    return step


def hilbert_path(nc_path: Path, tag: str) -> Path:
    """Output path for the envelope grid, next to the original .nc."""
    return nc_path.with_name(f"{nc_path.stem}_hilbert_{tag}.nc")


def build_hilbert_grid(nc_path: Path, boundaries: tuple = None,
                       process_axis: str = "row",
                       edge_pad_min: float = 60.0,
                       max_gap_min: float = 5.0,
                       edge_pad_deg: float = 10.0,
                       max_gap_deg: float = 0.5,
                       min_valid_fraction: float = 0.1,
                       save: bool = False):
    """
    Load one keogram and return its Hilbert-envelope grid, processed
    along time (process_axis="row", periods in minutes) or along the
    degree axis (process_axis="column", wavelengths in degrees).

    Returns dict(deg=..., hours=..., env=..., deg_dim=...) or None if
    the .nc file does not exist.
    """
    if not nc_path.exists():
        print(f"[skip] grid not found -> {nc_path.name}")
        return None

    deg_dim, deg, t_seconds, values = load_keogram(nc_path, boundaries)

    if process_axis == "row":
        step = axis_step(t_seconds, "time")                       # seconds
        kw = dict(step=step, pad_length=edge_pad_min * 60.0,
                  max_gap=max_gap_min * 60.0)
    else:
        step = axis_step(deg, deg_dim)                            # degrees
        kw = dict(step=step, pad_length=edge_pad_deg, max_gap=max_gap_deg)
    tag = process_axis

    env = keogram_envelope(values, process_axis,
                           min_valid_fraction=min_valid_fraction, **kw)

    if save:
        out = hilbert_path(nc_path, tag)
        da = xr.DataArray(
            env.astype(np.float32),
            coords={deg_dim: deg, "t_seconds": t_seconds},
            dims=(deg_dim, "t_seconds"),
            name="dtec_hilbert_envelope",
            attrs={
                "units": "TECU",
                "description": ("pure |hilbert(dTEC)| (no filter, mean not removed), along "
                                + ("time, per degree bin" if process_axis == "row"
                                   else f"{deg_dim}, per time step")),
                "process_axis": process_axis,
                "max_gap_min": max_gap_min,
                "min_valid_fraction": min_valid_fraction,
                "source_grid": nc_path.name,
            },
        )
        da.to_netcdf(out)
        print(f"[save] {out.name}")

    print(f"[ok]   {nc_path.name}: envelope ({tag}, step = {step:g}).")
    return dict(deg=deg, hours=t_seconds / 3600.0, env=env, deg_dim=deg_dim, tag=tag)


def crop_time(grid: dict, start_hour: float, end_hour: float):
    """Crop a grid dict to the [start_hour, end_hour] time interval."""
    if grid is None:
        return None
    mask = (grid["hours"] >= start_hour) & (grid["hours"] <= end_hour)
    return dict(grid, hours=grid["hours"][mask], env=grid["env"][:, mask])


# ==========================================
# 3. Plot
# ==========================================
def _format_time_axis(ax, start_hour: float, end_hour: float):
    """Major/minor ticks adapted to the length of the time window."""
    span = end_hour - start_hour
    ax.set_xlim(start_hour, end_hour)
    if span > 12:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    else:
        step = 1 if span <= 6 else 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 60))
        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(_minor_time_formatter))
        ax.tick_params(axis="x", which="minor", labelsize=7)


def _common_vmax(grids, percentile: float):
    """Shared color limit for both panels (percentile of valid values)."""
    vals = [g["env"][~np.isnan(g["env"])] for g in grids if g is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return 1.0
    return float(np.percentile(np.concatenate(vals), percentile))


def plot_hilbert_keograms(date_str: str, lon_grid: dict, lat_grid: dict,
                          start_hour: float, end_hour: float,
                          resolution: str = "ultra",
                          tag: str = "raw", dtec_threshold=None,
                          cmap: str = "inferno", vmax_percentile: float = 99.0,
                          out_path: Path = None):
    """
    One figure, two panels sharing the time axis:
    top = longitude envelope, bottom = latitude envelope.
    """
    lon_grid = crop_time(lon_grid, start_hour, end_hour)
    lat_grid = crop_time(lat_grid, start_hour, end_hour)
    if lon_grid is None and lat_grid is None:
        print(f"Nothing to plot for {date_str} (no grid available).")
        return

    # Color limit: DTEC_THRESHOLD if given, else a percentile of the data
    if dtec_threshold is not None:
        vmax = float(dtec_threshold)
        vmax_txt = f"{vmax:g} TECU (DTEC_THRESHOLD)"
    else:
        vmax = _common_vmax([lon_grid, lat_grid], vmax_percentile)
        vmax_txt = f"p{vmax_percentile:g}"

    fig, (ax_lon, ax_lat) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                         constrained_layout=True)
    mesh = None
    for ax, grid, name in ((ax_lon, lon_grid, "Longitude"), (ax_lat, lat_grid, "Latitude")):
        if grid is None or grid["hours"].size == 0:
            ax.text(0.5, 0.5, f"{name} grid not available",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
        else:
            mesh = ax.pcolormesh(grid["hours"], grid["deg"], grid["env"],
                                 shading="auto", cmap=cmap, vmin=0, vmax=vmax,
                                 rasterized=True)
            ax.set_ylim(grid["deg"].min(), grid["deg"].max())
        ax.set_ylabel(f"{name} (°)")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(True, which="major", axis="x", linestyle=":", color="white", alpha=0.4)

    _format_time_axis(ax_lat, start_hour, end_hour)
    ax_lat.set_xlabel("Time (UTC, hours)")

    fig.suptitle(
        f"Pure Hilbert envelope of dTEC (along {'time, per row' if tag == 'row' else 'lat/lon, per column'}"
        f") — "
        f"{format_date_label(date_str)} "
        f"({format_hour(start_hour)}-{format_hour(end_hour)} UTC) — resolution: {resolution}"
    )

    if mesh is not None:
        fig.colorbar(mesh, ax=[ax_lon, ax_lat], extend="max", pad=0.01,
                     label=f"Envelope (TECU) — color limit: {vmax_txt}")

    if out_path is None:
        out_dir = dtec_keogram_figs_dir().parent / "keograms_dtecs"
        out_dir.mkdir(parents=True, exist_ok=True)
        win = f"{format_hour(start_hour)}{format_hour(end_hour)}".replace(":", "")
        out_path = out_dir / f"keogram_hilbert_{tag}_{resolution}_{date_str}_{win}.png"

    # Sequential number (_0001, _0002, ...) -- never overwrite a figure
    out_path = next_numbered_path(out_path)

    add_provenance_footer(fig, Path(out_path).parent)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {out_path}")
    plt.close(fig)   # no interactive display: save only and free memory


# ==========================================
# 4. Driver
# ==========================================
def run(date_window: str, resolution="ultra", process_axis="row",
        dtec_threshold=None,
        edge_pad_min=60.0, max_gap_min=5.0,
        edge_pad_deg=10.0, max_gap_deg=0.5,
        min_valid_fraction=0.1,
        lat_boundaries=None, lon_boundaries=None,
        save_grids=False, cmap="inferno", vmax_percentile=99.0):
    """Build the envelope grids for ONE date and plot its figure."""
    date_str, start_hour, end_hour = parse_date_window(date_window)
    year, doy = date_to_year_doy(date_str)
    lat_path, lon_path = grid_paths(year, doy, resolution)

    kw = dict(process_axis=process_axis,
              edge_pad_min=edge_pad_min, max_gap_min=max_gap_min,
              edge_pad_deg=edge_pad_deg, max_gap_deg=max_gap_deg,
              min_valid_fraction=min_valid_fraction,
              save=save_grids)
    lon_grid = build_hilbert_grid(lon_path, lon_boundaries, **kw)
    lat_grid = build_hilbert_grid(lat_path, lat_boundaries, **kw)

    plot_hilbert_keograms(date_str, lon_grid, lat_grid, start_hour, end_hour,
                          resolution=resolution,
                          tag=process_axis,
                          dtec_threshold=dtec_threshold,
                          cmap=cmap, vmax_percentile=vmax_percentile)


# ==========================================
# 5. Entry point (Spyder / IDE)
# ==========================================
if __name__ == "__main__":

    # 'YYYYMMDD' (full day), 'YYYYMMDD-HHHH' or 'YYYYMMDD-HHMMHHMM'
    DATE_WINDOW = "20211023"
    DATE_WINDOW = "20260722"

    # Must match the resolution used when the grids were built
    RESOLUTION = "ultra"

    # Direction of the Hilbert transform (grid = lat/lon x time):
    #   "row"    -> along TIME, one signal per lat/lon bin
    #               (uses EDGE_PAD_MIN and MAX_GAP_MIN)
    #   "column" -> along LAT/LON, one signal per time step
    #               (uses EDGE_PAD_DEG and MAX_GAP_DEG)
    PROCESS_AXIS = "column"

    # Upper limit of the color scale (vmax, TECU). Data are NOT modified.
    # None -> use VMAX_PERCENTILE of the envelope values instead.
    DTEC_THRESHOLD = None
    DTEC_THRESHOLD = 0.15

    # --- PROCESS_AXIS = "row" (time) ---
    EDGE_PAD_MIN = 60.0        # reflective padding at the edges (0 = none)
    MAX_GAP_MIN = 5.0          # gaps up to this length are interpolated

    # --- PROCESS_AXIS = "column" (lat/lon) ---
    EDGE_PAD_DEG = 10.0        # reflective padding at the edges (0 = none)
    MAX_GAP_DEG = 0.5          # gaps up to this length are interpolated

    # Signals (rows or columns) with less valid data than this fraction
    # are discarded
    MIN_VALID_FRACTION = 0.1

    # Optional degree-range restriction
    LAT_BOUNDARIES = None   # e.g. (-30.0, -10.0)
    LON_BOUNDARIES = None   # e.g. (-60.0, -40.0)

    # Write the envelope grids as .nc next to the original keograms
    SAVE_GRIDS = False

    # Colormap and fallback color limit (used only if DTEC_THRESHOLD is None)
    CMAP = "inferno"
    VMAX_PERCENTILE = 99.0

    run(DATE_WINDOW, resolution=RESOLUTION, process_axis=PROCESS_AXIS,
        dtec_threshold=DTEC_THRESHOLD,
        edge_pad_min=EDGE_PAD_MIN, max_gap_min=MAX_GAP_MIN,
        edge_pad_deg=EDGE_PAD_DEG,
        max_gap_deg=MAX_GAP_DEG, min_valid_fraction=MIN_VALID_FRACTION,
        lat_boundaries=LAT_BOUNDARIES, lon_boundaries=LON_BOUNDARIES,
        save_grids=SAVE_GRIDS, cmap=CMAP, vmax_percentile=VMAX_PERCENTILE)