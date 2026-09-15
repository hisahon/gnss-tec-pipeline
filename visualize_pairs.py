# -*- coding: utf-8 -*-
"""
visualize_pairs.py — Plots TEC (pcarrier) for receiver x PRN pairs,
reading directly from the parquet files written by
process_tec_pcarrier.py.

Expected layout:
    tec_pc/{year}/{doy:03d}/{station}_{year}{doy:03d}.parquet
    columns: prn, horario, tec_pcarrier
    e.g.: tec_pc/2026/204/25ma_2026204.parquet

Usage:
    from visualize_pairs import plot_tec_pairs

    pairs = [("braz", "G01"), ("braz", "G02"), ("alar", "R05")]
    plot_tec_pairs(pairs, date=20251025)                 # plain date, default 12h-21h window
    plot_tec_pairs(pairs, date="20251025_1530-1700")     # date + explicit time window

Notes
-----
`pairs` is a list of (station, prn) tuples rather than a dict, on
purpose: a dict like {"itai": {"G29"}, "itai": {"R18"}} silently
loses the first entry, since dict keys must be unique in Python. A
list of tuples has no such issue and lets the same station repeat as
many times as needed.

If `tec_pc_root` is not given, it defaults to `tec_pc_dir(year, doy)`
from shared/config.py (the actual DB_ROOT-based path), instead of a
relative "tec_pc" folder that may not exist from wherever the script
is run.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from shared.config import tec_pc_dir

PairsInput = Iterable[tuple[str, str]]  # (station, prn)


def _year_doy(date: int | str) -> tuple[int, int]:
    """YYYYMMDD -> (year, doy). Disk layout is by doy; the API takes a date."""
    d = datetime.strptime(str(date), "%Y%m%d")
    return d.year, d.timetuple().tm_yday


def _parse_date_arg(date: int | str) -> tuple[str, float | None, float | None]:
    """
    Parse the `date` argument in one of two forms:
      - plain date: 20260723 (int or str)
          -> ("20260723", None, None)
      - date + time window: "20260723_1530-1700" (HHMM-HHMM)
          -> ("20260723", 15.5, 17.0)   -- fractional hours

    A window given this way overrides the hour_min/hour_max defaults.
    """
    s = str(date)
    if "_" not in s:
        return s, None, None

    date_part, window_part = s.split("_", 1)
    start_str, end_str = window_part.split("-")

    def _to_frac_hour(hhmm: str) -> float:
        hhmm = hhmm.strip().zfill(4)
        return int(hhmm[:2]) + int(hhmm[2:]) / 60.0

    return date_part, _to_frac_hour(start_str), _to_frac_hour(end_str)


def _load(
    pairs: PairsInput,
    tec_pc_root: Path,
    year: int,
    doy: int,
    hour_min: float = 12.0,
    hour_max: float = 21.0,
) -> pd.DataFrame:
    """Read, with parquet filter pushdown, only the rows for the requested pairs.

    Rows with `horario` outside [hour_min, hour_max] are dropped before
    returning.
    """
    day_dir = tec_pc_root / str(year) / f"{doy:03d}"
    cols = ["station", "prn", "horario", "tec_pcarrier"]

    if not day_dir.exists():
        print(f"[warn] Directory not found: {day_dir}")
        return pd.DataFrame(columns=cols)

    # group requested PRNs by station (also merges duplicate station entries)
    by_station: dict[str, set[str]] = defaultdict(set)
    for station, prn in pairs:
        by_station[station.lower()].add(prn)

    frames = []
    for station, prns in by_station.items():
        matches = sorted(day_dir.glob(f"{station}_*.parquet"))
        if not matches:
            print(f"[warn] No parquet file for station '{station}' in {day_dir}")
            continue

        for path in matches:
            table = pq.read_table(
                path,
                columns=["prn", "horario", "tec_pcarrier"],
                filters=[("prn", "in", list(prns))],
            )
            if table.num_rows:
                df = table.to_pandas()
                df["station"] = station
                frames.append(df)
            else:
                print(f"[warn] {path.name}: no rows matched PRNs {sorted(prns)}")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    if not df.empty:
        if pd.api.types.is_datetime64_any_dtype(df["horario"]):
            # horario is a full timestamp; compare only the time-of-day part
            hour_of_day = (
                df["horario"].dt.hour
                + df["horario"].dt.minute / 60.0
                + df["horario"].dt.second / 3600.0
            )
        else:
            hour_of_day = df["horario"].astype(float)
        df = df[(hour_of_day >= hour_min) & (hour_of_day <= hour_max)]
    return df


def plot_tec_pairs(
    pairs: PairsInput,
    date: int | str,
    tec_pc_root: Path | None = None,
    save_path: Path | None = None,
    hour_min: float = 12.0,
    hour_max: float = 21.0,
) -> plt.Axes:
    """One line per station-PRN pair: tec_pcarrier vs horario.

    `date` accepts two forms:
      - plain date: 20260723 (int or str)
      - date + time window: "20260723_1530-1700" (HHMM-HHMM), which
        overrides hour_min/hour_max with that window.

    `tec_pc_root` defaults to `tec_pc_dir(year, doy)` from shared/config.py
    when not given.
    Only rows with `horario` in [hour_min, hour_max] are plotted.
    """
    date_str, win_start, win_end = _parse_date_arg(date)
    if win_start is not None:
        hour_min, hour_max = win_start, win_end

    year, doy = _year_doy(date_str)
    if tec_pc_root is None:
        tec_pc_root = tec_pc_dir(year, doy).parent.parent  # tec_pc/ (without year/doy)

    df = _load(pairs, tec_pc_root, year, doy, hour_min=hour_min, hour_max=hour_max)

    fig, ax = plt.subplots(figsize=(11, 5))
    title_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    if df.empty:
        ax.set_title(f"No data found for {title_date} (doy {doy:03d})")
        print("[warn] Nothing to plot -- check the [warn] messages above for the reason.")
        return ax

    for (station, prn), group in df.groupby(["station", "prn"]):
        group = group.sort_values("horario")
        tec = group["tec_pcarrier"] - group["tec_pcarrier"].iloc[0]
        ax.plot(group["horario"], tec, lw=0.8, label=f"{station.upper()}-{prn}")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("TEC pcarrier (TECU, relative to first sample)")
    ax.set_title(f"TEC (pcarrier) — {title_date} (doy {doy:03d}), {hour_min:g}h-{hour_max:g}h")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        figs_dir = tec_pc_dir(year, doy).parent.parent.parent / "tec_pc_figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        save_path = figs_dir / f"tec_pairs_{date_str}_{hour_min:g}h-{hour_max:g}h.png"

    fig.savefig(save_path, dpi=150)
    print(f"Figure saved: {save_path}")

    return ax


def moving_average_convolve(values: pd.Series, times: pd.Series, window: str = "10min",
                             edge_policy: str = "nan") -> np.ndarray:
    """
    Smooth `values` (assumed already sorted by `times`) with a
    centered moving average computed via np.convolve (a boxcar /
    rectangular kernel).

    The kernel length (in samples) is derived from `window` (a pandas
    time offset alias, e.g. "10min") divided by the series' own
    median sampling interval, rounded to the nearest odd number so
    the convolution stays centered on each point.

    edge_policy controls what happens where the window isn't fully
    covered by real data (the first/last ~window/2 samples):
      - "nan"     (default) computes the average only where the
                  kernel is fully covered (np.convolve mode="valid"),
                  leaving NaN at the edges -- a gap instead of a
                  noisy, under-averaged value that can dominate the
                  y-scale.
      - "partial" divides by the actual overlap count instead (the
                  window shrinks near the edges rather than leaving a
                  gap), which is more prone to edge noise.

    Returns a numpy array aligned with `values`/`times` (same order).
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values.copy()

    t = pd.DatetimeIndex(times)
    dt_seconds = pd.Series(t).diff().dt.total_seconds().median()
    if not dt_seconds or np.isnan(dt_seconds) or dt_seconds <= 0:
        return values.copy()

    window_seconds = pd.Timedelta(window).total_seconds()
    window_samples = max(1, int(round(window_seconds / dt_seconds)))
    if window_samples % 2 == 0:
        window_samples += 1  # keep the kernel centered (odd length)

    if edge_policy == "nan":
        # only fully-covered windows are computed; edges are left as NaN
        kernel = np.ones(window_samples) / window_samples
        valid = np.convolve(values, kernel, mode="valid")
        pad = window_samples // 2
        baseline = np.full(values.shape, np.nan)
        baseline[pad: pad + valid.size] = valid
    elif edge_policy == "partial":
        kernel = np.ones(window_samples)
        sums = np.convolve(values, kernel, mode="same")
        counts = np.convolve(np.ones_like(values), kernel, mode="same")
        baseline = sums / counts
    else:
        raise ValueError(f"Unknown edge_policy '{edge_policy}' (use 'nan' or 'partial').")

    return baseline


# ── Lightning flash rate (LFR) — same data source as lightning_goes_trajec.py ──
# NOTE: these defaults mirror the constants found in lightning_goes_trajec.py
# (PARQUET_DIR, NEW_SAMPLING_RATE, ROI_LAT_*/ROI_LON_*). Adjust them if your
# lightning parquet folder or region of interest is different.
LIGHTNING_PARQUET_DIR = Path("D:/bolt_pipeline/data/parquet")
LIGHTNING_SAMPLING_RATE = 15  # matches "*_{15}s.parquet" naming
LIGHTNING_LAT_RANGE = (-35.0, -15.0)
LIGHTNING_LON_RANGE = (-65.0, -45.0)


def _load_lightning_flashes(
    date_str: str,
    parquet_dir: Path = LIGHTNING_PARQUET_DIR,
    sampling_rate: int = LIGHTNING_SAMPLING_RATE,
) -> pd.DataFrame:
    """
    Read the lightning-strike parquet(s) for `date_str` (YYYYMMDD),
    using the same naming convention as lightning_goes_trajec.py:
    "*{date_str}*_{sampling_rate}s.parquet" under `parquet_dir`.

    Returns columns: time_stamp (datetime), latitude, longitude.
    """
    parquet_dir = Path(parquet_dir)
    pattern = f"*{date_str}*_{sampling_rate}s.parquet"
    candidates = sorted(parquet_dir.glob(pattern))
    cols = ["time_stamp", "latitude", "longitude"]

    if not candidates:
        print(f"[warn] No lightning parquet found for {date_str}")
        print(f"[warn]   searched in: {parquet_dir.resolve()}")
        print(f"[warn]   pattern used: '{pattern}'")
        print("[warn]   if this path is wrong, pass lightning_parquet_dir=Path(r'...') explicitly")
        return pd.DataFrame(columns=cols)

    print(f"[info] Lightning: {len(candidates)} parquet file(s) matched in {parquet_dir.resolve()}")
    frames = [pd.read_parquet(p, columns=cols) for p in candidates]
    out = pd.concat(frames, ignore_index=True)
    out["time_stamp"] = pd.to_datetime(out["time_stamp"])
    print(f"[info] Lightning: {len(out):,} raw strike(s) loaded for {date_str}")
    return out


def compute_flash_rate(
    lightning: pd.DataFrame,
    lat_range: tuple | None = LIGHTNING_LAT_RANGE,
    lon_range: tuple | None = LIGHTNING_LON_RANGE,
) -> pd.Series:
    """
    Count lightning strikes per unique timestamp (the data is already
    resampled onto a fixed LIGHTNING_SAMPLING_RATE-second grid, so no
    additional binning is needed) and return the Lightning Flash Rate
    (LFR), in flashes/second, as a pandas Series indexed by timestamp.

    Restricts to `lat_range`/`lon_range` first when given (defaults
    to the same ROI used in lightning_goes_trajec.py).
    """
    df = lightning
    n_before = len(df)
    if lat_range is not None:
        lo, hi = sorted(lat_range)
        df = df[(df["latitude"] >= lo) & (df["latitude"] <= hi)]
    if lon_range is not None:
        lo, hi = sorted(lon_range)
        df = df[(df["longitude"] >= lo) & (df["longitude"] <= hi)]

    if n_before:
        print(f"[info] Lightning: {len(df):,}/{n_before:,} strike(s) kept after ROI "
              f"(lat {lat_range}, lon {lon_range})")

    if df.empty:
        print("[warn] Lightning: 0 strikes in the ROI -- LFR will be empty and dTEC x LFR all NaN.")
        return pd.Series(dtype=float)

    counts = df.groupby("time_stamp").size().sort_index()
    return counts / LIGHTNING_SAMPLING_RATE  # flashes per second


def _interp_to_times(series: pd.Series, times: pd.Series) -> np.ndarray:
    """
    Linearly interpolate a time-indexed Series (e.g. LFR, on a coarser
    time grid) onto arbitrary target timestamps, for elementwise
    multiplication with a finer-cadence series like dTEC. Targets
    outside the series' time range get NaN (no extrapolation).

    Both time arrays are normalized to datetime64[ns] before the
    numeric conversion, since `series.index` and `times` may come
    from different sources with different native resolutions (e.g.
    horario as datetime64[ms] vs. lightning time_stamp as
    datetime64[ns]) -- comparing raw int64 views of mismatched
    resolutions silently produces wrong (all-NaN) results.
    """
    if series.empty:
        return np.full(len(times), np.nan)

    s = series.sort_index()
    t_source = s.index.values.astype("datetime64[ns]").astype(np.int64)
    t_target = pd.DatetimeIndex(times).values.astype("datetime64[ns]").astype(np.int64)
    return np.interp(t_target, t_source, s.values, left=np.nan, right=np.nan)


def plot_tec_pairs_smoothed(
    pairs: PairsInput,
    date: int | str,
    tec_pc_root: Path | None = None,
    save_path: Path | None = None,
    hour_min: float = 12.0,
    hour_max: float = 21.0,
    smooth_window: str = "10min",
) -> plt.Axes:
    """
    Same data as plot_tec_pairs, but each pair's dTEC is computed as
    `tec_pcarrier - moving_average_convolve(tec_pcarrier, window)`:
    the raw TEC minus its own smoothed baseline (a moving average /
    boxcar convolution over `smooth_window`, e.g. "10min"). This
    detrends the slow background variation and highlights
    oscillations around zero.

    `date` accepts the same two forms as plot_tec_pairs (plain date,
    or "YYYYMMDD_HHMM-HHMM").
    """
    date_str, win_start, win_end = _parse_date_arg(date)
    if win_start is not None:
        hour_min, hour_max = win_start, win_end

    year, doy = _year_doy(date_str)
    if tec_pc_root is None:
        tec_pc_root = tec_pc_dir(year, doy).parent.parent  # tec_pc/ (without year/doy)

    df = _load(pairs, tec_pc_root, year, doy, hour_min=hour_min, hour_max=hour_max)

    fig, ax = plt.subplots(figsize=(11, 5))
    title_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    if df.empty:
        ax.set_title(f"No data found for {title_date} (doy {doy:03d})")
        print("[warn] Nothing to plot -- check the [warn] messages above for the reason.")
        return ax

    if not pd.api.types.is_datetime64_any_dtype(df["horario"]):
        raise TypeError(
            "plot_tec_pairs_smoothed needs 'horario' as a datetime column "
            "to apply a time-based moving average."
        )

    for (station, prn), group in df.groupby(["station", "prn"]):
        group = group.sort_values("horario")
        baseline = moving_average_convolve(group["tec_pcarrier"], group["horario"], window=smooth_window)
        dtec = group["tec_pcarrier"].values - baseline
        ax.plot(group["horario"], dtec, lw=0.8, label=f"{station.upper()}-{prn}")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(f"dTEC pcarrier (TECU, tec - {smooth_window} convolve)")
    ax.set_title(
        f"dTEC (pcarrier, tec - {smooth_window} convolve) — {title_date} "
        f"(doy {doy:03d}), {hour_min:g}h-{hour_max:g}h"
    )
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        figs_dir = tec_pc_dir(year, doy).parent.parent.parent / "tec_pc_figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        save_path = figs_dir / f"tec_pairs_dtec_{smooth_window}_{date_str}_{hour_min:g}h-{hour_max:g}h.png"

    fig.savefig(save_path, dpi=150)
    print(f"Figure saved: {save_path}")

    return ax


def plot_tec_pairs_dtec_times_lfr(
    pairs: PairsInput,
    date: int | str,
    tec_pc_root: Path | None = None,
    save_path: Path | None = None,
    hour_min: float = 12.0,
    hour_max: float = 21.0,
    smooth_window: str = "10min",
    lightning_parquet_dir: Path = LIGHTNING_PARQUET_DIR,
    lightning_lat_range: tuple = LIGHTNING_LAT_RANGE,
    lightning_lon_range: tuple = LIGHTNING_LON_RANGE,
) -> plt.Axes:
    """
    Figure 3: for each pair, plots dTEC(t) * LFR(t).

    dTEC is the same `tec_pcarrier - moving_average_convolve(...)`
    series used in plot_tec_pairs_smoothed. LFR (Lightning Flash Rate,
    in flashes/second) comes from the lightning-strike parquet files,
    read the same way as lightning_goes_trajec.py, restricted to
    `lightning_lat_range`/`lightning_lon_range`, counted per unique
    timestamp (the data is already resampled onto a fixed 15s grid),
    then linearly interpolated onto each pair's own timestamps.

    LFR is a regional quantity, not per-station: the same LFR curve
    is used for every pair, only dTEC differs between them.
    """
    date_str, win_start, win_end = _parse_date_arg(date)
    if win_start is not None:
        hour_min, hour_max = win_start, win_end

    year, doy = _year_doy(date_str)
    if tec_pc_root is None:
        tec_pc_root = tec_pc_dir(year, doy).parent.parent  # tec_pc/ (without year/doy)

    df = _load(pairs, tec_pc_root, year, doy, hour_min=hour_min, hour_max=hour_max)

    fig, ax = plt.subplots(figsize=(11, 5))
    title_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    if df.empty:
        ax.set_title(f"No data found for {title_date} (doy {doy:03d})")
        print("[warn] Nothing to plot -- check the [warn] messages above for the reason.")
        return ax

    if not pd.api.types.is_datetime64_any_dtype(df["horario"]):
        raise TypeError(
            "plot_tec_pairs_dtec_times_lfr needs 'horario' as a datetime column."
        )

    lightning = _load_lightning_flashes(date_str, lightning_parquet_dir)
    lfr = compute_flash_rate(
        lightning, lat_range=lightning_lat_range, lon_range=lightning_lon_range,
    )
    if lfr.empty:
        print("[warn] No lightning data available -- the product will be all NaN.")
    else:
        print(f"[info] LFR: {len(lfr)} bin(s), {lfr.index.min()} to {lfr.index.max()}, "
              f"max {lfr.max():.3f} flashes/sec")

    for (station, prn), group in df.groupby(["station", "prn"]):
        group = group.sort_values("horario")
        baseline = moving_average_convolve(group["tec_pcarrier"], group["horario"], window=smooth_window)
        dtec = group["tec_pcarrier"].values - baseline
        lfr_at_times = _interp_to_times(lfr, group["horario"])
        product = dtec * lfr_at_times
        ax.plot(group["horario"], product, lw=0.8, label=f"{station.upper()}-{prn}")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("dTEC \u00d7 LFR (TECU \u00b7 flashes/sec)")
    ax.set_title(
        f"dTEC \u00d7 Lightning Flash Rate — {title_date} (doy {doy:03d}), "
        f"{hour_min:g}h-{hour_max:g}h"
    )
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        figs_dir = tec_pc_dir(year, doy).parent.parent.parent / "tec_pc_figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        save_path = figs_dir / f"tec_pairs_dtec_x_lfr_{date_str}_{hour_min:g}h-{hour_max:g}h.png"

    fig.savefig(save_path, dpi=150)
    print(f"Figure saved: {save_path}")

    return ax


if __name__ == "__main__":
    date = "20260723_1600-1800"
    pairs = [
        ("fosa", "R18"),
        ("puma", "G11"),
        ("prcl", "G29"),
        ("itai", "G29"),
        ("itai", "R18"),
        ("stha", "G29"),
        ("stha", "R18"),
        ("scaq", "R18"),
    ]
    
    
    plot_tec_pairs(pairs, date=date)
    plot_tec_pairs_smoothed(pairs, date=date, smooth_window="10min")
    plot_tec_pairs_dtec_times_lfr(pairs, date=date, smooth_window="10min")
    plt.show()