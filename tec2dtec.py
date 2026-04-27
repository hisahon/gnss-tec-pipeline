# -*- coding: utf-8 -*-
"""
TEC (Total Electron Content) signal cleaner.

Reads parquet files containing TEC time-series per receiver, applies a
NaN-aware detrend + bandpass-filter pipeline, and writes the resulting
dTEC (detrended/filtered TEC) back as parquet. NaN gaps are treated as
segment separators and are never interpolated across.

Pipeline applied to each receiver column:
    1. Spike removal     - drop samples whose first difference exceeds
                           SPIKE_DIFF_LIMIT (TECu).
    2. Detrending        - subtract a centered Hann moving average of
                           length DETREND_WIN_SEC seconds.
    3. Bandpass filter   - zero-phase Butterworth, BANDPASS_ORDER-th order,
                           passband BANDPASS_CUTOFFS.

----------------------------------------------------------------------------
Author : Rogerio Hisashi Honda  <hisashi.brase@gmail.com>
Created: 2025-11-03
----------------------------------------------------------------------------
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt, hilbert

import config as cfg  # project-local config module (database paths, constants)


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ  = 4 / 60          # 4 samples per minute (15 s sampling)
DETREND_WIN_SEC   = 900             # moving-average window for detrending [s]
BANDPASS_CUTOFFS  = (5e-4, 1e-2)    # passband [Hz] -> ~33 min ... ~100 s
BANDPASS_ORDER    = 4
SPIKE_DIFF_LIMIT  = 5.0             # |diff(TEC)| threshold for spike removal


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_files(folder: Path | str, extension: str) -> list[Path]:
    """Recursively list files under `folder` ending with `extension`.

    The `extension` argument may be passed with or without the leading dot
    (e.g. 'parquet' or '.parquet'). Matching is case-insensitive and the
    returned list is sorted.
    """
    folder = Path(folder)
    ext = extension if extension.startswith('.') else f'.{extension}'
    ext = ext.lower()

    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == ext]
    files.sort()
    return files

# ---------------------------------------------------------------------------
# NaN-aware signal-processing primitives
# ---------------------------------------------------------------------------
# All helpers below split the input signal at NaN gaps and process each
# contiguous non-NaN segment independently. Original NaNs are preserved in
# the output. This avoids the moving-window contamination that would
# otherwise bleed across receiver outages.

def envelope_nan_segments(x: np.ndarray) -> np.ndarray:
    """Hilbert envelope |H(x)| computed segment-wise (NaN-aware)."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    isn = np.isnan(x)
    N = x.size

    i = 0
    while i < N:
        # Skip a run of NaNs.
        while i < N and isn[i]:
            i += 1
        if i >= N:
            break
        # Find the end of the next non-NaN segment.
        j = i
        while j < N and not isn[j]:
            j += 1

        seg = x[i:j]
        if seg.size >= 2:
            out[i:j] = np.abs(hilbert(seg))
        i = j
    return out


def moving_rms_nan_segments(x: np.ndarray, fs: float, win_sec: float) -> np.ndarray:
    """Centered moving-RMS (NaN-aware) - smoothed signal intensity."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)

    # Force odd window length so the result can be cleanly centered.
    n = max(1, int(round(win_sec * fs)))
    if n % 2 == 0:
        n += 1
    k = np.ones(n, float) / n  # uniform averaging kernel

    isn = np.isnan(x)
    N = x.size
    i = 0
    while i < N:
        while i < N and isn[i]:
            i += 1
        if i >= N:
            break
        j = i
        while j < N and not isn[j]:
            j += 1

        seg = x[i:j]
        if seg.size >= n:
            # RMS = sqrt(moving average of x^2), placed centered in `out`.
            y = np.sqrt(np.convolve(seg ** 2, k, mode="valid"))
            start = i + n // 2
            out[start:start + y.size] = y
        i = j
    return out


def butter_filter_nan_segments(
    x: np.ndarray,
    fs: float,
    kind: str,                    # 'lowpass' | 'highpass' | 'bandpass'
    cutoff,                       # float (lp/hp) or (f1, f2) for bandpass [Hz]
    order: int = 4,
    zero_phase: bool = True,      # True -> sosfiltfilt; False -> causal sosfilt
    min_seg_len: int | None = None,
) -> np.ndarray:
    """Apply a Butterworth filter segment-wise (NaN-aware).

    The filter is built once in SOS form and re-applied on every contiguous
    non-NaN segment. Short segments fall back to causal `sosfilt` (and finally
    to causal without padding) so the function never crashes on real data.

    Parameters
    ----------
    x          : 1D array, may contain NaNs.
    fs         : sampling rate [Hz].
    kind       : 'lowpass', 'highpass' or 'bandpass'.
    cutoff     : scalar for lp/hp, (f1, f2) for bandpass, all in Hz.
    order      : filter order.
    zero_phase : if True, use sosfiltfilt (no phase distortion); otherwise
                 sosfilt is used (causal).
    min_seg_len: minimum samples required to attempt zero-phase filtering.
                 Defaults to a conservative value derived from `order`.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    N = x.size
    isn = np.isnan(x)
    nyq = fs / 2.0

    # Normalize cutoff(s) and validate against Nyquist.
    if kind == "bandpass":
        if not (isinstance(cutoff, (tuple, list)) and len(cutoff) == 2):
            raise ValueError("For bandpass, cutoff must be (f1, f2) in Hz.")
        low, high = float(cutoff[0]), float(cutoff[1])
        if not (0 < low < high < nyq):
            raise ValueError(f"Invalid cutoffs: must satisfy 0 < f1 < f2 < {nyq} (Nyquist).")
        Wn = (low / nyq, high / nyq)
        btype = "bandpass"
    else:
        fc = float(cutoff)
        if not (0 < fc < nyq):
            raise ValueError(f"Invalid cutoff: must satisfy 0 < fc < {nyq} (Nyquist).")
        Wn = fc / nyq
        btype = "lowpass" if kind == "lowpass" else "highpass"

    # SOS form is numerically more stable than (b, a) for higher orders.
    sos = butter(order, Wn, btype=btype, output="sos")

    # Conservative padding heuristic for sosfiltfilt; min_seg_len ensures we
    # only attempt zero-phase filtering on segments long enough to support it.
    default_pad = max(1, 3 * order)
    if min_seg_len is None:
        min_seg_len = max(2 * default_pad + 5, 20)

    i = 0
    while i < N:
        while i < N and isn[i]:
            i += 1
        if i >= N:
            break
        j = i
        while j < N and not isn[j]:
            j += 1

        seg = x[i:j]
        m = seg.size
        if m <= 1:
            # Too short to filter; leave NaNs in `out`.
            i = j
            continue

        try:
            if zero_phase and m >= min_seg_len:
                padlen = min(default_pad, m - 1)  # padlen must be < segment length
                y = sosfiltfilt(sos, seg, padlen=padlen)
            else:
                # Causal fallback (introduces phase, but works on short segments).
                y = sosfilt(sos, seg)
            out[i:j] = y
        except ValueError:
            # Last-resort fallback if sosfiltfilt rejects the segment length.
            out[i:j] = sosfilt(sos, seg)

        i = j

    return out


def _apply_detrend_to_segment(
    out: np.ndarray,
    seg: np.ndarray,
    i: int,
    n: int,
    k: np.ndarray,
    align: str = "center",
) -> None:
    """Write `seg - moving_avg(seg, k)` of one segment into `out`.

    Despite operating via convolution, this writes the *residual* between the
    raw segment and its smoothed version - i.e. a high-pass / detrend, not
    the smoothed signal itself. `align` controls where the valid portion of
    the convolution is placed inside `out`.
    """
    if seg.size < n:
        return  # segment shorter than the kernel -> leave NaNs in place

    y = np.convolve(seg, k, mode="valid")  # length = m - n + 1
    if y.size < n:
        return

    off = n // 2  # half-window: used for both the slice and centered placement
    if align == "center":
        start = i + off
    elif align == "right":
        start = i + (n - 1)
    elif align == "left":
        start = i
    else:
        raise ValueError("align must be 'center', 'right' or 'left'.")

    end = start + y.size
    # Residual: original samples (aligned with the valid region) minus the
    # smoothed version. This is the detrended signal we actually want.
    out[start:end] = seg[off:-off] - y


def conv1d_nan_segments(
    x: np.ndarray,
    fs: float,
    win_sec: float,
    kernel: str = "boxcar",
) -> np.ndarray:
    """Detrend a 1D signal by subtracting a centered moving average (NaN-aware).

    The output is *signal minus smoothed signal*, computed segment-wise so
    that NaN gaps act as series separators. The kernel is normalized to unit
    DC gain so it does not bias the smoothing.

    Parameters
    ----------
    x       : 1D float array, may contain NaNs.
    fs      : sampling rate [Hz].
    win_sec : moving-average window length [s].
    kernel  : 'boxcar' (uniform) or 'hann' (raised cosine).
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if win_sec <= 0:
        raise ValueError("win_sec must be > 0")

    x = np.asarray(x, dtype=float)

    # Force odd window length for clean centering.
    n = max(1, int(round(win_sec * fs)))
    if n % 2 == 0:
        n += 1

    if kernel == "boxcar":
        k = np.ones(n, dtype=float)
    elif kernel == "hann":
        k = np.hanning(n).astype(float)
    else:
        raise ValueError("kernel must be 'boxcar' or 'hann'.")
    k /= k.sum()  # unit-DC-gain kernel

    out = np.full_like(x, np.nan)
    isn = np.isnan(x)
    N = x.size

    i = 0
    while i < N:
        while i < N and isn[i]:
            i += 1
        if i >= N:
            break
        j = i
        while j < N and not isn[j]:
            j += 1

        _apply_detrend_to_segment(out, x[i:j], i, n, k, align="center")
        i = j

    return out


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def plot_tec_por_receptor_de_arquivo(file: Path, show: bool = False) -> None:
    """Process a single per-PRN TEC parquet file into a dTEC parquet.

    The input file must follow the naming convention
        TECPC_{YYYY}-{DOY}_{PRN}.parquet
    with the following columns:
        - first column          : datetime (timezone-aware or convertible).
        - 'x_km','y_km','z_km'  : ionospheric pierce-point coordinates.
        - all remaining numeric : per-receiver TECc time-series.

    For each receiver column the function:
        1. Removes spikes where |diff| exceeds SPIKE_DIFF_LIMIT.
        2. Subtracts a Hann moving average of length DETREND_WIN_SEC.
        3. Applies a zero-phase Butterworth bandpass (BANDPASS_CUTOFFS).

    The result is saved as
        {db_root}/dtec/{YYYY}/{DOY}/dtec_{YYYY}-{DOY}_{PRN}.parquet,
    where {db_root} is inferred from the input path (4 levels up).

    If `show` is True, three diagnostic plots are produced per receiver
    (raw TEC, detrended TEC, bandpass-filtered TEC).
    """
    base_name   = file.stem                       # e.g. 'TECPC_2021-296_G02'
    folder_base = file.parent.parent.parent.parent  # walk back to the db root

    # Parse YYYY, DOY and PRN from the filename.
    _, year_doy, prn = base_name.split('_')
    year, doy = year_doy.split('-')

    save_dir = folder_base / 'dtec' / year / doy
    os.makedirs(save_dir, exist_ok=True)
    save_file = save_dir / f"dtec_{year_doy}_{prn}.parquet"

    df = pd.read_parquet(file)
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 columns: datetime + 1 receiver.")

    # Normalize the datetime column to UTC, rounded to the second.
    dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce', utc=True).dt.floor('s')

    # Output container: keep timestamp + IPP coordinates as-is.
    save_results = {
        dt_col : df[dt_col].to_numpy(),
        'x_km' : df['x_km'].to_numpy(),
        'y_km' : df['y_km'].to_numpy(),
        'z_km' : df['z_km'].to_numpy(),
    }

    df = df.sort_values(dt_col).set_index(dt_col)

    # Receiver columns = all numeric columns minus the trailing 3 IPP coords.
    num_cols = df.select_dtypes(include='number').columns.tolist()[:-3]

    for col in num_cols:
        s = df[col].astype(float)
        y = s.to_numpy()

        # 1. Spike removal: drop samples whose first difference is too large.
        y_diff = np.diff(y, prepend=np.nan)
        y[np.abs(y_diff) > SPIKE_DIFF_LIMIT] = np.nan

        # 2. Detrend by subtracting a Hann moving average.
        y_cv = conv1d_nan_segments(
            y, fs=SAMPLING_RATE_HZ, win_sec=DETREND_WIN_SEC, kernel="hann",
        )

        # 3. Zero-phase bandpass to isolate TIDs in the passband of interest.
        y_bw = butter_filter_nan_segments(
            y_cv, fs=SAMPLING_RATE_HZ, kind="bandpass",
            cutoff=BANDPASS_CUTOFFS, order=BANDPASS_ORDER,
        )

        # Skip empty receivers.
        if s.dropna().empty or len(y_diff) == 0:
            continue

        save_results[col] = y_bw

        if show:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(10, 6.5), sharex=True,
                gridspec_kw={'hspace': 0.15},
            )

            # Raw TEC after spike removal.
            ax1.plot(s.index, y, linewidth=1.2)
            ax1.set_ylabel("TECc")
            ax1.grid(True, linestyle=":", linewidth=0.6)

            # Detrended TEC (raw minus moving average).
            ax2.plot(s.index, y_cv, linewidth=1.2)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Detrended TEC")
            ax2.set_ylim(-0.3, 0.3)
            ax2.grid(True, linestyle=":", linewidth=0.6)

            # Bandpass-filtered detrended TEC.
            ax3.plot(s.index, y_bw, linewidth=1.2)
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Butterworth (bandpass)")
            ax3.set_ylim(-0.2, 0.2)
            ax3.grid(True, linestyle=":", linewidth=0.6)

            plt.show()

    # Save the dTEC parquet (timestamp + IPP coords + filtered receiver series).
    pd.DataFrame(save_results).to_parquet(save_file, index=False)
    print(f"{file.name} [done]")


def plot_tec_por_receptor_de_pasta(
    folder: Path | str,
    pattern: str = "*.parquet",
    show: bool = False,
) -> None:
    """Apply `plot_tec_por_receptor_de_arquivo` to every parquet in `folder`.

    Useful when there is one parquet per PRN sitting in the same directory.
    Errors on individual files are reported and do not stop the loop.
    """
    folder = Path(folder)
    for path in sorted(folder.glob(pattern)):
        try:
            plot_tec_por_receptor_de_arquivo(file=path, show=show)
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: process every PRN parquet for one specific day.
    dt_in = datetime(2021, 10, 23, 0, 0, 0)
    year  = dt_in.year
    doy   = int(dt_in.strftime("%j"))

    # folder_base = Path(cv.folder_db, 'ionosphere', 'tec_pc', f'{year}', f'{doy}')
    folder_base = cfg.dtec_pc_dir(year, doy)
    files = read_files(folder_base, 'parquet')

    for file in files:
        plot_tec_por_receptor_de_arquivo(file=file, show=True)