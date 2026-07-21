# -*- coding: utf-8 -*-
"""
shared/lightning_spectrogram.py

Lightning Flash Rate (LFR) wavelet spectrogram.

Given a lat/lon/time window, this module:
  1. locates and loads the day's lightning parquet from LIGHTNING_DIR
     (one parquet per full day),
  2. clips it to the requested spatial box + time window and resamples
     it into a regular flash-rate series (flashes/s) at
     LFR_SAMPLING_RATE seconds, filling gaps with 0,
  3. band-pass filters it (zero-phase Butterworth, via filtfilt) to
     keep only BANDPASS_LOW_MHZ-BANDPASS_HIGH_MHZ before the wavelet,
  4. runs a Morlet wavelet transform (Torrence & Compo, via
     shared.waveletFunctions) and slices it to WAV_FREQ_BOUNDS,
  5. normalises + log10s the power, ready to hand straight to
     ax.contourf(...).

Parameters and normalisation are copied from lfr_explorer.py / f2_spec.py
so results stay directly comparable with the standalone LFR explorer tool.
WAV_DT is in minutes (0.25 min = 15s, matching LFR_SAMPLING_RATE), which is
why the wavelet's `period` output — and the resulting frequency axis — is
naturally expressed in minutes before being converted to mHz.

Typical usage
-------------
    from shared.lightning_spectrogram import lfr_spectrogram

    result = lfr_spectrogram(year, doy, lat_range, lon_range, time_range)
    if result is not None:
        t_hours, freq, power = result
        ax.contourf(t_hours, freq, power, cmap="nipy_spectral", levels=20)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from shared.waveletFunctions import wavelet_on_data_new, wd_slicer

# ==========================================================================
# Config
# ==========================================================================
LIGHTNING_DIR = Path("D:/database/lightning/parquet_15s")   # one parquet per full day
LFR_SAMPLING_RATE = 15   # seconds -- must match the *_15s.parquet cadence

# wavelet parameters -- exact values from lfr_explorer.py / f2_spec.py
WAV_DT     = 0.25    # time unit = MINUTES (not hours, despite how the
                      # source script's comment reads). 15s sampling rate
                      # -> 15/60 = 0.25 min/sample, so this is correct and
                      # consistent with LFR_SAMPLING_RATE=15 below. `period`
                      # (returned by the wavelet transform) therefore comes
                      # out in minutes too, which is why the freq -> mHz
                      # conversion in wavelet_on_data_new uses a 1/60 factor
                      # (cycles/min -> Hz -> mHz).
WAV_PAD    = 1
WAV_DJ     = 0.15
WAV_S0     = 0.90
WAV_J1     = 24
WAV_LAG1   = 0.71
WAV_MOTHER = "MORLET"
WAV_FREQ_BOUNDS = [2, 10]   # mHz, passed to wd_slicer (crops the wavelet
                            # output for display -- separate from the
                            # time-domain band-pass below)

# Band-pass filter (Butterworth, zero-phase via filtfilt) applied to the
# LFR series BEFORE the wavelet transform, replacing the previous
# detrend + moving-average smoothing.
BANDPASS_LOW_MHZ  = 1    # mHz -- frequencies below this are removed
BANDPASS_HIGH_MHZ = 12   # mHz -- frequencies above this are removed
BANDPASS_ORDER    = 4


# ==========================================================================
# Internals
# ==========================================================================
def _bandpass_filter(
    series: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter (via filtfilt, so it doesn't
    shift the signal in time). Removes frequencies below `low_hz` and
    above `high_hz`.
    """
    nyq  = fs / 2.0
    low  = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, series)


def _find_lightning_file(year: int, doy: int) -> Path | None:
    """
    Locate the lightning parquet for a given year/day-of-year inside
    LIGHTNING_DIR (one parquet per full day). The exact file-naming
    convention isn't assumed -- we match any *.parquet whose name
    contains the 8-digit YYYYMMDD for that date (same approach as
    lfr_explorer.py's _parse_date, just used in reverse).
    """
    target_date = (
        pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    ).strftime("%Y%m%d")

    if not LIGHTNING_DIR.exists():
        print(f"[lightning] folder not found: {LIGHTNING_DIR}")
        return None

    candidates = sorted(LIGHTNING_DIR.glob(f"*{target_date}*.parquet"))
    if not candidates:
        print(f"[lightning] no parquet matching {target_date} in {LIGHTNING_DIR}")
        return None
    return candidates[0]


# ==========================================================================
# Public API
# ==========================================================================
def load_lightning_lfr(
    year: int,
    doy: int,
    lat_range: tuple,
    lon_range: tuple,
    time_range: tuple,
):
    """
    Load one day of lightning flashes, clip to the lat/lon/time window,
    and return the 15s-resolution flash-rate (LFR) series -- same logic
    as lfr_explorer.py's _load() + _lfr_15s().

    Returns a pandas Series (index = regular 15s DatetimeIndex, values =
    flashes/s), or None if the file is missing or nothing survives the
    spatial/time cuts.
    """
    path = _find_lightning_file(year, doy)
    if path is None:
        return None

    df = pd.read_parquet(path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    df = df[
        df["longitude"].between(lon_lo, lon_hi) &
        df["latitude"].between(lat_lo, lat_hi)
    ]
    if df.empty:
        print(f"[lightning] no flashes inside the lat/lon box for {year}-{doy:03d}")
        return None

    hmin, hmax = time_range
    hour = df["time_stamp"].dt.hour + df["time_stamp"].dt.minute / 60
    df = df[(hour >= hmin) & (hour < hmax)]
    if df.empty:
        print(f"[lightning] no flashes inside {hmin}-{hmax}h for {year}-{doy:03d}")
        return None

    day_start = df["time_stamp"].dt.normalize().iloc[0]
    t_start = day_start + pd.Timedelta(hours=hmin)
    t_end   = day_start + pd.Timedelta(hours=hmax) - pd.Timedelta(seconds=LFR_SAMPLING_RATE)
    full_idx = pd.date_range(start=t_start, end=t_end, freq=f"{LFR_SAMPLING_RATE}s")

    lfr = (
        df.groupby("time_stamp").size().rename("lfr") / LFR_SAMPLING_RATE
    )
    return lfr.reindex(full_idx, fill_value=0.0)


def compute_spectrogram_from_lfr(lfr_values: np.ndarray):
    """
    Morlet wavelet power spectrum of an LFR array.
    Pipeline: Butterworth band-pass filter (zero-phase, via filtfilt;
    keeps only BANDPASS_LOW_MHZ-BANDPASS_HIGH_MHZ) -> wavelet.
    Wavelet parameters and normalisation copied from lfr_explorer.py's
    _compute_wavelet(); the band-pass step replaces its detrend step.

    Returns (t_hours_elapsed, freq, power) or None if there isn't enough
    signal (matches the >=16 non-zero samples guard in lfr_explorer.py).
    t_hours_elapsed is time since the start of the window, in hours.
    """
    fs = 1.0 / LFR_SAMPLING_RATE
    signal = _bandpass_filter(
        lfr_values.astype(float), fs=fs,
        low_hz=BANDPASS_LOW_MHZ * 1e-3, high_hz=BANDPASS_HIGH_MHZ * 1e-3,
        order=BANDPASS_ORDER,
    )
    if np.count_nonzero(signal) < 16:
        return None

    _period, freq, power = wavelet_on_data_new(
        signal,
        dt=WAV_DT, pad=WAV_PAD, dj=WAV_DJ,
        s0=WAV_S0, j1=WAV_J1, lag1=WAV_LAG1,
        mother=WAV_MOTHER,
    )
    freq, power = wd_slicer(freq, power, WAV_FREQ_BOUNDS)

    # normalise then log10 -- identical to lfr_explorer.py
    power = (power - power.min()) / (power.max() + 1e-30)
    power = np.log10(power + 5e-2)

    n = lfr_values.shape[0]
    t_hours_elapsed = np.arange(n) * LFR_SAMPLING_RATE / 3600.0
    return t_hours_elapsed, freq, power


def lfr_spectrogram(
    year: int,
    doy: int,
    lat_range: tuple,
    lon_range: tuple,
    time_range: tuple,
):
    """
    One-call convenience wrapper: load the LFR series for the window and
    compute its wavelet spectrogram in one go.

    Returns (t_hours, freq, power) ready for
        ax.contourf(t_hours, freq, power, cmap=..., levels=...)
    where t_hours is already the absolute UTC decimal-hour axis (i.e.
    offset by time_range[0], so it lines up with any other panel plotted
    against the same t_hours axis) -- or None if there wasn't enough
    lightning data / signal to compute a spectrogram.
    """
    lfr = load_lightning_lfr(year, doy, lat_range, lon_range, time_range)
    if lfr is None:
        return None

    result = compute_spectrogram_from_lfr(lfr.values)
    if result is None:
        print(f"[lightning] insufficient signal for wavelet ({year}-{doy:03d})")
        return None

    t_hours_elapsed, freq, power = result
    hmin, _hmax = time_range
    t_hours = hmin + t_hours_elapsed
    return t_hours, freq, power