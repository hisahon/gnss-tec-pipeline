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

# import os
from datetime import datetime, timedelta
# from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt, hilbert

from shared.config import dtec_pc_dir
from shared.prn_store import load_tec_with_neighbors_v2, get_station_names


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
# SAMPLING_RATE_HZ  = 4 / 60          # 4 samples per minute (15 s sampling)
DETREND_WIN_SEC   = 600             # moving-average window for detrending [s]
BANDPASS_CUTOFFS  = (5e-4, 1e-2)    # passband [Hz] -> ~33 min ... ~100 s
BANDPASS_ORDER    = 4
SPIKE_DIFF_LIMIT  = 5.0             # |diff(TEC)| threshold for spike removal


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

def get_sampling_rate_hz(timestamps) -> float:
    """
    Calculates the sampling rate in Hertz from a sequence of timestamps.
    
    Uses the median time difference to ensure the calculation remains 
    accurate even if there are intermittent data gaps.
    
    Args:
        timestamps: A pandas DatetimeIndex or a pandas Series of datetimes.
        
    Returns:
        float: The sampling rate in Hz.
    """
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required to calculate a sampling rate.")
        
    # Standardize input to a pandas Series to easily use .diff()
    if isinstance(timestamps, pd.DatetimeIndex):
        ts_series = timestamps.to_series()
    elif isinstance(timestamps, pd.Series):
        ts_series = timestamps
    else:
        ts_series = pd.Series(timestamps)
        
    # Calculate consecutive time differences in seconds
    time_diffs = ts_series.diff().dt.total_seconds()
        
    # The median ignores missing epochs or isolated anomalies
    median_period_sec = time_diffs.median()
    
    if median_period_sec == 0 or pd.isna(median_period_sec):
        raise ValueError("Calculated median period is zero or NaN. Check for duplicate or invalid timestamps.")
        
    return 1.0 / median_period_sec


# def plot_tec_por_receptor_de_arquivo(file: Path, show: bool = False) -> None:
#     """Process a single per-PRN TEC parquet file into a dTEC parquet.

#     The input file must follow the naming convention
#         TECPC_{YYYY}-{DOY}_{PRN}.parquet
#     with the following columns:
#         - first column          : datetime (timezone-aware or convertible).
#         - 'x_km','y_km','z_km'  : ionospheric pierce-point coordinates.
#         - all remaining numeric : per-receiver TECc time-series.

#     For each receiver column the function:
#         1. Removes spikes where |diff| exceeds SPIKE_DIFF_LIMIT.
#         2. Subtracts a Hann moving average of length DETREND_WIN_SEC.
#         3. Applies a zero-phase Butterworth bandpass (BANDPASS_CUTOFFS).

#     The result is saved as
#         {db_root}/dtec/{YYYY}/{DOY}/dtec_{YYYY}-{DOY}_{PRN}.parquet,
#     where {db_root} is inferred from the input path (4 levels up).

#     If `show` is True, three diagnostic plots are produced per receiver
#     (raw TEC, detrended TEC, bandpass-filtered TEC).
#     """
#     base_name   = file.stem                       # e.g. 'TECPC_2021-296_G02'
#     folder_base = file.parent.parent.parent.parent  # walk back to the db root

#     # Parse YYYY, DOY and PRN from the filename.
#     _, year_doy, prn = base_name.split('_')
#     year, doy = year_doy.split('-')

#     save_dir = folder_base / 'dtec_pc' / year / doy
#     os.makedirs(save_dir, exist_ok=True)
#     save_file = save_dir / f"dtec_{year_doy}_{prn}.parquet"

#     df = pd.read_parquet(file)
#     if df.shape[1] < 2:
#         raise ValueError("DataFrame must have at least 2 columns: datetime + 1 receiver.")

#     # Normalize the datetime column to UTC, rounded to the second.
#     dt_col = df.columns[0]
#     df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce', utc=True).dt.floor('s')

#     # Output container: keep timestamp + IPP coordinates as-is.
#     save_results = {
#         dt_col : df[dt_col].to_numpy(),
#         'x_km' : df['x_km'].to_numpy(),
#         'y_km' : df['y_km'].to_numpy(),
#         'z_km' : df['z_km'].to_numpy(),
#     }

#     df = df.sort_values(dt_col).set_index(dt_col)

#     # Receiver columns = all numeric columns minus the trailing 3 IPP coords.
#     num_cols = df.select_dtypes(include='number').columns.tolist()[:-3]

#     for col in num_cols:

#         s = df[col].astype(float)
#         y = s.to_numpy()

#         # 1. Spike removal: drop samples whose first difference is too large.
#         y_diff = np.diff(y, prepend=np.nan)
#         y[np.abs(y_diff) > SPIKE_DIFF_LIMIT] = np.nan

#         # 2. Detrend by subtracting a Hann moving average.
#         y_cv = conv1d_nan_segments(
#             y, fs=SAMPLING_RATE_HZ, win_sec=DETREND_WIN_SEC, kernel="hann",
#         )

#         # 3. Zero-phase bandpass to isolate TIDs in the passband of interest.
#         y_bw = butter_filter_nan_segments(
#             y_cv, fs=SAMPLING_RATE_HZ, kind="bandpass",
#             cutoff=BANDPASS_CUTOFFS, order=BANDPASS_ORDER,
#         )

#         # Skip empty receivers.
#         if s.dropna().empty or len(y_diff) == 0:
#             continue

#         save_results[col] = y_bw

#         if show:
#             fig, (ax1, ax2, ax3) = plt.subplots(
#                 3, 1, figsize=(10, 6.5), sharex=True,
#                 gridspec_kw={'hspace': 0.15},
#             )

#             # Raw TEC after spike removal.
#             ax1.plot(s.index, y, linewidth=1.2)
#             ax1.set_ylabel("TECc")
#             ax1.grid(True, linestyle=":", linewidth=0.6)

#             # Detrended TEC (raw minus moving average).
#             ax2.plot(s.index, y_cv, linewidth=1.2)
#             ax2.set_xlabel("Time")
#             ax2.set_ylabel("Detrended TEC")
#             ax2.set_ylim(-1, 1)
#             ax2.grid(True, linestyle=":", linewidth=0.6)

#             # Bandpass-filtered detrended TEC.
#             ax3.plot(s.index, y_bw, linewidth=1.2)
#             ax3.set_xlabel("Time")
#             ax3.set_ylabel("Butterworth (bandpass)")
#             ax3.set_ylim(-0.2, 0.2)
#             ax3.grid(True, linestyle=":", linewidth=0.6)

#             plt.show()

#     # Save the dTEC parquet (timestamp + IPP coords + filtered receiver series).
#     pd.DataFrame(save_results).to_parquet(save_file, index=False)
#     print(f"{file.name} [done]")
    

def aplicar_filtros_dtec(df: pd.DataFrame, station: str = "Desconhecida", plot_figs: bool = False) -> pd.DataFrame:
    """
    Processa um DataFrame de TEC em formato long e aplica o pipeline de dTEC.
    
    Espera-se que o DataFrame contenha pelo menos:
    - Uma coluna temporal ('horario' ou na primeira posição)
    - Uma coluna 'prn'
    - Uma coluna de valores numéricos de TEC (ex: 'tec_pcarrier')
    
    Retorna um novo DataFrame com a coluna adicional 'dtec'.
    """
    if df.empty:
        return df
    
        
    if 'prn' not in df.columns:
        raise ValueError(f"O DataFrame precisa ter uma coluna 'prn'. Colunas encontradas: {df.columns.tolist()}")

    # Copia para não modificar o DataFrame original em memória
    df = df.copy()

    # Normalizar a coluna de tempo (pega 'horario' ou a primeira coluna)
    dt_col = 'horario' if 'horario' in df.columns else df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce').dt.floor('s')
    
    # df[dt_col] = df[dt_col].dt.tz_localize(None)
    df = df[(df[dt_col] > dt_in - timedelta(seconds=2*DETREND_WIN_SEC)) & 
            (df[dt_col] < dt_in + timedelta(seconds=2*DETREND_WIN_SEC))]
    
    if df.empty:
        return df
    
    # Identificar a coluna de TEC automaticamente
    tec_col = 'tec_pcarrier' if 'tec_pcarrier' in df.columns else df.select_dtypes(include='number').columns[0]


    
    processed_dfs = []
    
    SAMPLING_RATE_HZ = get_sampling_rate_hz(df[dt_col])

    # Iterar sobre cada satélite (PRN) único no DataFrame
    for prn_val in df['prn'].unique():
        # Isolar a série temporal específica deste PRN
        df_prn = df[df['prn'] == prn_val].copy()
        df_prn = df_prn.sort_values(dt_col).set_index(dt_col)
        
                
        s = df_prn[tec_col].astype(float)
        y = s.to_numpy()

        # 1. Spike removal: descartar amostras com diferença abrupta
        y_diff = np.diff(y, prepend=np.nan)
        y[np.abs(y_diff) > SPIKE_DIFF_LIMIT] = np.nan

        # 2. Detrend: subtrair a média móvel de Hann
        y_cv = conv1d_nan_segments(
            y, fs=SAMPLING_RATE_HZ, win_sec=DETREND_WIN_SEC, kernel="hann",
        )

        # # 3. Bandpass zero-phase para isolar os TIDs
        # y_bw = butter_filter_nan_segments(
        #     y_cv, fs=SAMPLING_RATE_HZ, kind="bandpass",
        #     cutoff=BANDPASS_CUTOFFS, order=BANDPASS_ORDER,
        # )

        # Pular se o sinal ficar totalmente vazio após os filtros
        if s.dropna().empty or len(y_diff) == 0:
            continue

        # Inserir o resultado processado como uma nova coluna
        # df_prn['dtec'] = y_cv
        
        df_result = pd.DataFrame({
            dt_col: df_prn.index,
            'prn': prn_val,
            'dtec': y_cv
        })
        
        df_result = df_result.dropna(subset=['dtec'])
        
        # Adiciona à lista apenas se sobrou algum dado após o dropna
        if not df_result.empty:
            processed_dfs.append(df_result)

        # Gerar os gráficos de diagnóstico, se solicitado
        if plot_figs:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(10, 6.5), sharex=True,
                gridspec_kw={'hspace': 0.15},
            )

            ax1.plot(df_prn.index, y, linewidth=1.2)
            ax1.set_ylabel(f"TECc ({prn_val})")
            ax1.set_title(f"Estação: {station} | PRN: {prn_val}")
            ax1.grid(True, linestyle=":", linewidth=0.6)

            ax2.plot(df_prn.index, y_cv, linewidth=1.2)
            ax2.set_xlabel("Tempo")
            ax2.set_ylabel("TEC s/ Tendência")
            ax2.set_ylim(-1, 1)
            ax2.grid(True, linestyle=":", linewidth=0.6)

            # ax3.plot(df_prn.index, y_bw, linewidth=1.2)
            # ax3.set_xlabel("Tempo")
            # ax3.set_ylabel("Filtro Butterworth")
            # ax3.set_ylim(-0.2, 0.2)
            # ax3.grid(True, linestyle=":", linewidth=0.6)

            plt.show()

    # Recombinar todos os PRNs processados
    if processed_dfs:
        df_final = pd.concat(processed_dfs, ignore_index=True)
        # Garantir que a ordem cronológica e o agrupamento por PRN sejam mantidos
        
        df_final = df_final[(df_final[dt_col] >= dt_in) & 
                            (df_final[dt_col] < dt_in + timedelta(days=1))]
        
        
        df_final = df_final.sort_values([dt_col, 'prn'])
        return df_final
    else:
        # Se nenhum dado válido sobrou, retorna um df vazio com a nova estrutura de colunas
        return pd.DataFrame(columns=list(df.columns) + ['dtec'])


# def plot_tec_por_receptor_de_pasta(
#     folder: Path | str,
#     pattern: str = "*.parquet",
#     show: bool = False,
# ) -> None:
#     """Apply `plot_tec_por_receptor_de_arquivo` to every parquet in `folder`.

#     Useful when there is one parquet per PRN sitting in the same directory.
#     Errors on individual files are reported and do not stop the loop.
#     """
#     folder = Path(folder)
#     for path in sorted(folder.glob(pattern)):
#         try:
#             plot_tec_por_receptor_de_arquivo(file=path, show=show)
#         except Exception as exc:
#             print(f"[ERROR] {path}: {exc}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    year, doy = 2025, 316
    dt_in = datetime.strptime(f"{year}-{doy}", "%Y-%j")

    # folder_base = Path(cv.folder_db, 'ionosphere', 'tec_pc', f'{year}', f'{doy}')
    # folder_base = cfg.tec_pc_dir(year, doy)
    # files = read_files(folder_base, 'parquet')
    stations = get_station_names(year, doy)

    for station in stations:
        df_station_tec = load_tec_with_neighbors_v2(year, doy, station)
        df_station_dtec = aplicar_filtros_dtec(df_station_tec, plot_figs=True)
        df_station_dtec.to_parquet(dtec_pc_dir(year, doy))
