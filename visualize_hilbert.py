# -*- coding: utf-8 -*-
"""
Spatio-temporal scatter of dTEC IPPs, keeping EACH receiver-PRN pair as an
independent curve (NO aggregation / NO flattening like the keogram does).

@author: hisas
"""

import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Dependências para o filtro de oscilações ---
from scipy.signal import butter, filtfilt, hilbert

# Import custom modules 
from shared.config import dtec_ipp_dir, dtec_keogram_figs_dir, storms_SouthAmerica
from shared.storm_tracking import plot_storms_tracked
from shared.lightning_spectrogram import (
    lfr_spectrogram, 
    WAV_FREQ_BOUNDS,
    load_lightning_lfr  # Importamos a função que realmente existe
)


# ==========================================================================
# 0b. Dedicated output folder for this script's figures
# ==========================================================================
def ttd_figs_dir() -> Path:
    folder = Path(dtec_keogram_figs_dir()).parent / "keograms_TTDs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ==========================================================================
# 0. Data-schema assumptions
# ==========================================================================
TIME_COL = "horario"     # datetime  (parquet index)
LAT_COL  = "lts"         # IPP latitude  (deg)
LON_COL  = "lns"         # IPP longitude (deg)
PRN_COL  = "prn"         # satellite id column inside each parquet


# ==========================================================================
# 1. Load + spatio-temporal cut
# ==========================================================================
def _station_from_path(path: Path) -> str:
    return path.name.split("_dtecIPP_")[0]

def _detect_dtec_col(file: str, candidates: list[str] | None = None) -> str:
    if candidates is None:
        candidates = ["dtec_5min", "dtec_10min", "dtec_15min", "dtec_30min"]

    con = duckdb.connect()
    available = set(con.execute(
        "SELECT * FROM read_parquet($file) LIMIT 0",
        {"file": file},
    ).df().columns)
    con.close()

    for col in candidates:
        if col in available:
            return col

    dtec_cols = sorted(c for c in available if c.startswith("dtec_"))
    if dtec_cols:
        return dtec_cols[0]

    raise ValueError(f"No dtec column found in {file}.")

def load_window(
    year: int, doy: int, lat_range: tuple, lon_range: tuple, time_range: tuple,
    dtec_col: str | None = None, stations: list | None = None,
) -> pd.DataFrame:
    ipp_folder = Path(dtec_ipp_dir(year, doy))
    files = sorted(ipp_folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    if not files:
        raise FileNotFoundError(f"No IPP files found in {ipp_folder}")
    if stations is not None:
        wanted = set(stations)
        files = [f for f in files if _station_from_path(f) in wanted]

    if dtec_col is None:
        dtec_col = _detect_dtec_col(str(files[0]))
        print(f"[load] dtec column auto-detected: '{dtec_col}'")

    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    file_list = [str(f) for f in files]

    con = duckdb.connect()
    query = f"""
        SELECT
            "{TIME_COL}"            AS {TIME_COL},
            "{LAT_COL}"             AS {LAT_COL},
            "{LON_COL}"             AS {LON_COL},
            "{PRN_COL}"             AS {PRN_COL},
            "{dtec_col}"            AS {dtec_col},
            regexp_extract(filename, '([^/\\\\]+)_dtecIPP_', 1) AS station
        FROM read_parquet($files, filename = true)
        WHERE "{LAT_COL}" BETWEEN $lat_lo AND $lat_hi
          AND "{LON_COL}" BETWEEN $lon_lo AND $lon_hi
          AND "{LAT_COL}"  IS NOT NULL
          AND "{LON_COL}"  IS NOT NULL
          AND "{dtec_col}" IS NOT NULL
    """
    out = con.execute(query, {
        "files":  file_list,
        "lat_lo": lat_lo, "lat_hi": lat_hi,
        "lon_lo": lon_lo, "lon_hi": lon_hi,
    }).df()
    con.close()

    if out.empty:
        raise ValueError("No IPPs fall inside the requested lat/lon box.")

    t = pd.to_datetime(out[TIME_COL])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    out[TIME_COL] = t
    out["t_hours"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0
    h0, h1 = time_range
    out = out[out["t_hours"].between(h0, h1)]
    if out.empty:
        raise ValueError("No IPPs fall inside the requested time window.")

    out = out.reset_index(drop=True)
    out["pair"] = out["station"].astype(str) + " · " + out[PRN_COL].astype(str)
    return out


# ==========================================================================
# 2. Pair selection (Original & Hilbert)
# ==========================================================================
def select_pairs(
    df: pd.DataFrame, mode: str = "topN", n: int = 12,
    pairs: list | None = None, min_points: int = 5,
) -> list:
    counts = df.groupby("pair").size().sort_values(ascending=False)
    counts = counts[counts >= min_points]

    if mode == "all":
        chosen = counts.index.tolist()
    elif mode == "topN":
        chosen = counts.head(n).index.tolist()
    elif mode == "list":
        norm = [f"{p[0]} · PRN {p[1]}" if isinstance(p, (tuple, list)) else str(p) for p in pairs]
        chosen = [p for p in norm if p in set(df["pair"])]
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")

    if not chosen:
        raise ValueError("No pairs survived the selection criteria.")
    print(f"[select] mode={mode} -> {len(chosen)} pair(s)")
    return chosen

def select_pairs_by_hilbert(
    df: pd.DataFrame, dtec_col: str, threshold: float,
    sample_rate_sec: float = 30.0, min_period_min: float = 10.0,
    max_period_min: float = 60.0, min_points: int = 30
) -> list:
    selected_pairs = []
    fs = 1.0 / sample_rate_sec
    nyq = 0.5 * fs
    low = (1.0 / (max_period_min * 60.0)) / nyq
    high = (1.0 / (min_period_min * 60.0)) / nyq
    b, a = butter(4, [low, high], btype='band')

    # --- TOLERÂNCIA ZERO PARA BURACOS ---
    expected_step_hours = sample_rate_sec / 3600.0
    # 1.1 garante que só passam variações de arredondamento. 
    # Se faltar 1 ponto, o gap vai para 2.0, sendo barrado imediatamente.
    gap_tolerance = expected_step_hours * 1.1 

    unique_pairs = df["pair"].unique()
    for pair in unique_pairs:
        sub = df[df["pair"] == pair].sort_values("t_hours")
        y = sub[dtec_col].to_numpy()
        x = sub['t_hours'].to_numpy()
        
        if len(y) < min_points:
            continue
            
        # ========================================================
        # DETECÇÃO RESTRITA DE BURACOS
        # ========================================================
        time_diffs = np.diff(x)
        if len(time_diffs) > 0:
            max_gap = np.max(time_diffs)
            if max_gap > gap_tolerance:
                # Opcional: print para confirmar que pegou a falha
                # print(f"[gap strict] {pair} descartado. Gap máx: {max_gap*3600:.1f}s")
                continue 
        # ========================================================

        try:
            y_filt = filtfilt(b, a, y)
            amplitude_envelope = np.abs(hilbert(y_filt))
            max_envelope = np.max(amplitude_envelope)
            
            if max_envelope > threshold:
                selected_pairs.append(pair)
                
            # --- Criação do Gráfico ---
            if True:
                print(f'Saving {pair}')
                plt.figure(figsize=(8, 4))
                
                plt.scatter(x, y_filt, color='gray', s=3, label='Sinal Filtrado')
                plt.plot(x, amplitude_envelope, color='red', label='Envelope', linewidth=1.)
                
                plt.xlim(TIME_RANGE[0], TIME_RANGE[1])
                plt.ylim(-0.5, 0.5)
                
                plt.title(f'{pair} amp.max: {max_envelope:.4f} Date: {tyear}{tdoy} {TIME_RANGE[0]}-{TIME_RANGE[1]}', fontsize=12)
                
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                # --- SALVAR NO DISCO ---
                # 1. Define o caminho da pasta
                pasta_destino = f'figures_overview/{tyear}{tdoy}_{TIME_RANGE[0]}{TIME_RANGE[1]}'
                
                # 2. Cria a pasta (e subpastas) se ela não existir. 
                # O 'exist_ok=True' impede que dê erro se a pasta já tiver sido criada na run anterior.
                os.makedirs(pasta_destino, exist_ok=True)
                
                # 3. Define o caminho completo do arquivo
                nome_arquivo = f'{pasta_destino}/run_{tyear}{tdoy}{TIME_RANGE[0]}{TIME_RANGE[1]}_{pair}.png'
                # nome_arquivo = f'figures_overview/{tyear}{tdoy}/run_{tyear}{tdoy}{pair}.png' 
                # Se estiver usando a pasta: nome_arquivo = f'graficos_runs/run_{pair}.png'
                
                # Salva a figura com uma boa resolução (dpi=150)
                plt.savefig(nome_arquivo, dpi=100)
                
                # FECHA A FIGURA PARA LIBERAR MEMÓRIA RAM! (Essencial dentro de um loop)
                plt.close()
                pass
                
        except ValueError:
            continue
            
    print(f"[hilbert filter] {len(selected_pairs)} pares encontrados com oscilação > {threshold} TECU.")
    if not selected_pairs:
        raise ValueError("Nenhum par sobreviveu ao filtro de Hilbert com este threshold.")
    return selected_pairs

# def select_pairs_by_hilbert(
#     df: pd.DataFrame, dtec_col: str, threshold: float,
#     sample_rate_sec: float = 30.0, min_period_min: float = 10.0,
#     max_period_min: float = 60.0, min_points: int = 30
# ) -> list:
#     selected_pairs = []
#     fs = 1.0 / sample_rate_sec
#     nyq = 0.5 * fs
#     low = (1.0 / (max_period_min * 60.0)) / nyq
#     high = (1.0 / (min_period_min * 60.0)) / nyq
#     b, a = butter(4, [low, high], btype='band')

#     unique_pairs = df["pair"].unique()
#     for pair in unique_pairs:
#         sub = df[df["pair"] == pair].sort_values("t_hours")
#         y = sub[dtec_col].to_numpy()
#         x = sub['t_hours'].to_numpy()
#         if len(y) < min_points:
#             continue
#         try:
#             y_filt = filtfilt(b, a, y)
#             amplitude_envelope = np.abs(hilbert(y_filt))
#             max_envelope = np.max(amplitude_envelope)
#             if max_envelope > threshold:
#                 selected_pairs.append(pair)
                
            
#         except ValueError:
#             continue
            
#     print(f"[hilbert filter] {len(selected_pairs)} pares encontrados com oscilação > {threshold} TECU.")
#     if not selected_pairs:
#         raise ValueError("Nenhum par sobreviveu ao filtro de Hilbert com este threshold.")
#     return selected_pairs


# ==========================================================================
# 3. Plotting Functions (Lines, Trajectory, and Combined)
# ==========================================================================

# --- 3A. Simple Line Stack ---
def plot_pairs_lines(
    df: pd.DataFrame, chosen_pairs: list, year: int, doy: int,
    dtec_col: str = "dtec_15min", offset: float = 1.0, trh: float | None = None,
    cmap_name: str = "tab20", lw: float = 1.0, out_path: Path | None = None, dpi: int = 300,
):
    movavg = int(re.findall(r"\d+", dtec_col)[0]) if re.search(r"\d", dtec_col) else None
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(chosen_pairs))]

    fig_h = max(6.0, 0.45 * len(chosen_pairs) + 3.0)
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)

    baselines = []
    for i, (color, pair) in enumerate(zip(colors, chosen_pairs)):
        sub = df[df["pair"] == pair].sort_values("t_hours")
        y = sub[dtec_col].astype(float)
        if trh is not None:
            y = y.clip(lower=-trh, upper=trh)
        baseline = i * offset
        baselines.append(baseline)
        ax.axhline(baseline, color="0.85", linewidth=0.6, zorder=1)
        ax.plot(sub["t_hours"], y + baseline, color=color, linewidth=lw, zorder=3)

    ax.set_yticks(baselines)
    ax.set_yticklabels(chosen_pairs, fontsize=8)
    bot_pad = 0.6 * offset
    top_pad = 1.4 * offset
    ax.set_ylim(baselines[0] - bot_pad, baselines[-1] + top_pad)

    sel = df["pair"].isin(chosen_pairs)
    h0 = float(df.loc[sel, "t_hours"].min())
    h1 = float(df.loc[sel, "t_hours"].max())
    ax.set_xlim(h0, h1)
    ax.set_xlabel("Time (UTC, hours)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)

    span = h1 - h0
    x_bar = h0 + 0.015 * span
    cap = 0.004 * span
    y_bot = baselines[-1] + 0.25 * offset
    y_top = y_bot + offset
    ax.plot([x_bar, x_bar], [y_bot, y_top], color="0.15", linewidth=2.0, zorder=4)
    ax.text(x_bar + 2 * cap, (y_bot + y_top) / 2, f"{offset:g} TECU", fontsize=8, va="center", color="0.15")

    title = f"dTEC line stack by receiver-PRN — {year} DOY {doy:03d}  |  {len(chosen_pairs)} pair(s)"
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path


# --- 3B. Trajectory (Original Spatial View) ---
def plot_pairs_trajectory(
    df: pd.DataFrame, chosen_pairs: list, year: int, doy: int,
    dtec_col: str = "dtec_15min", trh: float | None = None, cmap: str = "seismic_r", s: float = 12.0, 
    plot_storms: bool = True, add_spectrogram: bool = True, lat_range: tuple | None = None, 
    lon_range: tuple | None = None, time_range: tuple | None = None,
    lightning_lat_range: tuple | None = None, lightning_lon_range: tuple | None = None,
    out_path: Path | None = None, dpi: int = 300,
):
    eff_lat_range  = lat_range
    eff_lon_range  = lon_range
    eff_time_range = time_range
    eff_lightning_lat_range = lightning_lat_range if lightning_lat_range is not None else eff_lat_range
    eff_lightning_lon_range = lightning_lon_range if lightning_lon_range is not None else eff_lon_range

    sub = df.loc[df["pair"].isin(chosen_pairs)]
    vmax = trh if trh is not None else float(np.nanmax(np.abs(sub[dtec_col])))
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    storms_data = None
    if plot_storms:
        df_storms = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")
        t0 = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        t1 = t0 + pd.Timedelta(days=1)
        df_storms = df_storms[(df_storms['datetime'] >= t0) & (df_storms['datetime'] < t1)].copy()
        
        lat_lo, lat_hi = sorted(eff_lat_range)
        lon_lo, lon_hi = sorted(eff_lon_range)
        df_storms = df_storms[
            df_storms['centroid_lat'].between(lat_lo - 3, lat_hi + 3) &
            df_storms['centroid_lon'].between(lon_lo - 3, lon_hi + 3)
        ].copy()

        if not df_storms.empty:
            dt = pd.to_datetime(df_storms['datetime']).dt.tz_convert(None)
            t_hours = (dt - t0).dt.total_seconds().to_numpy() / 3600.0
            storms_data = dict(t=t_hours, lon_c=df_storms['centroid_lon'].to_numpy(), lat_c=df_storms['centroid_lat'].to_numpy())

    if add_spectrogram:
        fig, (ax_lon, ax_lat, ax_spec) = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    else:
        fig, (ax_lon, ax_lat) = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True, sharex=True)
        ax_spec = None

    absdtec = np.abs(sub[dtec_col])
    w = np.clip(absdtec / vmax, 0.0, 1.0)
    alpha = 0.10 + 0.90 * (w ** 2.0)
    s = absdtec * 25

    ax_lon.scatter(sub["t_hours"], sub[LON_COL], c=sub[dtec_col], s=s, cmap=cmap, norm=norm, edgecolors="none", alpha=alpha)
    ax_lat.scatter(sub["t_hours"], sub[LAT_COL], c=sub[dtec_col], s=s, cmap=cmap, norm=norm, edgecolors="none", alpha=alpha)

    if storms_data is not None:
            ax_lon.scatter(storms_data["t"], storms_data["lon_c"], marker="o", s=35, facecolors="cyan", edgecolors="black", zorder=4)
            ax_lat.scatter(storms_data["t"], storms_data["lat_c"], marker="o", s=35, facecolors="cyan", edgecolors="black", zorder=4)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    fig.colorbar(sm, ax=(ax_lon, ax_lat), location="right", shrink=1.0, pad=0.02, label="dTEC (TECU)")

    ax_lon.set_ylabel("IPP longitude, deg")
    ax_lat.set_ylabel("IPP latitude, deg")

    if add_spectrogram:
        spec_result = lfr_spectrogram(year, doy, lat_range=eff_lightning_lat_range, lon_range=eff_lightning_lon_range, time_range=eff_time_range)
        if spec_result is not None:
            t_hours_spec, freq, power = spec_result
            cf = ax_spec.contourf(t_hours_spec, freq, power, cmap="nipy_spectral", levels=20)
            fig.colorbar(cf, ax=ax_spec, location="right", shrink=1.0, pad=0.02, label="log10(power)")
        ax_spec.set_ylabel("Frequency (mHz)")

    bottom_ax = ax_spec if add_spectrogram else ax_lat
    bottom_ax.set_xlabel("Time (UTC, hours)")
    bottom_ax.set_xlim(float(sub["t_hours"].min()), float(sub["t_hours"].max()))
    
    axes_all = (ax_lon, ax_lat, ax_spec) if add_spectrogram else (ax_lon, ax_lat)
    for ax in axes_all:
        ax.set_axisbelow(False)
        ax.grid(True, linestyle=":", alpha=1., zorder=5)
        
    ax_lon.set_title(f"TRAVEL TIME DIAGRAM: {year}-{doy:03d} | {len(chosen_pairs)} pair(s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path


# --- 3C. New 3-Panel Plot (Line Stack + LFR + Wavelet) ---
def plot_combined_panels(
    df: pd.DataFrame, chosen_pairs: list, year: int, doy: int,
    dtec_col: str = "dtec_15min", offset: float = 1.0, cmap_name: str = "tab20", 
    lat_range: tuple | None = None, lon_range: tuple | None = None, 
    time_range: tuple | None = None,
    lightning_lat_range: tuple | None = None, lightning_lon_range: tuple | None = None,
    out_path: Path | None = None, dpi: int = 300,
):
    eff_lat_range  = lat_range
    eff_lon_range  = lon_range
    eff_time_range = time_range
    eff_lightning_lat_range = lightning_lat_range if lightning_lat_range is not None else eff_lat_range
    eff_lightning_lon_range = lightning_lon_range if lightning_lon_range is not None else eff_lon_range

    dtec_height = max(5.0, 0.45 * len(chosen_pairs))
    fig_h = dtec_height + 6.0  
    
    fig, (ax_dtec, ax_lfr, ax_spec) = plt.subplots(
        3, 1, figsize=(14, fig_h), sharex=True, 
        gridspec_kw={"height_ratios": [dtec_height, 3, 3]},
        constrained_layout=True
    )

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(chosen_pairs))]
    
    baselines = []
    for i, (color, pair) in enumerate(zip(colors, chosen_pairs)):
        sub = df[df["pair"] == pair].sort_values("t_hours")
        y = sub[dtec_col].astype(float)
        baseline = i * offset
        baselines.append(baseline)
        
        ax_dtec.axhline(baseline, color="0.85", linewidth=0.6, zorder=1)
        ax_dtec.plot(sub["t_hours"], y + baseline, color=color, linewidth=1.2, zorder=3)

    ax_dtec.set_yticks(baselines)
    ax_dtec.set_yticklabels(chosen_pairs, fontsize=8)
    
    bot_pad = 0.6 * offset
    top_pad = 1.4 * offset
    ax_dtec.set_ylim(baselines[0] - bot_pad, baselines[-1] + top_pad)
    ax_dtec.set_ylabel("Receiver-PRN (dTEC)")
    
    movavg = int(re.findall(r"\d+", dtec_col)[0]) if re.search(r"\d", dtec_col) else None
    title = f"Filtered Perturbations — {year} DOY {doy:03d} | {len(chosen_pairs)} pair(s) | offset {offset:g} TECU"
    ax_dtec.set_title(title)

    sel = df["pair"].isin(chosen_pairs)
    h0 = float(df.loc[sel, "t_hours"].min())
    h1 = float(df.loc[sel, "t_hours"].max())
    span = h1 - h0
    x_bar = h0 + 0.015 * span
    cap = 0.004 * span
    y_bot = baselines[-1] + 0.25 * offset
    y_top = y_bot + offset
    ax_dtec.plot([x_bar, x_bar], [y_bot, y_top], color="0.15", linewidth=2.0, zorder=4)
    ax_dtec.text(x_bar + 2 * cap, (y_bot + y_top) / 2, f"{offset:g} TECU", fontsize=8, va="center", color="0.15")

    try:
        # Puxamos os dados da função existente em lightning_spectrogram.py
        lfr_series = load_lightning_lfr(
            year, doy, lat_range=eff_lightning_lat_range, lon_range=eff_lightning_lon_range, time_range=eff_time_range
        )
        
        if lfr_series is not None:
            # Converte a taxa de flashes/s para flashes/minuto
            lfr_rate = lfr_series.values * 60.0
            
            # Recria o eixo de tempo (horas decimais), baseado na taxa de amostragem de 15s
            h0_lfr, _ = eff_time_range
            t_lfr = h0_lfr + np.arange(len(lfr_rate)) * 15.0 / 3600.0
            
            ax_lfr.plot(t_lfr, lfr_rate, color="black", linewidth=1.0)
            ax_lfr.fill_between(t_lfr, 0, lfr_rate, color="orange", alpha=0.4)
        else:
            ax_lfr.text(0.5, 0.5, "No lightning data for timeseries", transform=ax_lfr.transAxes, ha="center", va="center", color="red")
            
    except Exception as e:
        ax_lfr.text(0.5, 0.5, f"LFR timeseries error / missing data\n({e})", transform=ax_lfr.transAxes, ha="center", va="center", color="red")
        
    ax_lfr.set_ylabel("LFR (flashes/min)")
    ax_lfr.set_ylim(bottom=0)

    spec_result = lfr_spectrogram(
        year, doy, lat_range=eff_lightning_lat_range, lon_range=eff_lightning_lon_range, time_range=eff_time_range
    )
    if spec_result is not None:
        t_hours_spec, freq, power = spec_result
        cf = ax_spec.contourf(t_hours_spec, freq, power, cmap="nipy_spectral", levels=20)
        fig.colorbar(cf, ax=ax_spec, location="right", shrink=1.0, pad=0.02, label="log10(power)")
        ax_spec.set_title(f"Wavelet power — Lightning Flash Rate (Morlet, {WAV_FREQ_BOUNDS[0]}–{WAV_FREQ_BOUNDS[1]} mHz)", fontsize=9)
    else:
        ax_spec.text(0.5, 0.5, "No lightning data for wavelet", transform=ax_spec.transAxes, ha="center", va="center")
        
    ax_spec.set_ylabel("Frequency (mHz)")
    ax_spec.set_xlabel("Time (UTC, hours)")

    ax_spec.set_xlim(h0, h1)
    for ax in (ax_dtec, ax_lfr, ax_spec):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.grid(True, which="major", linestyle=":", alpha=0.6, zorder=0)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3, zorder=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path


# ==========================================================================
# 4. CLI parsing
# ==========================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Scatter dTEC IPPs & Lightning correlation.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy",  type=int, required=True)
    parser.add_argument("--lat",  type=float, nargs=2, required=True)
    parser.add_argument("--lon",  type=float, nargs=2, required=True)
    parser.add_argument("--time", type=float, nargs=2, default=(0.0, 24.0))
    parser.add_argument("--dtec-col", type=str, default="dtec_15min")
    parser.add_argument("--mode", choices=["all", "topN", "list", "hilbert"], default="topN")
    
    # --- Nova seleção de figura por CLI ---
    parser.add_argument("--kind", choices=["combined", "trajectory", "timeseries"], default="combined",
                        help="Tipo de figura a ser gerada.")
    
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--trh", type=float, default=None)
    parser.add_argument("--hilbert-threshold", type=float, default=0.15)
    parser.add_argument("--offset", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--min-period", type=float, default=10.0)
    parser.add_argument("--max-period", type=float, default=60.0)
    return parser.parse_args()


# ==========================================================================
# 5. Entry point
# ==========================================================================
def run(
    year, doy, lat_range, lon_range, time_range,
    dtec_col=None, kind="combined", offset=1.0, 
    mode="topN", n=12, pairs=None, min_points=5, trh=None, stations=None,
    add_spectrogram=True, lightning_lat_range=None, lightning_lon_range=None, out_path=None,
    hilbert_threshold=0.15, sample_rate_sec=30.0, min_period_min=10.0, max_period_min=60.0
):
    
    # ======================================================================
    # NOMEAÇÃO DINÂMICA SEGURA DO ARQUIVO (Prevenção contra sobrescrita)
    # ======================================================================
    if out_path is None:
        t0, t1 = time_range
        file_name = f"{kind}_{year}-{doy:03d}_{t0}-{t1}_{mode}.png"
        out_path = ttd_figs_dir() / file_name
        
    # Prevenção: verifica e avisa antes de rodar queries se já existir
    if Path(out_path).exists():
        print(f"[skip] {Path(out_path).name} already exists. To regenerate, delete the file first.")
        return out_path
        
    df = load_window(year, doy, lat_range=lat_range, lon_range=lon_range, time_range=time_range, dtec_col=dtec_col, stations=stations)
    dtec_col = next(c for c in df.columns if c.startswith("dtec_"))

    print(f"[load] {len(df)} IPPs in window | {df['station'].nunique()} stations | {df['pair'].nunique()} pairs | col='{dtec_col}'")

    if mode == "hilbert":
        chosen = select_pairs_by_hilbert(
            df=df, dtec_col=dtec_col, threshold=hilbert_threshold,
            sample_rate_sec=sample_rate_sec, min_period_min=min_period_min,
            max_period_min=max_period_min, min_points=max(min_points, 30)
        )
    else:
        chosen = select_pairs(df, mode=mode, n=n, pairs=pairs, min_points=min_points)

    # =======================================================================
    # DESCARTA SINAIS FRACOS, EXTREMOS (ANOMALIAS) E SÉRIES COM BURACOS
    # =======================================================================
    if trh is not None:
        valid_pairs = []
        UPPER_LIMIT = 1.0  # Limite máximo aceitável em TECU
        
        # Converte a taxa de amostragem esperada para horas (mesma unidade do t_hours)
        expected_step_hours = sample_rate_sec / 3600.0
        # Margem de tolerância (ex: se for 30s, aceitamos até 45s de gap por imprecisão do relógio)
        gap_tolerance = expected_step_hours * 1.5 
        
        for pair, group in df[df["pair"].isin(chosen)].groupby("pair"):
            # 1. Verifica amplitude do sinal (mínimo e máximo)
            max_val = group[dtec_col].abs().max()
            atingiu_minimo = max_val >= trh
            abaixo_do_limite = max_val <= UPPER_LIMIT
            
            # 2. Verifica se a série tem buracos no tempo
            group_sorted = group.sort_values("t_hours")
            time_diffs = group_sorted["t_hours"].diff().dropna()
            
            if len(time_diffs) == 0:
                continue
                
            max_gap = time_diffs.max()
            tem_buraco = max_gap > gap_tolerance
            
            if atingiu_minimo and abaixo_do_limite and not tem_buraco:
                valid_pairs.append(pair)
            elif tem_buraco:
                print(f"[gap drop] Par {pair} removido. Buraco detectado: {max_gap*3600:.1f}s (esperado ~{sample_rate_sec}s)")
            elif not abaixo_do_limite:
                print(f"[upper limit drop] Par {pair} removido. Pico máximo de {max_val:.2f} TECU excedeu o limite de {UPPER_LIMIT} TECU.")
        
        # Atualiza a lista mantendo apenas os pares que passaram em todos os testes
        chosen = [p for p in chosen if p in valid_pairs]
        
        print(f"[cutoff] {len(chosen)} par(es) mantido(s) contínuos e dentro da faixa [{trh} - {UPPER_LIMIT}] TECU.")
        
        if not chosen:
            print("Nenhum par sobreviveu aos filtros. O gráfico não será gerado.")
            return None
    # =======================================================================

    # --- Roteamento para a função de plot correta baseada no 'kind' ---
    if kind == "combined":
        return plot_combined_panels(
            df, chosen, year=year, doy=doy, dtec_col=dtec_col, offset=offset,
            lat_range=lat_range, lon_range=lon_range, time_range=time_range, 
            lightning_lat_range=lightning_lat_range, lightning_lon_range=lightning_lon_range, 
            out_path=out_path
        )
    elif kind == "trajectory":
        return plot_pairs_trajectory(
            df, chosen, year=year, doy=doy, dtec_col=dtec_col, trh=trh,
            add_spectrogram=add_spectrogram, lat_range=lat_range, lon_range=lon_range, time_range=time_range,
            lightning_lat_range=lightning_lat_range, lightning_lon_range=lightning_lon_range,
            out_path=out_path
        )
    elif kind == "timeseries":
        return plot_pairs_lines(
            df, chosen, year=year, doy=doy, dtec_col=dtec_col, offset=offset, trh=trh, out_path=out_path
        )
    else:
        raise ValueError(f"Unknown kind: {kind!r}")


# ==========================================================================
# 6. IDE / CLI entry point
# ==========================================================================
if __name__ == "__main__":
    RUN_MANUALLY = True
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")

        # runs = [
        #     # (2021, 296, 14, 19),
        #     (2022, 279, 18, 22),
        #     (2022,  97, 17, 20),
        #     (2023, 347, 8, 10),
        #     (2023, 347,  15, 17),
        #     (2023, 347,  17, 20),
        #     (2023, 348,  17, 21),
        #     # (2023, 348, 18, 22),
        #     (2024, 264,  11, 13),
        #     (2024, 264,  13, 15),
        #     (2024, 264, 16, 18),
        #     (2024, 264, 18,  21),
        #     # (2024, 264,  9, 22),
        #     (2024, 333,  10, 12),
        #     (2024, 333,  12, 14),
        #     (2024, 333,  14, 16),
        #     (2024, 333,  16, 18),
        #     # (2025, 298,  7, 12),
        #     # (2025, 298, 18, 21),
        # ]
        
        runs = [
            # (2021, 296, 14, 19),
            (2022, 279, 18, 22),
            # (2022,  97, 17, 20),
            # (2023, 347, 8, 10),
            # (2023, 347,  15, 17),
            # (2023, 347,  17, 20),
            (2023, 348,  17, 21),
            # (2023, 348, 18, 22),
            # (2024, 264,  11, 13),
            # (2024, 264,  13, 15),
            # (2024, 264, 16, 18),
            # (2024, 264, 18,  21),
            # (2024, 264,  9, 22),
            # (2024, 333,  10, 12),
            # (2024, 333,  12, 14),
            (2024, 333,  13, 17),
            # (2024, 333,  16, 18),
            # (2025, 298,  7, 12),
            # (2025, 298, 18, 21),
        ]

        LAT_RANGE = (-32.01, -17.99)    
        LON_RANGE = (-62.01, -47.99)    
        LIGHTNING_LAT_RANGE = None    
        LIGHTNING_LON_RANGE = None    

        # =========================================================
        # ->> CONTROLE PRINCIPAL DE FIGURA E FILTRO <<-
        # =========================================================
        FIG_KIND   = "combined"   # OPÇÕES: "combined" | "trajectory" | "timeseries"
        FILTER_MODE = "hilbert"   # OPÇÕES: "hilbert" | "topN" | "all" | "list"
        
        for tyear, tdoy, tbeg, tend in runs:
            TIME_RANGE = (tbeg, tend)
            
            run(
                year=tyear, doy=tdoy,
                lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
                dtec_col=None,          
                kind=FIG_KIND,           
                mode=FILTER_MODE,        
                hilbert_threshold=0.05,  
                offset=0.5,              
                trh=0.3501,               
                sample_rate_sec=15.0,
                min_period_min=1.2,
                max_period_min=10.0,
                min_points=120,
                add_spectrogram=True,
                lightning_lat_range=LIGHTNING_LAT_RANGE,
                lightning_lon_range=LIGHTNING_LON_RANGE,
                out_path=None,           
            )
    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        run(
            year=args.year, doy=args.doy, lat_range=tuple(args.lat), lon_range=tuple(args.lon),
            time_range=tuple(args.time), dtec_col=getattr(args, "dtec_col", None),
            kind=args.kind, mode=args.mode, n=args.n, offset=args.offset, trh=args.trh,
            hilbert_threshold=args.hilbert_threshold,
            sample_rate_sec=args.sample_rate,
            min_period_min=args.min_period,
            max_period_min=args.max_period
        )