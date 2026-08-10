# -*- coding: utf-8 -*-
"""
Spatio-temporal scatter of dTEC IPPs & Spectral Analysis (Wavelet + Cross-Spectrum)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, filtfilt, hilbert, csd

# ==========================================================================
# IMPORTAÇÕES DO SEU PROJETO
# ==========================================================================
from shared.config import dtec_ipp_dir
from shared.lightning_spectrogram import load_lightning_lfr
from shared.waveletFunctions import wavelet_on_data_new, wd_slicer


# ==========================================================================
# 0. BLOCO CENTRAL DE CONSTANTES E CONFIGURAÇÕES
# ==========================================================================
# Parâmetros de Esquema e Dados
TIME_COL = "horario"     # datetime (parquet index)
LAT_COL  = "lts"         # IPP latitude (deg)
LON_COL  = "lns"         # IPP longitude (deg)
PRN_COL  = "prn"         # ID do satélite

# Configurações de Diretórios e Amostragem de Relâmpagos
LIGHTNING_DIR = Path("D:/database/lightning/parquet_15s")
LFR_SAMPLING_RATE = 15   # segundos

# Parâmetros da Wavelet (exatos do lfr_explorer.py / f2_spec.py)
WAV_DT     = 0.25    # unidade de tempo = MINUTOS (15s / 60 = 0.25 min/sample)
WAV_PAD    = 1
WAV_DJ     = 0.15
WAV_S0     = 0.90
WAV_J1     = 24
WAV_LAG1   = 0.71
WAV_MOTHER = "MORLET"
WAV_FREQ_BOUNDS = [2, 7]    # mHz, passado para o wd_slicer

# Filtro Passa-Faixa (Butterworth, zero-phase via filtfilt) para o LFR
BANDPASS_LOW_MHZ  = 1    # mHz
BANDPASS_HIGH_MHZ = 12   # mHz
BANDPASS_ORDER    = 4

# Parâmetros gerais de Filtro e Amostragem GNSS
GNSS_SAMPLE_RATE_SEC = 15.0
MIN_PERIOD_MIN = 1.2
MAX_PERIOD_MIN = 10.0
HILBERT_THRESHOLD = 0.05
MIN_POINTS = 240

# Recortes Espaciais Padrão
LAT_RANGE = (-32.01, -17.99)    
LON_RANGE = (-62.01, -47.99)    
LIGHTNING_LAT_RANGE = None    
LIGHTNING_LON_RANGE = None    

# Diretório de Saída de Figuras
FIGURES_OUTPUT_DIR = "figures_spectra"


# ==========================================================================
# 1. FUNÇÕES DE SUPORTE E CARREGAMENTO
# ==========================================================================
def _bandpass_filter(data, fs, low_hz, high_hz, order=4):
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def _station_from_path(path: Path) -> str:
    return path.name.split("_dtecIPP_")[0]

def _detect_dtec_col(file: str, candidates: list[str] | None = None) -> str:
    if candidates is None:
        candidates = ["dtec_5min", "dtec_10min", "dtec_15min", "dtec_30min"]
    con = duckdb.connect()
    available = set(con.execute("SELECT * FROM read_parquet($file) LIMIT 0", {"file": file}).df().columns)
    con.close()
    for col in candidates:
        if col in available: return col
    dtec_cols = sorted(c for c in available if c.startswith("dtec_"))
    if dtec_cols: return dtec_cols[0]
    raise ValueError(f"No dtec column found in {file}.")

def load_gnss_window(
    year: int, doy: int, lat_range: tuple, lon_range: tuple, time_range: tuple,
    dtec_col: str | None = None, stations: list | None = None,
) -> pd.DataFrame:
    ipp_folder = Path(dtec_ipp_dir(year, doy))
    files = sorted(ipp_folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    if not files: raise FileNotFoundError(f"No IPP files found in {ipp_folder}")
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

    if out.empty: raise ValueError("No IPPs fall inside the requested lat/lon box.")

    t = pd.to_datetime(out[TIME_COL])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    out[TIME_COL] = t
    out["t_hours"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0
    h0, h1 = time_range
    out = out[out["t_hours"].between(h0, h1)]
    if out.empty: raise ValueError("No IPPs fall inside the requested time window.")

    out = out.reset_index(drop=True)
    out["pair"] = out["station"].astype(str) + " · " + out[PRN_COL].astype(str)
    return out

def prepare_lightning_data(year, doy, eff_lightning_lat, eff_lightning_lon, TIME_RANGE):
    """Carrega LFR apenas uma vez para manter em memória."""
    lfr_data = {"t_lfr": None, "lfr_rate": None}
    try:
        lfr_series = load_lightning_lfr(
            year, doy, lat_range=eff_lightning_lat, 
            lon_range=eff_lightning_lon, time_range=TIME_RANGE
        )
        if lfr_series is not None:
            lfr_data["lfr_rate"] = lfr_series.values * 60.0 # converte para flashes/min
            lfr_data["t_lfr"] = TIME_RANGE[0] + np.arange(len(lfr_data["lfr_rate"])) * LFR_SAMPLING_RATE / 3600.0
            print("[load] LFR data successfully loaded into memory.")
    except Exception as e:
        print(f"[aviso] Falha ao carregar LFR: {e}")
    return lfr_data


# ==========================================================================
# 2. PROCESSAMENTO DE SINAIS (Wavelet com parâmetros exatos)
# ==========================================================================
def compute_wavelet_power(signal: np.ndarray, dt_minutes: float, is_lfr: bool = False):
    """
    Calcula a potência Wavelet (Morlet) utilizando dt em minutos e aplicando 
    filtro passa-faixa opcional para o LFR.
    """
    sig = signal.astype(float)
    
    if is_lfr:
        fs = 1.0 / (dt_minutes * 60.0) # frequência em Hz baseada em minutos
        sig = _bandpass_filter(
            sig, fs=fs,
            low_hz=BANDPASS_LOW_MHZ * 1e-3, 
            high_hz=BANDPASS_HIGH_MHZ * 1e-3,
            order=BANDPASS_ORDER,
        )

    if np.count_nonzero(sig) < 16:
        return None, None
        
    _period, freq, power = wavelet_on_data_new(
        sig, dt=dt_minutes, pad=WAV_PAD, dj=WAV_DJ,
        s0=WAV_S0, j1=WAV_J1, lag1=WAV_LAG1, mother=WAV_MOTHER,
    )
    freq, power = wd_slicer(freq, power, WAV_FREQ_BOUNDS)

    power = (power - power.min()) / (power.max() + 1e-30)
    power = np.log10(power + 5e-2)
    return freq, power


# ==========================================================================
# 3. ROTINA DE PLOTAGEM (5 Painéis com Normalização e Expansão)
# ==========================================================================
# ==========================================================================
# 3. ROTINA DE PLOTAGEM (5 Painéis com Normalização, Expansão e TIME LAG)
# ==========================================================================
def plot_spectral_panels(
    t_hours: np.ndarray, y_filt: np.ndarray, amplitude_envelope: np.ndarray, 
    pair: str, max_envelope: float, tyear: int, tdoy: int, TIME_RANGE: tuple,
    sample_rate_sec: float, lfr_data: dict  
):
    print(f'Saving {pair} - Spectral Analysis (Cross-Wavelet Normalized with Lag)')
    
    t_lfr = lfr_data.get("t_lfr")
    lfr_rate = lfr_data.get("lfr_rate")
    
    if t_lfr is None or lfr_rate is None:
        print(f"[erro] LFR data missing for {pair}. Cannot compute cross-wavelet.")
        return

    fig, axs = plt.subplots(
        5, 1, figsize=(12, 12), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 2, 2, 2]}
    )
    ax_env, ax_lfr, ax_wav_tid, ax_wav_lfr, ax_cross = axs

    # --- 1. PAINEL: ENVELOPE (Tempo) ---
    ax_env.scatter(t_hours, y_filt, color='gray', s=3, label='Sinal Filtrado', zorder=2)
    ax_env.plot(t_hours, amplitude_envelope, color='red', label='Envelope', linewidth=1.5, zorder=3)
    ax_env.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    ax_env.set_ylim(-0.5, 0.5)
    ax_env.set_ylabel("dTEC (TECU)")
    ax_env.set_title(f'{pair} amp.max: {max_envelope:.4f} TECU | Data: {tyear}-{tdoy:03d}', fontsize=12)
    ax_env.legend(loc='upper right')

    # --- 2. PAINEL: LFR (Tempo) ---
    ax_lfr.plot(t_lfr, lfr_rate, color="black", linewidth=1.0)
    ax_lfr.fill_between(t_lfr, 0, lfr_rate, color="orange", alpha=0.4)
    ax_lfr.set_ylim(bottom=0)
    ax_lfr.set_ylabel("LFR (fl/min)")
    ax_lfr.sharex(ax_env)

    # =========================================================
    # CÁLCULOS WAVELET INDEPENDENTES
    # =========================================================
    gnss_dt_min = sample_rate_sec / 60.0
    lfr_dt_min = (t_lfr[1] - t_lfr[0]) * 60.0
    
    freq_tid, power_tid_short = compute_wavelet_power(y_filt, dt_minutes=gnss_dt_min, is_lfr=False)
    freq_lfr, power_lfr = compute_wavelet_power(lfr_rate, dt_minutes=lfr_dt_min, is_lfr=True)

    # =========================================================
    # NORMALIZAÇÃO [0, 1] E EXPANSÃO DA MATRIZ GNSS (COM TIME LAG)
    # =========================================================
    power_tid_expanded = None
    power_lfr_norm = None
    
    # === AQUI ESTÁ O PARÂMETRO DO ATRASO (9 MINUTOS) ===
    LAG_MINUTES = 9.0  
    
    if (freq_tid is not None) and (power_tid_short is not None) and (freq_lfr is not None) and (power_lfr is not None):
        
        # Normaliza LFR para [0, 1]
        p_lfr_min, p_lfr_max = np.min(power_lfr), np.max(power_lfr)
        power_lfr_norm = (power_lfr - p_lfr_min) / (p_lfr_max - p_lfr_min + 1e-10)
        
        # Normaliza GNSS (curto) para [0, 1]
        p_tid_min, p_tid_max = np.min(power_tid_short), np.max(power_tid_short)
        power_tid_short_norm = (power_tid_short - p_tid_min) / (p_tid_max - p_tid_min + 1e-10)

        # Cria matriz de zeros do tamanho exato da grade do LFR
        power_tid_expanded = np.zeros((len(freq_lfr), len(t_lfr)))
        
        # Converte minutos para horas e desloca o vetor de tempo do GNSS para o passado
        lag_hours = LAG_MINUTES / 60.0 
        t_hours_shifted = t_hours - lag_hours
        
        # Arredonda e encontra a intersecção usando o tempo já deslocado
        t_lfr_round = np.round(t_lfr, 5)
        t_hours_shifted_round = np.round(t_hours_shifted, 5)
        _, idx_lfr, idx_gnss = np.intersect1d(t_lfr_round, t_hours_shifted_round, return_indices=True)
        
        # Encaixa a matriz do GNSS no tempo correto da matriz LFR
        power_tid_expanded[:, idx_lfr] = power_tid_short_norm[:, idx_gnss]

    # --- 3. PAINEL: WAVELET TID (Matriz Expandida, Normalizada e Deslocada) ---
    if power_tid_expanded is not None:
        cf_tid = ax_wav_tid.contourf(t_lfr, freq_tid, power_tid_expanded, cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_tid, ax=ax_wav_tid, location="right", shrink=1.0, pad=0.02, label="Norm Power [0-1]")
    else:
        ax_wav_tid.text(0.5, 0.5, "Insufficient signal for TID Wavelet", transform=ax_wav_tid.transAxes, ha="center")
        
    ax_wav_tid.set_ylabel("Freq (mHz)")
    ax_wav_tid.set_title(f"Wavelet Power - TID (Lag: -{LAG_MINUTES} min)", fontsize=10)
    ax_wav_tid.sharex(ax_env)

    # --- 4. PAINEL: WAVELET LFR (Normalizado) ---
    if power_lfr_norm is not None:
        cf_lfr = ax_wav_lfr.contourf(t_lfr, freq_lfr, power_lfr_norm, cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_lfr, ax=ax_wav_lfr, location="right", shrink=1.0, pad=0.02, label="Norm Power [0-1]")
    else:
        ax_wav_lfr.text(0.5, 0.5, "Insufficient signal for LFR Wavelet", transform=ax_wav_lfr.transAxes, ha="center")
    
    ax_wav_lfr.set_ylabel("Freq (mHz)")
    ax_wav_lfr.set_title(f"Wavelet Power - LFR ({BANDPASS_LOW_MHZ}–{BANDPASS_HIGH_MHZ} mHz)", fontsize=10)
    ax_wav_lfr.sharex(ax_env)

    # --- 5. PAINEL: CROSS-WAVELET (Multiplicação) ---
    if (power_tid_expanded is not None) and (power_lfr_norm is not None):
        cross_power = power_tid_expanded * power_lfr_norm
        
        cf_cross = ax_cross.contourf(t_lfr, freq_lfr, cross_power, cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_cross, ax=ax_cross, location="right", shrink=1.0, pad=0.02, label="Cross Power [0-1]")
    else:
        ax_cross.text(0.5, 0.5, "Cross-Wavelet computation failed", transform=ax_cross.transAxes, ha="center", color="red")
        
    ax_cross.set_ylabel("Freq (mHz)")
    ax_cross.set_xlabel("Time (UTC, hours)")
    ax_cross.set_title(f"Cross-Wavelet Power (TID × LFR | {LAG_MINUTES} min lag)", fontsize=10)
    ax_cross.sharex(ax_env)

    # --- FORMATAÇÃO GERAL ---
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        # AQUI ESTÁ A MUDANÇA: 10/60 representa exatos 10 minutos em horas decimais
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 60.0)) 
        
        ax.grid(True, which="major", linestyle=":", alpha=0.6, zorder=0)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3, zorder=0)

    fig.text(0.99, 0.05, f'Script: visualize_hilbert.py\nFunction: plot_spectral_panels\nStorage: {FIGURES_OUTPUT_DIR}/', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=10, color='gray', alpha=0.7)
    
    pasta_destino = f'{FIGURES_OUTPUT_DIR}/{tyear}{tdoy}_{TIME_RANGE[0]}{TIME_RANGE[1]}'
    os.makedirs(pasta_destino, exist_ok=True)
    nome_arquivo = f'{pasta_destino}/run_{tyear}{tdoy}{TIME_RANGE[0]}{TIME_RANGE[1]}_{pair}.png'
    plt.savefig(nome_arquivo, dpi=120)
    plt.close()


# ==========================================================================
# 4. LOOP DE FILTRAGEM E ORQUESTRAÇÃO
# ==========================================================================
def filter_and_analyze_pairs(
    df: pd.DataFrame, dtec_col: str, threshold: float,
    tyear: int, tdoy: int, TIME_RANGE: tuple, lfr_data: dict,
    sample_rate_sec: float, min_period_min: float,
    max_period_min: float, min_points: int
) -> list:
    selected_pairs = []
    fs = 1.0 / sample_rate_sec
    nyq = 0.5 * fs
    low = (1.0 / (max_period_min * 60.0)) / nyq
    high = (1.0 / (min_period_min * 60.0)) / nyq
    b, a = butter(4, [low, high], btype='band')

    expected_step_hours = sample_rate_sec / 3600.0
    gap_tolerance = expected_step_hours * 1.1 

    unique_pairs = df["pair"].unique()
    for pair in unique_pairs:
        sub = df[df["pair"] == pair].sort_values("t_hours")
        y = sub[dtec_col].to_numpy()
        x = sub['t_hours'].to_numpy()
        
        if len(y) < min_points:
            continue
            
        time_diffs = np.diff(x)
        if len(time_diffs) > 0 and np.max(time_diffs) > gap_tolerance:
            continue 

        try:
            y_filt = filtfilt(b, a, y)
            amplitude_envelope = np.abs(hilbert(y_filt))
            max_envelope = np.max(amplitude_envelope)
            
            if max_envelope > threshold:
                selected_pairs.append(pair)
                
                plot_spectral_panels(
                    t_hours=x, y_filt=y_filt, amplitude_envelope=amplitude_envelope,
                    pair=pair, max_envelope=max_envelope, tyear=tyear, tdoy=tdoy, 
                    TIME_RANGE=TIME_RANGE, sample_rate_sec=sample_rate_sec, lfr_data=lfr_data
                )
                
        except ValueError:
            continue
            
    print(f"[analysis] {len(selected_pairs)} pares analisados (amp > {threshold} TECU).")
    return selected_pairs


# ==========================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    runs = [
        # (2021, 296, 14, 19),
        (2022, 279, 18.5, 20),
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

    for tyear, tdoy, tbeg, tend in runs:
        TIME_RANGE = (tbeg, tend)
        print(f"\n--- Iniciando processamento para {tyear}-{tdoy:03d} [{tbeg}h - {tend}h] ---")
        
        # 1. Carrega dados GNSS (já limitados ao recorte espaço-temporal)
        df_gnss = load_gnss_window(
            year=tyear, doy=tdoy, lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE
        )
        dtec_col = next(c for c in df_gnss.columns if c.startswith("dtec_"))
        
        # 2. Carrega e mantém dados dos relâmpagos na memória
        eff_lat = LIGHTNING_LAT_RANGE if LIGHTNING_LAT_RANGE else LAT_RANGE
        eff_lon = LIGHTNING_LON_RANGE if LIGHTNING_LON_RANGE else LON_RANGE
        
        lfr_data = prepare_lightning_data(tyear, tdoy, eff_lat, eff_lon, TIME_RANGE)
        
        # 3. Filtra GNSS, aplica Hilbert e gera os painéis espectrais
        filter_and_analyze_pairs(
            df=df_gnss, dtec_col=dtec_col, threshold=HILBERT_THRESHOLD,
            tyear=tyear, tdoy=tdoy, TIME_RANGE=TIME_RANGE, lfr_data=lfr_data,
            sample_rate_sec=GNSS_SAMPLE_RATE_SEC, min_period_min=MIN_PERIOD_MIN,
            max_period_min=MAX_PERIOD_MIN, min_points=MIN_POINTS
        )