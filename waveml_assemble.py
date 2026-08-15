# -*- coding: utf-8 -*-
"""
WaveML_Assemble.py
Spatio-temporal scatter of dTEC IPPs & Spectral Analysis (Wavelet + Cross-Spectrum)

VERSÃO ML:
  - Extrai um vetor de features numéricas por par (station·prn) e salva em parquet.
  - Cada figura recebe um sample_id que casa 1:1 com a linha do parquet.
  - Se existir um modelo treinado (MODEL_PATH), pontua cada par e prefixa o
    nome da figura com a probabilidade (ex.: p087_run_...png), de modo que as
    figuras já fiquem ordenadas da mais promissora para a menos promissora.

Ciclo de uso:
  1. Rode este script  -> gera figuras + features_YYYYDDD_t0t1.parquet
  2. Faça sua seleção manual movendo figuras para a subpasta 'selecionadas/'
  3. Rode shared/waveml_build_labels.py -> gera labels.csv a partir das pastas
  4. Rode shared/waveml_train_model.py  -> treina e salva ml_model/selector.joblib
  5. Rode este script de novo -> figuras já saem ranqueadas pelo modelo
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
from scipy.signal import butter, filtfilt, hilbert, find_peaks

# ==========================================================================
# IMPORTAÇÕES DO SEU PROJETO
# ==========================================================================
from shared.config import dtec_ipp_dir
from shared.lightning_spectrogram import load_lightning_lfr
from shared.waveletFunctions import wavelet_on_data_new, wd_slicer


# ==========================================================================
# 0. BLOCO CENTRAL DE CONSTANTES E CONFIGURAÇÕES
# ==========================================================================
TIME_COL = "horario"
LAT_COL  = "lts"
LON_COL  = "lns"
PRN_COL  = "prn"

LIGHTNING_DIR = Path("D:/database/lightning/parquet_15s")
LFR_SAMPLING_RATE = 15   # segundos

# Parâmetros da Wavelet (exatos do lfr_explorer.py / f2_spec.py)
WAV_DT     = 0.25
WAV_PAD    = 1
WAV_DJ     = 0.15
WAV_S0     = 0.90
WAV_J1     = 24
WAV_LAG1   = 0.71
WAV_MOTHER = "MORLET"
WAV_FREQ_BOUNDS = [2, 7]    # mHz

BANDPASS_LOW_MHZ  = 1
BANDPASS_HIGH_MHZ = 12
BANDPASS_ORDER    = 4

GNSS_SAMPLE_RATE_SEC = 15.0
MIN_PERIOD_MIN = 1.2
MAX_PERIOD_MIN = 10.0
HILBERT_THRESHOLD = 0.05
MIN_POINTS = 240

LAT_RANGE = (-32.01, -17.99)
LON_RANGE = (-62.01, -47.99)
LIGHTNING_LAT_RANGE = None
LIGHTNING_LON_RANGE = None

FIGURES_OUTPUT_DIR = "ml_model/figures_spectra"

# --- NOVO: Machine Learning ---
FEATURES_OUTPUT_DIR = "ml_model/features"    # parquets com as features por rodada
MODEL_PATH = Path("ml_model/selector.joblib")  # modelo treinado por shared/waveml_train_model.py
LAG_MINUTES = 9.0                              # atraso TID vs LFR (era hard-coded no plot)
# Se True, gera figura para TODOS os pares que passam no QC (não só amp > threshold).
# Útil na fase de coleta de rótulos, para o modelo aprender também exemplos negativos
# de baixa amplitude. Depois pode voltar para False.
PLOT_ALL_QC_PAIRS = False


# ==========================================================================
# 1. FUNÇÕES DE SUPORTE E CARREGAMENTO  (inalteradas)
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
            lfr_data["lfr_rate"] = lfr_series.values * 60.0
            lfr_data["t_lfr"] = TIME_RANGE[0] + np.arange(len(lfr_data["lfr_rate"])) * LFR_SAMPLING_RATE / 3600.0
            print("[load] LFR data successfully loaded into memory.")
    except Exception as e:
        print(f"[aviso] Falha ao carregar LFR: {e}")
    return lfr_data


# ==========================================================================
# 2. PROCESSAMENTO DE SINAIS  (inalterado)
# ==========================================================================
def compute_wavelet_power(signal: np.ndarray, dt_minutes: float, is_lfr: bool = False):
    sig = signal.astype(float)

    if is_lfr:
        fs = 1.0 / (dt_minutes * 60.0)
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
# 2b. NOVO: WAVELET DO LFR — calculada UMA VEZ POR RODADA
# ==========================================================================
def compute_lfr_wavelet(lfr_data: dict) -> dict:
    """
    A wavelet do LFR (raios) depende só de lfr_data (t_lfr, lfr_rate), que é
    o MESMO para todos os pares station·PRN de uma rodada (mesma data + mesmo
    horário). Recalculá-la a cada par, como acontecia antes, era desperdício
    puro — para uma rodada com 50 pares, a mesma wavelet era feita 50 vezes.
    Chame esta função UMA VEZ por rodada (fora do loop de pares) e reaproveite
    o resultado em compute_spectral_products.
    """
    t_lfr = lfr_data.get("t_lfr")
    lfr_rate = lfr_data.get("lfr_rate")
    result = {
        "t_lfr": t_lfr, "lfr_rate": lfr_rate,
        "freq_lfr": None, "power_lfr": None, "power_lfr_norm": None,
    }
    if t_lfr is not None and lfr_rate is not None and len(t_lfr) > 1:
        lfr_dt_min = (t_lfr[1] - t_lfr[0]) * 60.0
        freq_lfr, power_lfr = compute_wavelet_power(lfr_rate, dt_minutes=lfr_dt_min, is_lfr=True)
        result["freq_lfr"], result["power_lfr"] = freq_lfr, power_lfr
        if power_lfr is not None:
            result["power_lfr_norm"] = (
                (power_lfr - power_lfr.min()) / (power_lfr.max() - power_lfr.min() + 1e-10)
            )
    return result


# ==========================================================================
# 2c. NOVO: PRODUTOS ESPECTRAIS (separado do plot para reuso pelas features)
# ==========================================================================
def compute_spectral_products(t_hours, y_filt, sample_rate_sec, lfr_wavelet: dict):
    """
    Calcula tudo que antes vivia dentro de plot_spectral_panels:
    wavelet do TID (única por par), normalização [0,1], expansão da matriz
    GNSS na grade do LFR (com lag) e a cross-power. A wavelet do LFR já vem
    pronta em `lfr_wavelet` (ver compute_lfr_wavelet) — não é recalculada aqui.
    Retorna um dicionário 'products'; campos podem ser None se algo faltar.
    """
    prod = {
        "t_lfr": lfr_wavelet["t_lfr"],
        "lfr_rate": lfr_wavelet["lfr_rate"],
        "freq_tid": None, "power_tid_short": None,
        "freq_lfr": lfr_wavelet["freq_lfr"], "power_lfr": lfr_wavelet["power_lfr"],
        "power_tid_expanded": None, "power_lfr_norm": lfr_wavelet["power_lfr_norm"],
        "power_tid_short_norm": None, "cross_power": None,
        "idx_lfr": None, "idx_gnss": None,
    }

    gnss_dt_min = sample_rate_sec / 60.0
    prod["freq_tid"], prod["power_tid_short"] = compute_wavelet_power(
        y_filt, dt_minutes=gnss_dt_min, is_lfr=False)

    t_lfr = prod["t_lfr"]
    ok = all(prod[k] is not None for k in
             ("freq_tid", "power_tid_short", "freq_lfr", "power_lfr"))
    if ok:
        p_tid = prod["power_tid_short"]
        prod["power_tid_short_norm"] = (p_tid - p_tid.min()) / (p_tid.max() - p_tid.min() + 1e-10)

        power_tid_expanded = np.zeros((len(prod["freq_lfr"]), len(t_lfr)))
        lag_hours = LAG_MINUTES / 60.0
        t_lfr_round = np.round(t_lfr, 5)
        t_shift_round = np.round(t_hours - lag_hours, 5)
        _, idx_lfr, idx_gnss = np.intersect1d(t_lfr_round, t_shift_round, return_indices=True)
        power_tid_expanded[:, idx_lfr] = prod["power_tid_short_norm"][:, idx_gnss]

        prod["power_tid_expanded"] = power_tid_expanded
        prod["idx_lfr"], prod["idx_gnss"] = idx_lfr, idx_gnss
        prod["cross_power"] = power_tid_expanded * prod["power_lfr_norm"]

    return prod


# ==========================================================================
# 2d. NOVO: EXTRAÇÃO DE FEATURES PARA O MODELO
# ==========================================================================
def extract_features(t_hours, y_filt, env, prod, sample_rate_sec) -> dict:
    """
    Condensa o par em um vetor de features numéricas. NaN quando o produto
    correspondente não pôde ser calculado (o modelo lida com isso via imputação).
    """
    f = {}

    # ---- Envelope de Hilbert / série temporal ----
    f["env_max"]  = float(np.max(env))
    f["env_mean"] = float(np.mean(env))
    f["env_std"]  = float(np.std(env))
    f["env_p90"]  = float(np.percentile(env, 90))
    f["env_frac_above_thr"] = float(np.mean(env > HILBERT_THRESHOLD))
    f["env_dur_above_thr_min"] = f["env_frac_above_thr"] * len(env) * sample_rate_sec / 60.0
    # razão pico/mediana: distingue um pacote de ondas localizado de ruído difuso
    f["env_peak_to_median"] = float(np.max(env) / (np.median(env) + 1e-10))
    # número de pacotes distintos no envelope
    peaks, _ = find_peaks(env, height=HILBERT_THRESHOLD,
                          distance=int(120.0 / sample_rate_sec))  # >= 2 min entre picos
    f["env_n_packets"] = int(len(peaks))
    f["sig_rms"] = float(np.sqrt(np.mean(y_filt ** 2)))
    f["n_samples"] = int(len(y_filt))
    f["duration_hours"] = float(t_hours[-1] - t_hours[0])
    # instante (em horas) do máximo do envelope, relativo ao início da janela
    f["t_env_max_rel_h"] = float(t_hours[np.argmax(env)] - t_hours[0])

    # ---- Wavelet do TID ----
    if prod["power_tid_short_norm"] is not None:
        p = prod["power_tid_short_norm"]
        fr = prod["freq_tid"]
        mean_by_freq = p.mean(axis=1)
        f["tid_pow_mean"] = float(p.mean())
        f["tid_pow_p95"]  = float(np.percentile(p, 95))
        f["tid_dom_freq_mhz"] = float(fr[int(np.argmax(mean_by_freq))])
        # concentração espectral: quanta potência está na frequência dominante
        f["tid_spec_concentration"] = float(mean_by_freq.max() / (mean_by_freq.sum() + 1e-10))
    else:
        f.update({k: np.nan for k in
                  ("tid_pow_mean", "tid_pow_p95", "tid_dom_freq_mhz", "tid_spec_concentration")})

    # ---- LFR ----
    lfr_rate, t_lfr = prod["lfr_rate"], prod["t_lfr"]
    if lfr_rate is not None:
        f["lfr_rate_max"]  = float(np.max(lfr_rate))
        f["lfr_rate_mean"] = float(np.mean(lfr_rate))
        f["lfr_total_flashes"] = float(np.sum(lfr_rate) * LFR_SAMPLING_RATE / 60.0)
    else:
        f.update({k: np.nan for k in ("lfr_rate_max", "lfr_rate_mean", "lfr_total_flashes")})

    if prod["power_lfr_norm"] is not None:
        f["lfr_pow_mean"] = float(prod["power_lfr_norm"].mean())
        f["lfr_pow_p95"]  = float(np.percentile(prod["power_lfr_norm"], 95))
    else:
        f["lfr_pow_mean"] = np.nan
        f["lfr_pow_p95"]  = np.nan

    # ---- Cross-wavelet (o que seu olho procura no 5º painel) ----
    cp = prod["cross_power"]
    if cp is not None and prod["idx_lfr"] is not None and len(prod["idx_lfr"]) > 0:
        # restringe às colunas onde de fato há dados GNSS (fora delas é zero por construção)
        cp_valid = cp[:, prod["idx_lfr"]]
        f["cross_max"]  = float(cp_valid.max())
        f["cross_mean"] = float(cp_valid.mean())
        f["cross_p95"]  = float(np.percentile(cp_valid, 95))
        f["cross_frac_gt_03"] = float(np.mean(cp_valid > 0.3))
        f["cross_frac_gt_05"] = float(np.mean(cp_valid > 0.5))
        i_f, i_t = np.unravel_index(np.argmax(cp_valid), cp_valid.shape)
        f["cross_freq_at_max_mhz"] = float(prod["freq_lfr"][i_f])
        # correlação temporal entre envelope (deslocado pelo lag) e taxa LFR
        env_on_lfr = np.full(len(t_lfr), np.nan)
        env_on_lfr[prod["idx_lfr"]] = env[prod["idx_gnss"]]
        m = ~np.isnan(env_on_lfr)
        if m.sum() > 10 and np.std(env_on_lfr[m]) > 0 and np.std(lfr_rate[m]) > 0:
            f["corr_env_lfr"] = float(np.corrcoef(env_on_lfr[m], lfr_rate[m])[0, 1])
        else:
            f["corr_env_lfr"] = np.nan
    else:
        f.update({k: np.nan for k in
                  ("cross_max", "cross_mean", "cross_p95", "cross_frac_gt_03",
                   "cross_frac_gt_05", "cross_freq_at_max_mhz", "corr_env_lfr")})

    return f


# ==========================================================================
# 2e. NOVO: PONTUAÇÃO COM MODELO TREINADO (se existir)
# ==========================================================================
def load_model_if_available():
    if MODEL_PATH.exists():
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            print(f"[ml] Modelo carregado de {MODEL_PATH} "
                  f"(treinado com {bundle.get('n_train', '?')} exemplos).")
            return bundle
        except Exception as e:
            print(f"[ml][aviso] Falha ao carregar modelo: {e}. Seguindo sem pontuação.")
    else:
        print("[ml] Nenhum modelo treinado encontrado — figuras sem ranqueamento.")
    return None

def score_features(bundle, feats: dict) -> float | None:
    if bundle is None:
        return None
    cols = bundle["feature_names"]
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
    try:
        return float(bundle["model"].predict_proba(X)[0, 1])
    except Exception as e:
        print(f"[ml][aviso] Falha ao pontuar: {e}")
        return None


# ==========================================================================
# 3. ROTINA DE PLOTAGEM (5 Painéis) — agora recebe os produtos prontos
# ==========================================================================
def plot_spectral_panels(
    t_hours, y_filt, amplitude_envelope, pair, max_envelope,
    tyear, tdoy, TIME_RANGE, prod, sample_id, prob=None
):
    print(f'Saving {pair} - Spectral Analysis'
          + (f' | score={prob:.2f}' if prob is not None else ''))

    t_lfr, lfr_rate = prod["t_lfr"], prod["lfr_rate"]
    if t_lfr is None or lfr_rate is None:
        print(f"[erro] LFR data missing for {pair}. Cannot compute cross-wavelet.")
        return None

    fig, axs = plt.subplots(
        5, 1, figsize=(12, 12), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 1.5, 2, 2, 2]}
    )
    ax_env, ax_lfr, ax_wav_tid, ax_wav_lfr, ax_cross = axs

    # --- 1. ENVELOPE ---
    ax_env.scatter(t_hours, y_filt, color='gray', s=3, label='Sinal Filtrado', zorder=2)
    ax_env.plot(t_hours, amplitude_envelope, color='red', label='Envelope', linewidth=1.5, zorder=3)
    ax_env.set_xlim(TIME_RANGE[0], TIME_RANGE[1])
    ax_env.set_ylim(-0.5, 0.5)
    ax_env.set_ylabel("dTEC (TECU)")
    title = f'{pair} amp.max: {max_envelope:.4f} TECU | Data: {tyear}-{tdoy:03d}'
    if prob is not None:
        title += f' | ML score: {prob:.2f}'
    ax_env.set_title(title, fontsize=12)
    ax_env.legend(loc='upper right')

    # --- 2. LFR ---
    ax_lfr.plot(t_lfr, lfr_rate, color="black", linewidth=1.0)
    ax_lfr.fill_between(t_lfr, 0, lfr_rate, color="orange", alpha=0.4)
    ax_lfr.set_ylim(bottom=0)
    ax_lfr.set_ylabel("LFR (fl/min)")
    ax_lfr.sharex(ax_env)

    # --- 3. WAVELET TID ---
    if prod["power_tid_expanded"] is not None:
        cf_tid = ax_wav_tid.contourf(t_lfr, prod["freq_tid"], prod["power_tid_expanded"],
                                     cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_tid, ax=ax_wav_tid, location="right", shrink=1.0, pad=0.02,
                     label="Norm Power [0-1]")
    else:
        ax_wav_tid.text(0.5, 0.5, "Insufficient signal for TID Wavelet",
                        transform=ax_wav_tid.transAxes, ha="center")
    ax_wav_tid.set_ylabel("Freq (mHz)")
    ax_wav_tid.set_title(f"Wavelet Power - TID (Lag: -{LAG_MINUTES} min)", fontsize=10)
    ax_wav_tid.sharex(ax_env)

    # --- 4. WAVELET LFR ---
    if prod["power_lfr_norm"] is not None:
        cf_lfr = ax_wav_lfr.contourf(t_lfr, prod["freq_lfr"], prod["power_lfr_norm"],
                                     cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_lfr, ax=ax_wav_lfr, location="right", shrink=1.0, pad=0.02,
                     label="Norm Power [0-1]")
    else:
        ax_wav_lfr.text(0.5, 0.5, "Insufficient signal for LFR Wavelet",
                        transform=ax_wav_lfr.transAxes, ha="center")
    ax_wav_lfr.set_ylabel("Freq (mHz)")
    ax_wav_lfr.set_title(f"Wavelet Power - LFR ({BANDPASS_LOW_MHZ}–{BANDPASS_HIGH_MHZ} mHz)", fontsize=10)
    ax_wav_lfr.sharex(ax_env)

    # --- 5. CROSS-WAVELET ---
    if prod["cross_power"] is not None:
        cf_cross = ax_cross.contourf(t_lfr, prod["freq_lfr"], prod["cross_power"],
                                     cmap="nipy_spectral", levels=20)
        fig.colorbar(cf_cross, ax=ax_cross, location="right", shrink=1.0, pad=0.02,
                     label="Cross Power [0-1]")
    else:
        ax_cross.text(0.5, 0.5, "Cross-Wavelet computation failed",
                      transform=ax_cross.transAxes, ha="center", color="red")
    ax_cross.set_ylabel("Freq (mHz)")
    ax_cross.set_xlabel("Time (UTC, hours)")
    ax_cross.set_title(f"Cross-Wavelet Power (TID × LFR | {LAG_MINUTES} min lag)", fontsize=10)
    ax_cross.sharex(ax_env)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10 / 60.0))
        ax.grid(True, which="major", linestyle=":", alpha=0.6, zorder=0)
        ax.grid(True, which="minor", linestyle=":", alpha=0.3, zorder=0)

    fig.text(0.99, 0.05,
             f'Script: WaveML_Assemble.py\nFunction: plot_spectral_panels\nStorage: {FIGURES_OUTPUT_DIR}/',
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=10, color='gray', alpha=0.7)

    pasta_destino = f'{FIGURES_OUTPUT_DIR}/{tyear}{tdoy}_{TIME_RANGE[0]}{TIME_RANGE[1]}'
    os.makedirs(pasta_destino, exist_ok=True)

    # Prefixo com o score: as figuras ficam ordenadas por probabilidade no explorer
    prefix = f'p{int(round(prob * 100)):03d}_' if prob is not None else ''
    nome_arquivo = f'{pasta_destino}/{prefix}{sample_id}.png'
    plt.savefig(nome_arquivo, dpi=120)
    plt.close()
    return nome_arquivo


# ==========================================================================
# 4. LOOP DE FILTRAGEM, EXTRAÇÃO DE FEATURES E ORQUESTRAÇÃO
# ==========================================================================
def make_sample_id(tyear, tdoy, TIME_RANGE, pair) -> str:
    """ID único e estável — é ele que liga a figura à linha do parquet."""
    pair_clean = pair.replace(" · ", "-").replace(" ", "")
    return f'run_{tyear}{tdoy}{TIME_RANGE[0]}{TIME_RANGE[1]}_{pair_clean}'

def filter_and_analyze_pairs(
    df, dtec_col, threshold, tyear, tdoy, TIME_RANGE, lfr_data,
    sample_rate_sec, min_period_min, max_period_min, min_points,
    ml_bundle=None,
) -> pd.DataFrame:
    fs = 1.0 / sample_rate_sec
    nyq = 0.5 * fs
    low = (1.0 / (max_period_min * 60.0)) / nyq
    high = (1.0 / (min_period_min * 60.0)) / nyq
    b, a = butter(4, [low, high], btype='band')

    expected_step_hours = sample_rate_sec / 3600.0
    gap_tolerance = expected_step_hours * 1.1

    # A wavelet do LFR (raios) não depende do par GNSS — é a mesma data e o
    # mesmo horário para todos os pares desta rodada. Calcula UMA VEZ aqui,
    # fora do loop, e reaproveita para cada par abaixo.
    lfr_wavelet = compute_lfr_wavelet(lfr_data)
    if lfr_wavelet["power_lfr"] is not None:
        print(f"[wavelet] LFR calculada 1x para a rodada "
              f"{tyear}-{tdoy:03d} [{TIME_RANGE[0]}h-{TIME_RANGE[1]}h] "
              f"e reaproveitada em todos os pares.")

    rows = []
    n_plotted = 0
    for pair in df["pair"].unique():
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
            env = np.abs(hilbert(y_filt))
            max_envelope = np.max(env)
        except ValueError:
            continue

        # -------- FEATURES: para TODOS os pares que passaram no QC --------
        # (mesmo abaixo do threshold — o modelo precisa ver negativos também)
        # Só a wavelet do TID é calculada aqui; a do LFR já vem pronta (acima).
        prod = compute_spectral_products(x, y_filt, sample_rate_sec, lfr_wavelet)
        feats = extract_features(x, y_filt, env, prod, sample_rate_sec)

        sample_id = make_sample_id(tyear, tdoy, TIME_RANGE, pair)
        prob = score_features(ml_bundle, feats)

        meta = {
            "sample_id": sample_id, "pair": pair,
            "station": sub["station"].iloc[0], "prn": str(sub[PRN_COL].iloc[0]),
            "year": tyear, "doy": tdoy,
            "t_beg": TIME_RANGE[0], "t_end": TIME_RANGE[1],
            "ml_score": prob if prob is not None else np.nan,
            "above_threshold": bool(max_envelope > threshold),
        }

        fig_path = None
        if max_envelope > threshold or PLOT_ALL_QC_PAIRS:
            fig_path = plot_spectral_panels(
                t_hours=x, y_filt=y_filt, amplitude_envelope=env,
                pair=pair, max_envelope=max_envelope, tyear=tyear, tdoy=tdoy,
                TIME_RANGE=TIME_RANGE, prod=prod, sample_id=sample_id, prob=prob,
            )
            n_plotted += 1
        meta["figure_path"] = fig_path if fig_path else ""

        rows.append({**meta, **feats})

    feat_df = pd.DataFrame(rows)
    n_above = int(feat_df["above_threshold"].sum()) if not feat_df.empty else 0
    print(f"[analysis] {len(rows)} pares com features | "
          f"{n_above} acima de {threshold} TECU | {n_plotted} figuras geradas.")

    # -------- Salva as features da rodada --------
    if not feat_df.empty:
        os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)
        out_path = (f'{FEATURES_OUTPUT_DIR}/features_{tyear}{tdoy}'
                    f'_{TIME_RANGE[0]}{TIME_RANGE[1]}.parquet')
        feat_df.to_parquet(out_path, index=False)
        print(f"[analysis] Features salvas em {out_path}")

    return feat_df


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

    ml_bundle = load_model_if_available()

    for tyear, tdoy, tbeg, tend in runs:
        TIME_RANGE = (tbeg, tend)
        print(f"\n--- Iniciando processamento para {tyear}-{tdoy:03d} [{tbeg}h - {tend}h] ---")

        df_gnss = load_gnss_window(
            year=tyear, doy=tdoy, lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE
        )
        dtec_col = next(c for c in df_gnss.columns if c.startswith("dtec_"))

        eff_lat = LIGHTNING_LAT_RANGE if LIGHTNING_LAT_RANGE else LAT_RANGE
        eff_lon = LIGHTNING_LON_RANGE if LIGHTNING_LON_RANGE else LON_RANGE
        lfr_data = prepare_lightning_data(tyear, tdoy, eff_lat, eff_lon, TIME_RANGE)

        filter_and_analyze_pairs(
            df=df_gnss, dtec_col=dtec_col, threshold=HILBERT_THRESHOLD,
            tyear=tyear, tdoy=tdoy, TIME_RANGE=TIME_RANGE, lfr_data=lfr_data,
            sample_rate_sec=GNSS_SAMPLE_RATE_SEC, min_period_min=MIN_PERIOD_MIN,
            max_period_min=MAX_PERIOD_MIN, min_points=MIN_POINTS,
            ml_bundle=ml_bundle,
        )