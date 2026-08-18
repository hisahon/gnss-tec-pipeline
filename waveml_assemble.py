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

# --- NOVO: banda de ressonância observada entre TID e LFR ---
# Faixa onde você observou acoplamento consistente entre as ondulações do TEC
# e a taxa de raios. Precisa estar contida em WAV_FREQ_BOUNDS para os dados
# já calculados cobrirem essa faixa.
RESONANCE_BAND_MHZ = (2.0, 5.0)
RESONANCE_ACTIVE_THRESHOLD = 0.5   # potência normalizada [0,1] acima disso = "banda ativa"

# Abaixo disso (fl/min de pico), a janela é considerada "sem chuva de verdade"
# e não serve como controle negativo — ajuste depois de ver os valores reais
# que aparecem no print [lfr] para alguns dias do seu dataset.
MIN_LFR_PEAK_FOR_CONTROL = 1.0

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
LAG_MINUTES = 9.0                              # atraso de referência (usado no plot, ~tempo vertical troposfera-ionosfera)

# --- NOVO: varredura de lag como parâmetro, não valor fixo ---
# 9 min é uma estimativa para propagação vertical pura. Se houver componente
# horizontal (distância entre a célula convectiva e o IPP), o atraso real
# tende a ser maior. Em vez de assumir um valor, varremos uma faixa e
# deixamos o modelo aprender com o lag que de fato maximiza a correlação
# em cada par — ver _lag_scan_features.
LAG_SEARCH_RANGE_MIN = (0.0, 20.0)   # faixa de atrasos testados, em minutos
LAG_SEARCH_STEP_MIN  = 1.0            # resolução da varredura, em minutos
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

def hours_to_hhmm(h: float) -> str:
    """Converte hora decimal em string HHMM. Ex.: 18.5 -> '1830', 18.0 -> '1800'."""
    total_minutes = int(round(h * 60))
    hh, mm = divmod(total_minutes, 60)
    return f"{hh:02d}{mm:02d}"

def time_range_label(TIME_RANGE: tuple) -> str:
    """Rótulo legível da janela de tempo, formato HHMM-HHMM. Ex.: (18.5, 20) -> '1830-2000'."""
    return f"{hours_to_hhmm(TIME_RANGE[0])}-{hours_to_hhmm(TIME_RANGE[1])}"

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

            # Resumo de atividade — essencial para validar "dias/horários de
            # controle": aqui só serve como negativo se HOUVE raio de verdade
            # (senão os campos lfr_*/cross_res_* ficam NaN e não ensinam nada
            # sobre ausência de acoplamento, só ausência de dado).
            rate = lfr_data["lfr_rate"]
            peak = float(np.max(rate))
            total = float(np.sum(rate) * LFR_SAMPLING_RATE / 60.0)
            frac_active = float(np.mean(rate > 0))
            flag = " ⚠ praticamente sem raios nesta janela — NÃO serve como controle " \
                   "(precisa ter chuva, só sem acoplamento TID observado)." \
                   if peak < MIN_LFR_PEAK_FOR_CONTROL else ""
            print(f"[lfr] Atividade nesta janela: pico={peak:.1f} fl/min | "
                  f"total≈{total:.0f} flashes | {100*frac_active:.0f}% do tempo com algum raio.{flag}")
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
# 2d. NOVO: HELPERS PARA A BANDA DE RESSONÂNCIA (ex.: 3-4 mHz)
# ==========================================================================
def _band_mask(freq: np.ndarray, band: tuple) -> np.ndarray:
    return (freq >= band[0]) & (freq <= band[1])


def _fwhm_bandwidth(freq: np.ndarray, profile: np.ndarray, center_freq: float) -> float:
    """
    Largura de banda real (FWHM — largura a meia altura) do pico espectral
    mais próximo de `center_freq`. Isso é diferente da largura fixa da janela
    (ex.: 1 mHz, de 3 a 4): aqui medimos o quão largo o pico É DE FATO no
    espectro daquele par. Um pico estreito sugere uma ressonância bem
    definida; um pico largo sugere algo mais difuso/banda larga.
    """
    if freq is None or profile is None or len(freq) < 3:
        return np.nan
    center_idx = int(np.argmin(np.abs(freq - center_freq)))
    # procura o pico local de verdade numa vizinhança do centro, em vez de
    # assumir que o bin mais próximo do centro já é o pico
    radius = max(1, len(freq) // 6)
    lo, hi = max(0, center_idx - radius), min(len(freq), center_idx + radius + 1)
    peak_idx = lo + int(np.argmax(profile[lo:hi]))
    peak_val = profile[peak_idx]
    if peak_val <= 0:
        return np.nan
    half = peak_val / 2.0

    i = peak_idx
    while i > 0 and profile[i - 1] >= half:
        i -= 1
    j = peak_idx
    while j < len(freq) - 1 and profile[j + 1] >= half:
        j += 1
    return float(freq[j] - freq[i])


def _band_stats(freq: np.ndarray, power2d: np.ndarray, band: tuple, dt_min: float,
                active_threshold: float = RESONANCE_ACTIVE_THRESHOLD) -> dict:
    """
    Parâmetros de uma banda de frequência específica (ex.: a banda de
    ressonância 3-4 mHz), a partir de uma matriz de potência normalizada
    [0,1] com formato (freq x tempo):
      - pow_mean / pow_max / pow_p95 : intensidade dentro da banda
      - frac_time_active / duration_min : por quanto tempo a banda ficou ativa
      - peak_freq_mhz  : frequência de pico dentro da banda
      - bandwidth_mhz  : largura real (FWHM) do pico espectral mais próximo
      - share_of_total : fração da potência total (toda a faixa calculada,
                         ex. 2-7 mHz) que está concentrada nessa banda
    Também devolve 'time_series' (potência média da banda por instante),
    reaproveitada depois para correlacionar TID x LFR dentro da banda.
    """
    keys = ("pow_mean", "pow_max", "pow_p95", "frac_time_active",
            "duration_min", "peak_freq_mhz", "bandwidth_mhz", "share_of_total")
    empty = {**{k: np.nan for k in keys}, "time_series": None}
    if freq is None or power2d is None:
        return empty

    mask = _band_mask(freq, band)
    if not mask.any():
        return empty

    band_power = power2d[mask, :]
    profile = power2d.mean(axis=1)               # perfil espectral médio (toda a faixa)
    # usa o MÁXIMO entre as frequências da banda em cada instante (não a média):
    # um pico de ressonância estreito, ativo em só algumas frequências da banda,
    # não pode ser diluído pelo ruído de fundo do restante da janela de 1 mHz.
    band_time_series = band_power.max(axis=0)
    active = band_time_series > active_threshold
    center = (band[0] + band[1]) / 2.0

    return {
        "pow_mean": float(band_power.mean()),
        "pow_max": float(band_power.max()),
        "pow_p95": float(np.percentile(band_power, 95)),
        "frac_time_active": float(active.mean()),
        "duration_min": float(active.sum() * dt_min),
        "peak_freq_mhz": float(freq[mask][int(np.argmax(profile[mask]))]),
        "bandwidth_mhz": _fwhm_bandwidth(freq, profile, center),
        "share_of_total": float(profile[mask].sum() / (profile.sum() + 1e-10)),
        "time_series": band_time_series,
    }


# ==========================================================================
# 2d2. NOVO: HELPERS PARA A VARREDURA DE LAG (troposfera -> ionosfera)
# ==========================================================================
def _lag_response_curve(t_gnss, series_gnss, t_lfr, series_lfr,
                        lag_range_min: tuple, lag_step_min: float):
    """
    Varre uma faixa de lags candidatos e, para cada um, alinha `series_gnss`
    (deslocada por esse lag) com `series_lfr` na grade do LFR e calcula a
    correlação entre as duas. Retorna o vetor de lags testados e o vetor de
    correlações correspondente (NaN onde não houve pontos suficientes).
    """
    lags_min = np.arange(lag_range_min[0], lag_range_min[1] + 1e-9, lag_step_min)
    corrs = np.full(len(lags_min), np.nan)
    t_lfr_round = np.round(t_lfr, 5)
    for i, lag in enumerate(lags_min):
        lag_hours = lag / 60.0
        t_shift_round = np.round(t_gnss - lag_hours, 5)
        _, idx_lfr, idx_gnss = np.intersect1d(t_lfr_round, t_shift_round, return_indices=True)
        if len(idx_lfr) < 10:
            continue
        a, b = series_gnss[idx_gnss], series_lfr[idx_lfr]
        if np.std(a) > 0 and np.std(b) > 0:
            corrs[i] = np.corrcoef(a, b)[0, 1]
    return lags_min, corrs


def _lag_scan_features(t_gnss, series_gnss, t_lfr, series_lfr,
                       lag_range_min: tuple, lag_step_min: float,
                       reference_lag_min: float) -> dict:
    """
    Resume a varredura de lag em features:
      - best_lag_min   : atraso que maximiza |correlação| — a estimativa de
                         lag "de verdade" para este par, em vez do valor fixo
      - best_corr      : correlação nesse lag ótimo
      - corr_at_reference_lag : correlação no lag de referência (LAG_MINUTES),
                         para comparar com o que aparece na figura
      - sharpness_min  : largura (em minutos) do pico de correlação ao redor
                         do lag ótimo — um pico estreito indica um atraso bem
                         definido; um pico largo indica algo mais ambíguo
      - n_lags_valid   : quantos lags da varredura puderam ser calculados
                         (poucos pontos válidos = resultado pouco confiável)
    """
    keys = ("best_lag_min", "best_corr", "corr_at_reference_lag",
            "sharpness_min", "n_lags_valid")
    empty = {k: np.nan for k in keys}
    if series_gnss is None or series_lfr is None or t_lfr is None:
        return empty

    lags, corrs = _lag_response_curve(t_gnss, series_gnss, t_lfr, series_lfr,
                                      lag_range_min, lag_step_min)
    valid = ~np.isnan(corrs)
    if valid.sum() < 3:
        return empty

    abs_corrs = np.where(valid, np.abs(corrs), -np.inf)
    best_idx = int(np.argmax(abs_corrs))
    peak_val = abs_corrs[best_idx]
    half = peak_val / 2.0

    i = best_idx
    while i > 0 and valid[i - 1] and abs_corrs[i - 1] >= half:
        i -= 1
    j = best_idx
    while j < len(lags) - 1 and valid[j + 1] and abs_corrs[j + 1] >= half:
        j += 1
    sharpness = float(lags[j] - lags[i])

    ref_idx = int(np.argmin(np.abs(lags - reference_lag_min)))
    corr_ref = float(corrs[ref_idx]) if valid[ref_idx] else np.nan

    return {
        "best_lag_min": float(lags[best_idx]),
        "best_corr": float(corrs[best_idx]),
        "corr_at_reference_lag": corr_ref,
        "sharpness_min": sharpness,
        "n_lags_valid": int(valid.sum()),
    }


# ==========================================================================
# 2e. NOVO: EXTRAÇÃO DE FEATURES PARA O MODELO
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

    # ---- Banda de ressonância (RESONANCE_BAND_MHZ, ex.: 3-4 mHz) ----
    # Mesmo conjunto de estatísticas do TID e do LFR, mas restrito à banda
    # onde você observou o acoplamento, mais a força do cruzamento entre
    # os dois especificamente nessa banda.
    gnss_dt_min = sample_rate_sec / 60.0
    lfr_dt_min = LFR_SAMPLING_RATE / 60.0

    tid_band = _band_stats(prod["freq_tid"], prod["power_tid_short_norm"],
                           RESONANCE_BAND_MHZ, dt_min=gnss_dt_min)
    lfr_band = _band_stats(prod["freq_lfr"], prod["power_lfr_norm"],
                           RESONANCE_BAND_MHZ, dt_min=lfr_dt_min)

    for k in ("pow_mean", "pow_max", "pow_p95", "frac_time_active",
             "duration_min", "peak_freq_mhz", "bandwidth_mhz", "share_of_total"):
        f[f"tid_res_{k}"] = tid_band[k]
        f[f"lfr_res_{k}"] = lfr_band[k]

    # cross-wavelet restrito à banda de ressonância (só nas colunas com dado GNSS real)
    cross_res_keys = ("cross_res_pow_mean", "cross_res_pow_max", "cross_res_pow_p95",
                      "cross_res_frac_gt_03", "cross_res_frac_gt_05", "cross_res_duration_min")
    if cp is not None and prod["idx_lfr"] is not None and len(prod["idx_lfr"]) > 0 \
            and prod["freq_lfr"] is not None:
        cp_valid_res = cp[:, prod["idx_lfr"]]
        mask_res = _band_mask(prod["freq_lfr"], RESONANCE_BAND_MHZ)
        if mask_res.any():
            cp_res = cp_valid_res[mask_res, :]
            active_res = cp_res.max(axis=0) > RESONANCE_ACTIVE_THRESHOLD
            f["cross_res_pow_mean"] = float(cp_res.mean())
            f["cross_res_pow_max"] = float(cp_res.max())
            f["cross_res_pow_p95"] = float(np.percentile(cp_res, 95))
            f["cross_res_frac_gt_03"] = float(np.mean(cp_res > 0.3))
            f["cross_res_frac_gt_05"] = float(np.mean(cp_res > 0.5))
            f["cross_res_duration_min"] = float(active_res.sum() * lfr_dt_min)
        else:
            f.update({k: np.nan for k in cross_res_keys})
    else:
        f.update({k: np.nan for k in cross_res_keys})

    # correlação entre as séries temporais de potência do TID e do LFR,
    # ambas restritas à banda de ressonância — alinhadas na mesma grade e
    # com o mesmo lag usado no cross-wavelet (LAG_MINUTES)
    f["corr_res_tid_lfr"] = np.nan
    if (tid_band["time_series"] is not None and lfr_band["time_series"] is not None
            and prod["idx_lfr"] is not None and prod["idx_gnss"] is not None
            and t_lfr is not None and len(prod["idx_lfr"]) > 10):
        tid_res_on_lfr_grid = np.full(len(t_lfr), np.nan)
        tid_res_on_lfr_grid[prod["idx_lfr"]] = tid_band["time_series"][prod["idx_gnss"]]
        m_res = ~np.isnan(tid_res_on_lfr_grid)
        if m_res.sum() > 10:
            a, b = tid_res_on_lfr_grid[m_res], lfr_band["time_series"][m_res]
            if np.std(a) > 0 and np.std(b) > 0:
                f["corr_res_tid_lfr"] = float(np.corrcoef(a, b)[0, 1])

    # ---- Varredura de lag troposfera->ionosfera (em vez de valor fixo) ----
    # 9 min (LAG_MINUTES) é uma referência para propagação vertical pura.
    # Aqui testamos uma faixa (LAG_SEARCH_RANGE_MIN) e reportamos o lag que
    # de fato maximiza a correlação em CADA par — um valor bem acima de 9 min
    # pode ser indício de componente de propagação horizontal até o IPP.
    # Duas versões: banda larga (envelope x taxa de raios crua) e restrita
    # à banda de ressonância (mais específica ao fenômeno que você descreveu).
    lag_broadband = _lag_scan_features(
        t_hours, env, t_lfr, lfr_rate,
        LAG_SEARCH_RANGE_MIN, LAG_SEARCH_STEP_MIN, LAG_MINUTES)
    lag_resband = _lag_scan_features(
        t_hours, tid_band["time_series"], t_lfr, lfr_band["time_series"],
        LAG_SEARCH_RANGE_MIN, LAG_SEARCH_STEP_MIN, LAG_MINUTES)

    for k, v in lag_broadband.items():
        f[f"lag_broadband_{k}"] = v
    for k, v in lag_resband.items():
        f[f"lag_res_{k}"] = v

    return f


# ==========================================================================
# 2f. NOVO: PONTUAÇÃO COM MODELO TREINADO (se existir)
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

    pasta_destino = f'{FIGURES_OUTPUT_DIR}/{tyear}{tdoy}_{time_range_label(TIME_RANGE)}'
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
    return f'run_{tyear}{tdoy}_{time_range_label(TIME_RANGE)}_{pair_clean}'

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
                    f'_{time_range_label(TIME_RANGE)}.parquet')
        feat_df.to_parquet(out_path, index=False)
        print(f"[analysis] Features salvas em {out_path}")

    return feat_df


# ==========================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    runs = [
        # ---- dias já rotulados (com evento) — horário estendido para pegar
        # negativos "do mesmo contexto", horas antes/depois do evento ----
        # (2024, 333, 13, 17),      # era (18.5, 20) — janela ampliada
        # (2023, 347, 6, 12),      # era (17, 21) — janela ampliada
        # (2023, 347, 6, 18),      # era (17, 21) — janela ampliada
        (2021, 296, 15, 18),      # era (17, 21) — janela ampliada

        # ---- dias/janelas de controle: PRECISAM ter chuva/raios de verdade
        # na região (LFR ativo — confira o print [lfr] no terminal ao rodar),
        # só sem acoplamento TID observado por você. Um dia sem chuva NÃO
        # serve: os campos lfr_*/cross_res_* ficam NaN e não ensinam nada
        # sobre ausência de acoplamento, só ausência de dado. Isso não tem
        # relação com índice geomagnético (Kp/Dst) — é sobre convecção local.
        # (2024, 50, 8, 22),
        # (2024, 150, 8, 22),
        # (2024, 200, 8, 22),
        # (2023, 100, 8, 22),
        # (2023, 200, 8, 22),
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