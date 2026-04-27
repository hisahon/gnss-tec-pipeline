# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
Created by Rogério Hisashi Honda on Fri Nov  7 10:44:04 2025
hisashi.brase@gmail.com

----------------------------------------------------------------------------
"""

# -*- coding: utf-8 -*-
# IMPLEMENTAÇÃO COMPLETA: GERA MAPAS DE DENSIDADE (TEMPO × LAT, TEMPO × LON) PARA MÚLTIPLOS ARQUIVOS,
# ACUMULANDO TODAS AS ESTAÇÕES E TODOS OS PRNs EM UMA ÚNICA FIGURA.
# EXIGÊNCIAS ATENDIDAS:
# - normalize_time_grid com sequência estrita
# - trajectories_to_pcolormesh com modos "count" / "sum" / "mean" e coluna obrigatória "w" para sum/mean
# - uso pesado (centenas de trajetórias)
# - Datashader + Matplotlib (pcolormesh + colorbar por painel)
# - resolução configurável (plot_width/plot_height)
# - labels, títulos, shading="auto", constrained_layout=True
# - loop em vários arquivos .parquet produzindo UMA figura final combinada
# - eixo X no tempo (DatetimeIndex)
# - nenhum atalho

# - v4 aqui eu vou alterar para scatter plot ao inves de pcolormesh

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from datetime import datetime
import datashader as ds
# import datashader.transfer_functions as tf  # não diretamente usado, mas útil em extensões
import datashader.reductions as rd


# ---------------------------------------------------------------------
# UTILITÁRIAS DE ARQUIVOS
# ---------------------------------------------------------------------
def read_files(folder: Path | str, extension: str) -> list[Path]:
    folder = Path(folder)
    ext = extension if extension.startswith('.') else f'.{extension}'
    ext = ext.lower()
    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == ext]
    files.sort()
    return files


# ---------------------------------------------------------------------
# NORMALIZAÇÃO TEMPORAL
# ---------------------------------------------------------------------
def infer_freq_from_index(dt_index: pd.DatetimeIndex) -> str | None:
    """Tenta inferir a frequência dominante (moda) em segundos e retorna algo como '15S'."""
    if len(dt_index) < 2:
        return None
    deltas = (dt_index[1:] - dt_index[:-1]).to_series().dt.total_seconds()
    vals = deltas[deltas > 0].round().astype('Int64').dropna()
    if vals.empty:
        return None
    mode = int(vals.mode().iloc[0])
    if mode <= 0:
        return None
    return f"{mode}S"


def normalize_time_grid(
    df: pd.DataFrame,
    time_col: str = "horario",
    freq: str | None = "15s",
    pos_cols=("x_km","y_km","z_km")
) -> pd.DataFrame:
    """
    1) Converte 'horario' para datetime e ordena.
    2) Remove duplicados (mantém o primeiro).
    3) Infere/usa a freq (padrão '15s').
    4) Reindexa para grade contínua (preenche buracos com NaN).
    5) Interpola APENAS as colunas de posição do PRN.
    6) Retorna DataFrame com índice temporal contínuo.
    """
    df = df.copy()

    # 1) datetime + ordenação
    # if not np.issubdtype(df[time_col].dtype, np.datetime64):
    #     df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    # 2) deduplicação
    df = df.drop_duplicates(subset=[time_col], keep="first").set_index(time_col)

    # 3) frequência
    if freq is None:
        freq = infer_freq_from_index(df.index)
        if freq is None:
            freq = "15s"

    # 4) grade contínua
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)

    # 5) interpolar apenas posições do PRN
    # cols_exist = [c for c in pos_cols if c in df.columns]
    # if cols_exist:
    #     df[cols_exist] = df[cols_exist].interpolate(method="time", limit_direction="both")

    # 6) retorno
    return df


# ---------------------------------------------------------------------
# GEOMETRIA: IPP (THIN-SHELL)
# ---------------------------------------------------------------------
def ipp_calc(prn_coordinates, station_coordinates, H_km, RE_km=6371.0):
    x_s, y_s, z_s, ln_s, lt_s, h_s = station_coordinates
    x_p, y_p, z_p = prn_coordinates

    dx = x_p - x_s
    dy = y_p - y_s
    dz = z_p - z_s

    lat = np.radians(lt_s)
    lon = np.radians(ln_s)

    sphi, cphi = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    # ENU
    e = -slon*dx +  clon*dy
    n = -clon*sphi*dx - slon*sphi*dy + cphi*dz
    u =  clon*cphi*dx + slon*cphi*dy + sphi*dz

    # elevação e azimute
    hor = np.hypot(e, n)
    El  = np.arctan2(u, hor)          # [-pi/2, pi/2]
    Az  = np.arctan2(e, n)            # [-pi, pi]

    # thin-shell
    cosE  = np.cos(El)
    schi  = (RE_km/(RE_km + H_km)) * cosE
    schi  = np.clip(schi, -1.0, 1.0)
    chi   = np.arcsin(schi)

    psi = (np.pi/2.0) - El - chi

    spsi, cpsi = np.sin(psi), np.cos(psi)

    # latitude do IPP
    sin_lat_pp = sphi*cpsi + cphi*spsi*np.cos(Az)
    sin_lat_pp = np.clip(sin_lat_pp, -1.0, 1.0)
    lat_pp = np.arcsin(sin_lat_pp)

    # longitude do IPP
    num = spsi*np.sin(Az)
    den = cphi*cpsi - sphi*spsi*np.cos(Az)
    dlon = np.arctan2(num, den)
    lon_pp = lon + dlon

    # normalizar lon para (-pi, pi]
    lon_pp = (lon_pp + np.pi) % (2.0*np.pi) - np.pi

    # IPP inválido para El < 0
    mask = El < 0.0
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)

    # EXEMPLOS DE MÁSCARAS ESPECÍFICAS (AJUSTE SE DESEJAR)
    # (mantidas do seu esboço; revise conforme sua área de interesse)
    # máscara latitude > 20°
    mask = np.degrees(lat_pp) > 20
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)
    # máscara longitude > -20° (parece intencional do seu rascunho)
    mask = np.degrees(lon_pp) > -20
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)

    return np.degrees(lat_pp), np.degrees(lon_pp), np.degrees(El)


def ipp_geometric(prn_coordinates, station_coordinates, H_km, RE_km=6371.0):
    """
    Calcula ponto de perfuração ionosférico (IPP) em casca fina esférica
    e elevação geométrica no receptor, para N posições de satélite.

    Parâmetros
    ----------
    prn_coordinates : (x_p, y_p, z_p)
        Cada um array 1D de tamanho N (km), coordenadas ECEF do satélite.
    station_coordinates : (x_s, y_s, z_s, lon_s, lat_s, h_s)
        Coordenadas da estação:
        - x_s, y_s, z_s: ECEF (km)
        - lon_s, lat_s: longitude/latitude (assumidas em rad)
        - h_s: altura (não usada aqui)
    H_km : float
        Altura da casca ionosférica acima da superfície (km).
    RE_km : float, opcional
        Raio médio da Terra (km).

    Retorna
    -------
    lat_pp_deg : array (N,)
        Latitude geocêntrica do IPP, em graus.
    lon_pp_deg : array (N,)
        Longitude geocêntrica do IPP, em graus.
    El_deg : array (N,)
        Elevação geométrica no receptor, em graus.
    """

    # Desempacota entrada
    x_s, y_s, z_s, lon_s, lat_s, h_s = station_coordinates
    x_p, y_p, z_p = prn_coordinates

    # Vetor posição do receptor: r
    r = np.array([x_s, y_s, z_s], dtype=float)  # (3,)

    # Vetor posição do satélite: s (N,3)
    s = np.vstack((x_p, y_p, z_p)).T           # (N,3)

    # Vetor LOS receptor→satélite: v = s - r
    v = s - r                                  # (N,3)

    # Raio da casca ionosférica: R_ion = RE + H
    R_ion = RE_km + H_km

    # Coeficientes da quadrática em λ: ||r + λ v||^2 = R_ion^2
    # a = ||v||^2
    a = np.sum(v * v, axis=1)                  # (N,)
    # b = 2 r·v
    b = 2.0 * np.dot(v, r)                     # (N,)
    # c = ||r||^2 - R_ion^2
    r2 = np.dot(r, r)
    c = r2 - R_ion**2                          # escalar

    # Discriminante: Δ = b^2 - 4ac
    Delta = b**2 - 4.0 * a * c

    # √Δ, truncando negativos numéricos
    sqrtD = np.sqrt(np.maximum(Delta, 0.0))

    # Raízes λ± = (-b ± √Δ) / (2a)
    lambda1 = (-b + sqrtD) / (2.0 * a)
    lambda2 = (-b - sqrtD) / (2.0 * a)

    # Escolhe menor λ>0 (primeira interseção à frente)
    lambda_star = np.full_like(lambda1, np.nan, dtype=float)

    cond1 = (lambda1 > 0.0) & (lambda2 > 0.0)
    lambda_star[cond1] = np.minimum(lambda1[cond1], lambda2[cond1])

    cond2 = (lambda1 > 0.0) & ~cond1
    lambda_star[cond2] = lambda1[cond2]

    cond3 = (lambda2 > 0.0) & ~cond1 & ~cond2
    lambda_star[cond3] = lambda2[cond3]

    # Ponto de perfuração: p* = r + λ* v
    # (broadcast de λ* sobre as 3 componentes)
    p_star = r + lambda_star[:, None] * v      # (N,3)
    x_pp, y_pp, z_pp = p_star.T

    # Latitude/longitude geocêntricas do IPP:
    # lon_pp = atan2(y, x)
    # lat_pp = atan2(z, sqrt(x^2 + y^2))
    lon_pp = np.arctan2(y_pp, x_pp)
    lat_pp = np.arctan2(z_pp, np.hypot(x_pp, y_pp))

    # Vetor normal local no receptor (zenite): n_r = r / ||r||
    n_r = r / np.sqrt(r2)                      # (3,)

    # Direção LOS unitária: d = v / ||v||
    v_norm = np.sqrt(a)                        # (N,)
    # Evita divisão por zero
    with np.errstate(invalid='ignore', divide='ignore'):
        d = v / v_norm[:, None]                # (N,3)

    # Elevação geométrica: sin(El) = d·n_r
    dn = np.dot(d, n_r)                        # (N,)
    dn = np.clip(dn, -1.0, 1.0)
    El = np.arcsin(dn)                         # (N,) rad

    # Máscara de casos sem interseção física (Δ<0 ou λ* nan ou El<=0)
    mask_bad = (Delta < 0.0) | (~np.isfinite(lambda_star)) | (El <= 0.0)

    lat_pp[mask_bad] = np.nan
    lon_pp[mask_bad] = np.nan
    El[mask_bad] = np.nan

    # Converte para graus
    return np.degrees(lat_pp), np.degrees(lon_pp), np.degrees(El)
# ---------------------------------------------------------------------
# RASTERIZAÇÃO DAS TRAJETÓRIAS PARA PCOLORMSH
# ---------------------------------------------------------------------
def _ensure_2d_time_matrix(t_like, n_cols: int) -> np.ndarray:
    """
    Recebe um array DatetimeIndex/array datetime64 ou um array 2D.
    Se 1D, repete em colunas para formar uma matriz (N × n_cols).
    Retorna np.ndarray 2D de dtype datetime64[ns].
    """
    # Se já for DataFrame/Series com datetime:
    if isinstance(t_like, (pd.Series, pd.DatetimeIndex, pd.Index)):
        t_arr = t_like.to_numpy()
    else:
        t_arr = np.asarray(t_like)

    if t_arr.ndim == 1:
        return np.tile(t_arr[:, None], (1, n_cols))
    elif t_arr.ndim == 2:
        if t_arr.shape[1] != n_cols:
            raise ValueError("Quando X_df é 2D, o número de colunas deve igualar Y_df.")
        return t_arr
    else:
        raise ValueError("X_df inválido: deve ser 1D (tempo) ou 2D (matriz tempo).")


def trajectories_to_pcolormesh(
    X_df,                     # 1D (tempo) ou 2D (tempo replicado) compatível com Y_df
    Y_df: pd.DataFrame,       # 2D, linhas = instantes, colunas = trajetórias
    W_df: pd.DataFrame | None = None,  # pesos/atributo para sum/mean (mesma forma de Y_df)
    plot_width: int = 1600,
    plot_height: int = 1600,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
    agg: str = "count"        # "count", "sum", "mean"
):
    """
    Converte trajetórias (x,y) em grid (x_edges, y_edges, Z) para pcolormesh.
    - X_df: 1D (DatetimeIndex/array) ou 2D (N×M). Se 1D, será replicado para M colunas.
    - Y_df: DataFrame (N×M) com y (lat ou lon).
    - W_df: DataFrame (N×M) com pesos/atributo para 'sum'/'mean'. Obrigatório para esses aggs.
    """
    if not isinstance(Y_df, pd.DataFrame):
        raise TypeError("Y_df deve ser um pandas.DataFrame.")

    N, M = Y_df.shape
    X_mat = _ensure_2d_time_matrix(X_df, M)

    if X_mat.shape != Y_df.shape:
        raise ValueError("X (tempo) e Y (trajetórias) precisam ter a mesma forma 2D após expansão.")

    # Flatten
    x_time = X_mat.ravel()                         # dtype datetime64[ns]
    y_vals = Y_df.to_numpy().ravel()

    # Monta DataFrame longo
    df_long = pd.DataFrame({"x": x_time, "y": y_vals})

    # Para sum/mean, exigir w
    if agg in ("sum", "mean"):
        if W_df is None:
            raise ValueError('Para agg="sum" ou "mean", forneça W_df com mesma forma de Y_df.')
        if W_df.shape != Y_df.shape:
            raise ValueError("W_df deve ter a mesma forma de Y_df.")
        df_long["w"] = W_df.to_numpy().ravel()
        df_long = df_long.dropna(subset=["w"])
        
    df_long = df_long.dropna(subset=["x", "y"])

    # Faixas
    if x_range is None:
        x_range = (df_long["x"].min(), df_long["x"].max())
    else:
        x_range = (pd.to_datetime(x_range[0]), pd.to_datetime(x_range[1]))
    if y_range is None:
        y_range = (df_long["y"].min(), df_long["y"].max())

    # Canvas
    cvs = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=x_range,
        y_range=y_range
    )

    # Agregador
    if agg == "count":
        agg_da = cvs.points(df_long, "x", "y", rd.count())
        cbar_label = "Contagem por célula"
    elif agg == "sum":
        agg_da = cvs.points(df_long, "x", "y", rd.sum("w"))
        cbar_label = "Soma por célula (w)"
    elif agg == "mean":
        agg_da = cvs.points(df_long, "x", "y", rd.mean("w"))
        cbar_label = "Média por célula (w)"
    else:
        raise ValueError('agg deve ser "count", "sum" ou "mean".')

    # Matriz Z (plot_height × plot_width)
    Z = np.asarray(agg_da)

    # Arestas (pcolormesh exige arestas)
    # Para tempo (datetime), o Matplotlib aceita arrays datetime64 nos edges.
    # x_edges = pd.date_range(x_range[0], x_range[1], periods=plot_width + 1).to_numpy()
    x_edges = np.linspace(x_range[0], x_range[1], plot_width + 1)
    y_edges = np.linspace(y_range[0], y_range[1], plot_height + 1)

    return x_edges, y_edges, Z, cbar_label



def load_and_filter_stations(station_file: Path | str) -> pd.DataFrame:
    """
    Reads station positions and filters for the specific lat/lon region.
    """
    df_sp = pd.read_parquet(station_file, engine="fastparquet").set_index('name')
    
    # Filter specific geographic slice (South America / Brazil sector based on coords)
    stations_slice = df_sp[
        (df_sp['ln_deg'] >= -60) & (df_sp['ln_deg'] < -45) & 
        (df_sp['lt_deg'] >= -38) & (df_sp['lt_deg'] < -20)
    ]
    return stations_slice

def process_prn_file(
    file_path: Path, 
    stations_slice: pd.DataFrame, 
    time_freq: str, 
    shell_height_km: float
    ):
    """
    Reads a single PRN file, cleans time, and calculates IPP (Lat/Lon/El) 
    for all valid stations. Returns aligned DataFrames.
    """
    prn = file_path.stem.split('_')[-1]
    
    # 1. Extract "2021-296" from the filename
    date_str = file_path.stem.split('_')[1]
    t_start = pd.to_datetime(date_str, format='%Y-%j', utc=True)
    t_end = t_start + pd.Timedelta(days=1)
    
    # 1. Read and Time Filter
    df = pd.read_parquet(file_path, engine="fastparquet")
    
    # Note: Ideally these dates should be arguments, currently kept as per original logic
    # t_start = pd.Timestamp(2021, 10, 23, tz='UTC')
    # t_end   = pd.Timestamp(2021, 10, 24, tz='UTC')
    df = df[(df['horario'] >= t_start) & (df['horario'] < t_end)]
    
    if df.empty:
        return None

    # 2. Time Normalization
    df = normalize_time_grid(df, time_col="horario", freq=time_freq, pos_cols=("x_km","y_km","z_km"))
    
    # 3. Prepare Coordinates
    prn_coordinates = (
        df['x_km'].to_numpy(),
        df['y_km'].to_numpy(),
        df['z_km'].to_numpy()
    )
    
    stations_in_region = stations_slice.index.to_numpy()
    
    # Accumulators
    lts, lns, els = {}, {}, {}
    valid_stations = []

    # 4. Calculation Loop (IPP)
    for st in stations_in_region:
        # Validation checks
        if st not in df.columns: continue
        
        station_coords = stations_slice.loc[st].values # [x, y, z, lon, lat, h]
        if np.any(np.abs(station_coords) < 0.001): continue

        # Calculate IPP
        lt, ln, el = ipp_calc(prn_coordinates, station_coords, H_km=shell_height_km)
        
        lts[st] = lt
        lns[st] = ln
        els[st] = el
        valid_stations.append(st)

    if not lts:
        return None

    # 5. Construct Result DataFrames
    idx_time = df.index
    df_lt = pd.DataFrame(lts, index=idx_time)
    df_ln = pd.DataFrame(lns, index=idx_time)
    df_el = pd.DataFrame(els, index=idx_time)
    df_dtec = df[valid_stations] # Extract the data values matching the valid stations

    return {
        "prn": prn,
        "df_lt": df_lt,
        "df_ln": df_ln,
        "df_el": df_el,
        "df_dtec": df_dtec,
        "time_index": idx_time,
        "date_string": date_str,
        "t_start": t_start,
        "t_end": t_end
        }

def plot_ipp_trajectory(
    data_pack: dict, 
    output_dir: str = "imagens", 
    cmap: str = "turbo", 
    trh: float = 0.5
    ):
    """
    Generates the Tempo x Lat and Tempo x Lon scatter plots.
    """
    prn = data_pack["prn"]
    df_lt = data_pack["df_lt"]
    df_ln = data_pack["df_ln"]
    df_dtec = data_pack["df_dtec"]
    t_index = data_pack["time_index"]
    date_string = data_pack["date_string"]
    t_start = data_pack["t_start"]
    t_end = data_pack["t_end"]
    
    df_storms = data_pack["df_storms"]

    t_rep = np.repeat(t_index, df_lt.shape[1])
    
    lt_flat = df_lt.to_numpy().ravel()
    ln_flat = df_ln.to_numpy().ravel()
    dtec_flat = df_dtec.to_numpy().ravel()

    mask_common = ~np.isnan(dtec_flat)
    
    # Ensure masks align with individual coordinate availability if necessary
    mask_lat = mask_common & ~np.isnan(lt_flat)
    mask_lon = mask_common & ~np.isnan(ln_flat)
    
    
    centroid_time = df_storms['datetime']
    centroid_lat = df_storms['centroid_lat']
    centroid_lon = df_storms['centroid_lon']
    centroid_radius = df_storms['radius_deg']
    centroid_temp = df_storms['temp_min_C']
    
    # 3. Plotting
    # fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    # cmap_use = cmap if cmap else "viridis"
    date_fmt = mdates.DateFormatter('%H:%M')

    # # Subplot 1: Lat vs Time
    # sc1 = axes[0].scatter(
    #     t_rep[mask_lat], lt_flat[mask_lat], c=dtec_flat[mask_lat],
    #     s=2, alpha=0.8, marker='|', cmap=cmap_use, vmin=-trh, vmax=trh
    #     )
    

    # axes[0].vlines(
    #     x=centroid_time, 
    #     ymin=centroid_lat - (centroid_radius/2), 
    #     ymax=centroid_lat + (centroid_radius/2),
    #     colors='black', linewidth=1., alpha=0.2, label='Eventos'
    # )
    
    
    ################################
    # 1. Definição da Coluna de Temperatura
    # storm_temps = df_storms['temperatura'] # Substitua pelo nome real
    norm_storm = plt.Normalize(vmin=-95, vmax=-65)
    cmap_storm = plt.get_cmap('Blues_r') 
    storm_colors = cmap_storm(norm_storm(centroid_temp.values))
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    cmap_use = cmap if cmap else "viridis"
    
    sc1 = axes[0].scatter(
        t_rep[mask_lat], lt_flat[mask_lat], c=dtec_flat[mask_lat],
        s=1, alpha=0.8, marker='|', cmap=cmap_use, vmin=-trh, vmax=trh,
        zorder = 1
    )
    axes[0].vlines(
        x=centroid_time, 
        ymin=centroid_lat - (centroid_radius/2), 
        ymax=centroid_lat + (centroid_radius/2),
        colors=storm_colors,  # <--- Cores calculadas (-80 a -20)
        linewidth=3.,          # Mais grosso para ver a cor bem
        alpha=0.2,
        zorder = 2,
        label='Tempestades'
    )   
    # LINHAS DAS TEMPESTADES (Coloridas pela temperatura negativa)

    
    axes[0].set_title(f"{date_string} [{prn}] Storm colorscale from -65 to -95 ºC")
    axes[0].set_ylabel("Latitude (°)")
    axes[0].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    cb1 = plt.colorbar(sc1, ax=axes[0])
    cb1.set_label("dTEC, TECU")

    # Subplot 2: Lon vs Time
    
    sc2 = axes[1].scatter(
        t_rep[mask_lon], ln_flat[mask_lon], c=dtec_flat[mask_lon],
        s=1, alpha=0.8, marker='|', cmap=cmap_use, vmin=-trh, vmax=trh,
        zorder = 1
        )
    
    axes[1].vlines(
        x=centroid_time, 
        ymin=centroid_lon - (centroid_radius/2), 
        ymax=centroid_lon + (centroid_radius/2),
        colors=storm_colors,  # <--- Cores calculadas (-80 a -20)
        linewidth=3.,          # Mais grosso para ver a cor bem
        alpha=0.2,
        zorder = 2,
        label='Tempestades'
    )

    # axes[1].set_title(f"Scatter — Tempo × Longitude IPP")
    axes[1].set_ylabel("Longitude (°)")
    axes[1].set_xlabel(f"[{date_string}] Time (UTC)") # X-axis is actually Timestamps now, formatted auto by mpl
    axes[1].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    axes[1].set_xlim(t_start, t_end)
    axes[1].xaxis.set_major_formatter(date_fmt)
    
    cb2 = plt.colorbar(sc2, ax=axes[1])
    cb2.set_label("dTEC, TECU")

    # 4. Save
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    save_path = Path(output_dir) / f"{date_string}_{prn}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {prn}")
    
    
def run_pipeline_and_plot(
    folder_base: Path | str,
    station_positions_file: Path | str,
    shell_height_km: float = 300.0,
    parquet_extension: str = "parquet",
    time_freq: str | None = "15s",
    cmap: str = "turbo",
    trh: float = 0.5
    ):
    
    folder_base = Path(folder_base)
    
    
    

    
    
    # 1. Get Files
    files = read_files(folder_base, parquet_extension)
    if not files:
        raise FileNotFoundError(f"No files found in {folder_base}")

    # 2. Load Stations (Refactored Function 1)
    stations_slice = load_and_filter_stations(station_positions_file)
    
    storms_path = Path('C:/Users/hisas/Dropbox/python/cptec/merged/c13_2018-2025_SouthAmerica.parquet')
    df_storms = pd.read_parquet(storms_path, engine="fastparquet")
    

    
    
    
    df_storms['radius_deg'] = np.sqrt(df_storms['area_km2']/np.pi)/110
    
    
    # 3. Loop and Process
    for file in files:
        # Calculation (Refactored Function 2)
        data_pack = process_prn_file(
            file, 
            stations_slice, 
            time_freq, 
            shell_height_km
            )
        
        if data_pack is None:
            continue
        

       
        t_start = data_pack["t_start"].tz_localize(None)
        t_end = data_pack["t_end"].tz_localize(None)
        
        df_stormsS = df_storms[
                # (df_storms['centroid_lon'] >= -60) & (df_storms['centroid_lon'] < -45) & 
                # (df_storms['centroid_lat'] >= -38) & (df_storms['centroid_lat'] < -20) &
                (df_storms['datetime'] >= pd.to_datetime(t_start)) & (df_storms['datetime'] < pd.to_datetime(t_end))
            ]
        
        data_pack['df_storms'] = df_stormsS  
            
        plot_ipp_trajectory(data_pack, cmap=cmap, trh=trh)
        

# ---------------------------------------------------------------------
# EXEMPLO DE USO (AJUSTE OS CAMINHOS ANTES DE EXECUTAR)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Caminhos de exemplo – SUBSTITUA pelos seus:
    BASE_DB = Path("/home/hisashi/database/gnss/")    
    BASE_DB = Path("D:/database")   
    # STATION_POSITIONS_PARQUET = BASE_LINUX / "station_positions" / "SP_2021-296.parquet"
    STATION_POSITIONS_PARQUET = BASE_DB / "station_positions" / "SP_2021-296.parquet"
    FOLDER_PARQUETS = BASE_DB / "dtec" / "2021" / "296"
    # FOLDER_PARQUETS = Path("D:/banco_dados/dtec/2021/296")

    # MODO DE AGREGAÇÃO:
    # - "count"  → densidade de passagens
    # - "mean"   → média de 'w' (definir weight_source="elevation" para média da elevação, por exemplo)
    # - "sum"    → soma de 'w'
    run_pipeline_and_plot(
        folder_base=FOLDER_PARQUETS,
        station_positions_file=STATION_POSITIONS_PARQUET,
        shell_height_km=200,
        parquet_extension="parquet",
        time_freq="15s",
        # plot_width=1200,
        # plot_height=1200,
        # agg_mode="mean",            # troque para "mean" ou "sum" conforme necessidade
        # weight_source=None,          # use "elevation" para média/soma por elevação; ou nome de coluna por estação
        # weight_source='dtec',          # use "elevation" para média/soma por elevação; ou nome de coluna por estação
        cmap="turbo",              # "rainbow" se preferir, conforme seu exemplo
        # cmap="seismic",              # "rainbow" se preferir, conforme seu exemplo
        trh=.5,
        # set_equal_aspect=False       # True somente se fizer sentido
    )

    # EXEMPLOS:
    # 1) Média da elevação por célula:
    # run_pipeline_and_plot(
    #     FOLDER_PARQUETS, STATION_POSITIONS_PARQUET, agg_mode="mean", weight_source="elevation"
    # )
    #
    # 2) Soma (ou média) de um atributo por estação armazenado nos parquets (ex.: TEC incremental 'dtec'):
    # run_pipeline_and_plot(
    #     FOLDER_PARQUETS, STATION_POSITIONS_PARQUET, agg_mode="mean", weight_source="dtec"
    # )


