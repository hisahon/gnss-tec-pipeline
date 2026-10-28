# -*- coding: utf-8 -*-
"""
Keograms of dTEC IPPs aggregating ALL stations into single global canvases
using datashader + dask. Generates TWO independent keograms per day:

  - Time x Latitude
  - Time x Longitude

VERSÃO TERMINAL: lê todos os pares (year, doy) disponíveis, mostra uma lista
numerada e processa UM dia por vez, escolhido pelo número.

Reads the per-station IPP parquets produced by `visualize_ipp.py` and
generates one .nc grid + one .png image per axis.

Uso::

    python visualize_keo_datashader.py                  # lista todos os dias
    python visualize_keo_datashader.py --year 2021      # lista só um ano
    python visualize_keo_datashader.py --resolution high --smooth 0
    python visualize_keo_datashader.py --stage plot     # só refaz a figura
    python visualize_keo_datashader.py --raw            # PNG cru do datashader
    python visualize_keo_datashader.py --lat-range -43 -10 --lon-range -60 -40

Na lista, digite o número do dia e Enter. Depois de processar, a lista volta
a aceitar outro número ("l" mostra a lista de novo, "q" sai).

Os padrões (resolução, trh, suavização, coluna de dTEC) reproduzem o modo
manual do script original.

@author: hisas
"""

import argparse
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import dask                                   # controle de threads
import dask.dataframe as dd
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr
import colorcet as cc
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")                                # backend sem tela (Ubuntu/servidor)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.ndimage import uniform_filter1d

# Import custom modules
from shared.config import dtec_ipp_dir, dtec_keogram_dir, dtec_keogram_figs_dir, storms_SouthAmerica
from shared.pipeline_utils import build_arg_parser, resolve_pairs


# ==========================================
# 0. Constants
# ==========================================

RESOLUTIONS = {            # graus por pixel
    "low": 0.1,
    "high": 0.05,
    "veryhigh": 0.025,
    "ultra": 0.005,
}

# Chave da pasta de referência usada por resolve_pairs() para descobrir os
# pares (year, doy). Mesma dos outros scripts do pipeline.
PAIRS_REFERENCE = "rinexPKLReferenceFolder"

# Padrões = último modo manual do script original
DEFAULT_RESOLUTION = "ultra"
DEFAULT_DTEC_COL   = "dtec_15min"
DEFAULT_TRH        = 0.1501
DEFAULT_SMOOTH     = 20          # px na vertical; 0 = desligado

# Linhas de referência tracejadas na figura matplotlib (RS)
REF_LON = -53.57
REF_LAT = -30.23


# ==========================================
# 1. Core: build aggregated grid
# ==========================================

def smooth_vertical_nan(arr, size, axis=0, treat_zero_as_missing=True, block=500):
    """Vertical moving average (degree axis) ignoring NaN and, optionally,
    zeros (empty bins). size=None or <=1 returns the array untouched.

    Baixo consumo de RAM: processa em blocos de colunas (tempo) e usa float32.
    Como a suavização é só vertical (axis=0), cada coluna é independente.
    """
    if size is None or size <= 1:
        return arr

    out = np.empty(arr.shape, dtype=np.float32)
    for j in range(0, arr.shape[1], block):          # blocos de colunas (tempo)
        sl = arr[:, j:j + block].astype(np.float32)
        valid = np.isfinite(sl)
        if treat_zero_as_missing:
            valid &= (sl != 0.0)

        filled = np.where(valid, sl, np.float32(0))
        num = uniform_filter1d(filled, size=size, axis=axis, mode="nearest")
        den = uniform_filter1d(valid.astype(np.float32), size=size, axis=axis, mode="nearest")

        out[:, j:j + block] = np.divide(num, den,
                                        out=np.full_like(sl, np.nan),
                                        where=den > 0)
    return out


def build_keogram_grid(
    parquet_files,
    y_col: str,
    y_range: tuple,
    plot_width: int = 5760,
    plot_height: int = 1200,
    dtec_col: str = DEFAULT_DTEC_COL,
    num_workers: int = 6,     # nº de threads do dask
    batch_size: int = 12,     # nº de parquets agregados por lote
):
    if not parquet_files:
        raise FileNotFoundError("No IPP parquet files provided.")

    canvas = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=(0, 86400),
        y_range=y_range,
    )

    run_max = run_min = None  # acumuladores float32 entre lotes

    # Limita as threads e processa os arquivos em lotes: o pico de RAM passa a
    # depender de num_workers/batch_size, e não do nº total de estações.
    with dask.config.set(scheduler="threads", num_workers=num_workers):
        for i in range(0, len(parquet_files), batch_size):
            batch = parquet_files[i:i + batch_size]

            # 1. Leia apenas as colunas que não são índice
            df_all = dd.read_parquet(batch, columns=[y_col, dtec_col])
            df_all = df_all.dropna(subset=[y_col, dtec_col])

            # 2. O reset_index() transforma o índice original (horario) em coluna
            df_all = df_all.reset_index()

            # 3. Agora a coluna "horario" existe para o cálculo dos segundos
            df_all["t_seconds"] = (
                df_all["horario"].dt.hour * 3600
                + df_all["horario"].dt.minute * 60
                + df_all["horario"].dt.second
            ).astype("int32")

            # Mapeia os extremos superiores (picos) e inferiores (depleções)
            a_max = canvas.points(df_all, "t_seconds", y_col, agg=ds.max(dtec_col)).astype("float32")
            a_min = canvas.points(df_all, "t_seconds", y_col, agg=ds.min(dtec_col)).astype("float32")

            if run_max is None:
                run_max, run_min = a_max, a_min
            else:
                # fmax/fmin ignoram NaN, então os "buracos" da grade são preservados
                np.fmax(run_max.data, a_max.data, out=run_max.data)
                np.fmin(run_min.data, a_min.data, out=run_min.data)
            del df_all, a_max, a_min

    # Combina: usa o min onde a magnitude dele é maior que a do max, senão mantém o max.
    # Onde não há dado, a comparação com NaN dá False e o NaN é mantido.
    use_min = np.abs(run_min.data) > np.abs(run_max.data)
    run_max.data[use_min] = run_min.data[use_min]

    return run_max


# ==========================================
# 2. Save grid
# ==========================================
def save_grid(agg, nc_path: Path):
    nc_path = Path(nc_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    # grava sempre em float32 (com compressão)
    agg = agg.astype("float32")
    agg.encoding.update(dtype="float32", zlib=True, complevel=4)

    agg.to_netcdf(str(nc_path))
    print(f"Grid saved: {nc_path}")


# ==========================================
# 3. Plot from saved grid
# ==========================================
def plot_keogram(nc_path: Path, trh: float = 0.5, cmap=cc.coolwarm):
    nc_path  = Path(nc_path)
    agg      = xr.open_dataarray(nc_path)
    img      = tf.shade(agg, cmap=cmap, span=[-trh, trh], how="linear")
    img_path = nc_path.with_suffix("")
    img.to_pil().convert("RGB").save(str(img_path) + ".png")
    print(f"Image saved: {img_path}.png")


def plot_keograms_storm_mpl(nc_path_lat: Path, nc_path_lon: Path,
                            year: int, doy: int, resolution: str,
                            movavg_min=None,
                            trh: float = 0.5, cmap: str = "RdBu_r",
                            smooth_size: int = None,
                            plot_storms: bool = False,
                            dpi: int = 300, out_path: Path = None):

    # --- Storm overlay styling (fixed, not exposed in the signature) -
    storm_temp_col  = "temp_min_C"
    storm_cmap      = "rainbow"
    storm_vmin      = -90.0
    storm_vmax      = -65.0
    storm_linewidth = 6.0
    storm_alpha     = 0.25

    agg_lat = xr.open_dataarray(Path(nc_path_lat))
    agg_lon = xr.open_dataarray(Path(nc_path_lon))

    # --- Optional vertical smoothing (degree axis only) -----------------
    lat_vals = agg_lat.values
    lon_vals = agg_lon.values
    if smooth_size and smooth_size > 1:
        lat_vals = smooth_vertical_nan(lat_vals, smooth_size)
        lon_vals = smooth_vertical_nan(lon_vals, smooth_size)

        # degrees-per-pixel derived from each grid's own coordinates
        deg_per_px_lat = abs(float(agg_lat["lts"][1] - agg_lat["lts"][0]))
        deg_per_px_lon = abs(float(agg_lon["lns"][1] - agg_lon["lns"][0]))
        print(f"[smoothing] ON — vertical window = {smooth_size} px "
              f"(≈ {smooth_size * deg_per_px_lat:.3f}° lat, "
              f"{smooth_size * deg_per_px_lon:.3f}° lon).")
    else:
        print("[smoothing] OFF — raw data.")

    extent_lat = [
        float(agg_lat["t_seconds"].min())/3600, float(agg_lat["t_seconds"].max())/3600,
        float(agg_lat["lts"].min()),     float(agg_lat["lts"].max()),
    ]
    extent_lon = [
        float(agg_lon["t_seconds"].min())/3600, float(agg_lon["t_seconds"].max())/3600,
        float(agg_lon["lns"].min()),     float(agg_lon["lns"].max()),
    ]

    # --- Storms overlay: pre-compute X (hours), Y spans and colors ---
    storms_data = None
    if plot_storms:
        try:
            df_storms = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")
            # equivalent radius in degrees from area (km²)
            df_storms['radius_deg'] = np.sqrt(df_storms['area_km2'] / np.pi) / 110.0

            # clip to the target day
            t0 = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            t1 = t0 + pd.Timedelta(days=1)
            df_storms = df_storms[
                (df_storms['datetime'] >= t0) & (df_storms['datetime'] < t1)
            ].copy()

            if not df_storms.empty:
                dt = pd.to_datetime(df_storms['datetime'])
                if dt.dt.tz is not None:
                    dt = dt.dt.tz_convert(None)
                t_hours = (dt - t0).dt.total_seconds().to_numpy() / 3600.0

                norm_s = mpl.colors.Normalize(vmin=storm_vmin, vmax=storm_vmax)
                sm_s   = mpl.cm.ScalarMappable(norm=norm_s, cmap=storm_cmap)
                sm_s.set_array([])
                storm_colors = sm_s.to_rgba(df_storms[storm_temp_col].to_numpy())

                lon_c = df_storms['centroid_lon'].to_numpy()
                lat_c = df_storms['centroid_lat'].to_numpy()
                rad2  = df_storms['radius_deg'].to_numpy() / 2.0

                storms_data = dict(
                    t=t_hours, colors=storm_colors, sm=sm_s,
                    lon_lo=lon_c - rad2, lon_hi=lon_c + rad2,
                    lat_lo=lat_c - rad2, lat_hi=lat_c + rad2,
                )
                print(f"[storms] ON — {len(df_storms)} registros no dia.")
            else:
                print("[storms] ON, mas sem tempestades neste dia.")
        except Exception as exc:  # arquivo ausente, fastparquet não instalado, colunas...
            print(f"[storms] não foi possível carregar ({exc!r}); seguindo sem overlay.")
    else:
        print("[storms] OFF — no overlay.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             constrained_layout=True, sharex=True)

    # --- Top: Longitude ---
    im_lon = axes[0].imshow(
        lon_vals, extent=extent_lon, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh,
        interpolation="nearest",
    )

    avg_txt = f"{movavg_min}-min moving average  |  " if movavg_min else ""
    axes[0].set_title(
        f"dTEC Keogram — {year} DOY {doy:03d}  |  {avg_txt}"
        f"resolution: {RESOLUTIONS.get(resolution, 0.1)} deg x 15 sec"
    )
    axes[0].set_ylabel("Longitud, deg")
    axes[0].set_ylim(extent_lon[2], extent_lon[3])

    axes[0].grid(True, linestyle=":", alpha=0.4)
    plt.colorbar(im_lon, ax=axes[0]).set_label("dTEC (TECU)")

    # Storm overlay (top panel: longitude span)
    if storms_data is not None:
        axes[0].vlines(storms_data["t"],
                       storms_data["lon_lo"], storms_data["lon_hi"],
                       colors=storms_data["colors"],
                       linewidth=storm_linewidth, alpha=storm_alpha,
                       zorder=3, label='Storms')

    axes[0].axhline(REF_LON, color="black", linewidth=1., linestyle="--", zorder=4)

    # --- Bottom: Latitude ---
    im_lat = axes[1].imshow(
        lat_vals, extent=extent_lat, origin="lower", aspect="auto",
        cmap=cmap, vmin=-trh, vmax=trh, interpolation="nearest",
    )
    axes[1].set_ylabel("Latitud, deg")

    axes[1].set_xlim(0, 24)
    axes[1].set_ylim(extent_lat[2], extent_lat[3])
    axes[1].set_xticks(np.arange(0, 25, 3))
    axes[1].set_xlabel("Time (UTC, hours)")
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[1].grid(True, which='major', linestyle=":", alpha=0.4)
    axes[1].grid(True, which='minor', linestyle=":", alpha=0.2)

    plt.colorbar(im_lat, ax=axes[1]).set_label("dTEC (TECU)")

    # Storm overlay (bottom panel: latitude span)
    if storms_data is not None:
        axes[1].vlines(storms_data["t"],
                       storms_data["lat_lo"], storms_data["lat_hi"],
                       colors=storms_data["colors"],
                       linewidth=storm_linewidth, alpha=storm_alpha,
                       zorder=3, label='Storms')

    axes[1].axhline(REF_LAT, color="black", linewidth=1.2, linestyle="--", zorder=4)

    fig.text(0.9, -0.01,
             f'Script: {Path(__file__).name}\nFunction: plot_keograms_storm_mpl\n'
             f'Storage: database/ionosphere/keograms_figs/',
             horizontalalignment='right', verticalalignment='bottom',
             fontsize=10, color='red', alpha=1.)

    if out_path is None:
        out_path = dtec_keogram_figs_dir() / f"keogram_{year}{doy:03d}_{resolution}.mpl.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")


# ==========================================
# 4. Stages: build / plot (por dia)
# ==========================================

def find_ipp_files(year, doy):
    """Parquets de IPP de um dia. Aceita doy com zeros (296, 045) ou sem (45)."""
    folder = Path(dtec_ipp_dir(year, doy))
    if not folder.exists():
        return []
    for tag in (f"{year}{doy:03d}", f"{year}{doy}"):
        files = sorted(folder.glob(f"*_dtecIPP_{tag}.parquet"))
        if files:
            return files
    return []


def grid_paths(year, doy, resolution):
    folder = Path(dtec_keogram_dir(year, doy))
    return (folder / f"keogram_lat_{year}_{doy:03d}_{resolution}.nc",
            folder / f"keogram_lon_{year}_{doy:03d}_{resolution}.nc")


def movavg_minutes(dtec_col):
    m = re.findall(r"\d+", dtec_col)
    return int(m[0]) if m else None


def main_build(year, doy, resolution=DEFAULT_RESOLUTION, dtec_col=DEFAULT_DTEC_COL,
               overwrite=False, lat_range=None, lon_range=None,
               num_workers=6, batch_size=12):
    """
    Stage 1 — aggregate all station parquets into TWO NetCDF grids
    (one for latitude, one for longitude).

    lat_range / lon_range = None -> limites extraídos dos próprios dados.
    """
    deg_per_px = RESOLUTIONS.get(resolution, 0.1)
    lat_path, lon_path = grid_paths(year, doy, resolution)

    lat_exists = lat_path.exists()
    lon_exists = lon_path.exists()

    if lat_exists and lon_exists and not overwrite:
        print(f"Both grids already exist for {year} DOY {doy:03d} ({resolution}). Skipping build.")
        return

    parquet_files = find_ipp_files(year, doy)
    if not parquet_files:
        print(f"No IPP files found in {dtec_ipp_dir(year, doy)}")
        return
    print(f"Found {len(parquet_files)} station files. Aggregating...")

    # ==========================================
    # EXTRAÇÃO DINÂMICA DE LIMITES (MIN/MAX)
    # ==========================================
    if lat_range is None or lon_range is None:
        print("Limites não definidos. Extraindo lat/lon min e max dos dados via Dask...")
        # Lemos apenas as duas colunas necessárias para não sobrecarregar a memória
        df_bounds = dd.read_parquet(parquet_files, columns=["lts", "lns"])

        if lat_range is None:
            lts_min, lts_max = dask.compute(df_bounds["lts"].min(), df_bounds["lts"].max())
            lat_range = (float(lts_min), float(lts_max))
            print(f" -> LAT_RANGE dinâmico: {lat_range}")

        if lon_range is None:
            lns_min, lns_max = dask.compute(df_bounds["lns"].min(), df_bounds["lns"].max())
            lon_range = (float(lns_min), float(lns_max))
            print(f" -> LON_RANGE dinâmico: {lon_range}")

    plot_height_lat = max(1, int((lat_range[1] - lat_range[0]) / deg_per_px))
    plot_height_lon = max(1, int((lon_range[1] - lon_range[0]) / deg_per_px))

    # --- Latitude keogram ---
    if lat_exists and not overwrite:
        print(f"Lat grid exists, skipping: {lat_path.name}")
    else:
        print("Building lat keogram...")
        agg_lat = build_keogram_grid(
            parquet_files, y_col="lts", y_range=lat_range,
            plot_width=5760, plot_height=plot_height_lat, dtec_col=dtec_col,
            num_workers=num_workers, batch_size=batch_size,
        )
        save_grid(agg_lat, lat_path)

    # --- Longitude keogram ---
    if lon_exists and not overwrite:
        print(f"Lon grid exists, skipping: {lon_path.name}")
    else:
        print("Building lon keogram...")
        agg_lon = build_keogram_grid(
            parquet_files, y_col="lns", y_range=lon_range,
            plot_width=5760, plot_height=plot_height_lon, dtec_col=dtec_col,
            num_workers=num_workers, batch_size=batch_size,
        )
        save_grid(agg_lon, lon_path)


def main_plot(year, doy, resolution=DEFAULT_RESOLUTION, dtec_col=DEFAULT_DTEC_COL,
              trh=DEFAULT_TRH, cmap=cc.coolwarm,
              use_matplotlib: bool = True, smooth_size: int = None,
              plot_storms: bool = True):
    """
    Stage 2 — load both saved grids and export their PNGs.

    use_matplotlib : True  -> uma figura com 2 painéis (eixos, colorbar, título)
                     False -> dois PNGs crus do datashader (rápido, sem eixos)
    smooth_size    : janela vertical em pixels (None/0 desliga).
    """
    nc_lat, nc_lon = grid_paths(year, doy, resolution)
    if not nc_lat.exists() or not nc_lon.exists():
        print("Grids not found. Run the build stage first.")
        return
    if use_matplotlib:
        mpl_cmap = "seismic_r" if cmap is cc.coolwarm else cmap
        plot_keograms_storm_mpl(
            nc_lat, nc_lon, year=year, doy=doy, resolution=resolution,
            movavg_min=movavg_minutes(dtec_col),
            trh=trh, cmap=mpl_cmap, plot_storms=plot_storms,
            smooth_size=smooth_size,
        )
    else:
        plot_keogram(nc_lat, trh=trh, cmap=cmap)
        plot_keogram(nc_lon, trh=trh, cmap=cmap)


# ==========================================
# 5. Terminal: lista numerada + seleção
# ==========================================

def parse_arguments():
    parser = build_arg_parser(
        "Keogramas de dTEC — escolha um dia em uma lista numerada"
    )
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default=DEFAULT_RESOLUTION,
                        help="graus/pixel: low=0.1, high=0.05, veryhigh=0.025, ultra=0.005 "
                             f"(padrão: {DEFAULT_RESOLUTION})")
    parser.add_argument("--dtec-col", default=DEFAULT_DTEC_COL,
                        help=f"coluna de dTEC nos parquets de IPP (padrão: {DEFAULT_DTEC_COL})")
    parser.add_argument("--trh", type=float, default=DEFAULT_TRH,
                        help=f"limite da escala de cor ± TECU (padrão: {DEFAULT_TRH})")
    parser.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH,
                        help=f"janela de suavização vertical em px, 0 = desligado (padrão: {DEFAULT_SMOOTH})")
    parser.add_argument("--stage", choices=("build", "plot", "both"), default="both",
                        help="build = só grades .nc; plot = só figura; both (padrão)")
    parser.add_argument("--raw", action="store_true",
                        help="PNGs crus do datashader em vez da figura matplotlib")
    parser.add_argument("--no-storms", action="store_true",
                        help="não sobrepor tempestades na figura matplotlib")
    parser.add_argument("--overwrite", action="store_true",
                        help="refaz as grades mesmo que já existam")
    parser.add_argument("--lat-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="limites de latitude; padrão: extraídos dos dados")
    parser.add_argument("--lon-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="limites de longitude; padrão: extraídos dos dados")
    parser.add_argument("--workers", type=int, default=6,
                        help="threads do dask na agregação (padrão: 6)")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="parquets por lote na agregação (padrão: 12)")
    return parser.parse_args()


def _date_str(year, doy):
    try:
        return datetime.strptime(f"{year}-{doy:03d}", "%Y-%j").strftime("%Y-%m-%d")
    except ValueError:
        return "?"


def day_status(year, doy, resolution):
    """Situação de um dia: nº de parquets de IPP e quais grades já existem."""
    nc_lat, nc_lon = grid_paths(year, doy, resolution)
    return dict(
        year=year, doy=doy, date=_date_str(year, doy),
        n_ipp=len(find_ipp_files(year, doy)),
        has_lat=nc_lat.exists(), has_lon=nc_lon.exists(),
    )


def print_table(rows, resolution):
    print(f"\n  {'#':>4}  {'Ano':>4}  {'DOY':>3}  {'Data':<10}  {'IPP':>5}  Grades ({resolution})")
    for n, r in enumerate(rows, start=1):
        if r["has_lat"] and r["has_lon"]:
            grids = "lat+lon"
        elif r["has_lat"]:
            grids = "só lat"
        elif r["has_lon"]:
            grids = "só lon"
        else:
            grids = "-"
        ipp = str(r["n_ipp"]) if r["n_ipp"] else "sem"
        print(f"  {n:>4}  {r['year']:>4}  {r['doy']:03d}  {r['date']:<10}  {ipp:>5}  {grids}")
    print()


def check_runnable(row, stage):
    """Devolve (ok, motivo). Evita escolher um dia que não tem o que processar."""
    both_grids = row["has_lat"] and row["has_lon"]
    if stage == "plot" and not both_grids:
        return False, "não há grades .nc deste dia nesta resolução (rode com --stage both ou build)."
    if stage in ("build", "both") and row["n_ipp"] == 0 and not both_grids:
        return False, "não há parquets de IPP neste dia."
    return True, ""


def run_day(args, year, doy):
    lat_range = tuple(args.lat_range) if args.lat_range else None
    lon_range = tuple(args.lon_range) if args.lon_range else None

    if args.stage in ("build", "both"):
        main_build(year, doy, resolution=args.resolution, dtec_col=args.dtec_col,
                   overwrite=args.overwrite, lat_range=lat_range, lon_range=lon_range,
                   num_workers=args.workers, batch_size=args.batch_size)
    if args.stage in ("plot", "both"):
        main_plot(year, doy, resolution=args.resolution, dtec_col=args.dtec_col,
                  trh=args.trh, smooth_size=args.smooth or None,
                  use_matplotlib=not args.raw, plot_storms=not args.no_storms)


def main():
    args = parse_arguments()

    pairs = sorted({(int(y), int(d)) for y, d in resolve_pairs(args, PAIRS_REFERENCE)})
    if not pairs:
        print("Nenhuma pasta year/doy encontrada.")
        return 1

    print(f"Lendo {len(pairs)} dia(s)...")
    rows = [day_status(y, d, args.resolution) for y, d in pairs]
    print_table(rows, args.resolution)

    while True:
        try:
            choice = input(f"Número do dia [1-{len(rows)}]  (l = lista, q = sair): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("q", "quit", "exit", "sair"):
            break
        if choice in ("l", "lista", "list"):
            print_table(rows, args.resolution)
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(rows):
            print(f"  Digite um número entre 1 e {len(rows)}.")
            continue

        idx = int(choice) - 1
        row = rows[idx]
        ok, why = check_runnable(row, args.stage)
        if not ok:
            print(f"  {row['year']}/{row['doy']:03d}: {why}")
            continue

        print(f"\n=== {row['year']} / doy {row['doy']:03d} ({row['date']}) — "
              f"{args.resolution}, stage={args.stage} ===")
        try:
            run_day(args, row["year"], row["doy"])
        except KeyboardInterrupt:
            print("\n  Interrompido. Voltando à lista.")
        except Exception:
            traceback.print_exc()
            print("  Falhou; você pode escolher outro dia.")
        # Atualiza o status (grades recém-criadas) sem reler a árvore inteira
        rows[idx] = day_status(row["year"], row["doy"], args.resolution)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
