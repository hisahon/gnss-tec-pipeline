# -*- coding: utf-8 -*-
"""
Spatio-temporal scatter of dTEC IPPs, keeping EACH receiver-PRN pair as an
independent curve (NO aggregation / NO flattening like the keogram does).

Given a lat / lon / time window, this routine:
  1. scans the per-station IPP parquets produced by `visualize_ipp.py`,
  2. keeps only the IPPs that fall inside the requested box and time span,
  3. groups the surviving points by (receiver, PRN),
  4. selects a subset of those pairs (all / top-N by sample count / explicit
     list), and
  5. draws a scatter plot where every receiver-PRN pair is its own series.

Unlike `visualize_keo.py` this does NOT use datashader/dask and does NOT
produce a flat keogram — the goal is to inspect a handful of individual
satellite passes inside a small region.

@author: hisas
"""

import argparse
import re
from pathlib import Path


import numpy as np
import pandas as pd
import duckdb

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Import custom modules (same project layout as visualize_keo.py)
from shared.config import dtec_ipp_dir, dtec_keogram_figs_dir, storms_SouthAmerica
from shared.storm_tracking import plot_storms_tracked
from shared.lightning_spectrogram import lfr_spectrogram, WAV_FREQ_BOUNDS


# ==========================================================================
# 0b. Dedicated output folder for this script's figures
# ==========================================================================
def ttd_figs_dir() -> Path:
    """
    Output folder for visualize_scatter.py figures only
    (kept separate from dtec_keogram_figs_dir(), which is used by
    visualize_keo.py). Created next to it if it doesn't exist yet.
    """
    folder = Path(dtec_keogram_figs_dir()).parent / "keograms_TTDs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ==========================================================================
# 0. Data-schema assumptions  ---  ADJUST HERE IF YOUR PARQUET DIFFERS
# ==========================================================================
# The parquet index is the timestamp (`horario`), exactly as in
# visualize_keo.py (it does reset_index() then reads df["horario"]).
TIME_COL = "horario"     # datetime  (parquet index)
LAT_COL  = "lts"         # IPP latitude  (deg)
LON_COL  = "lns"         # IPP longitude (deg)
PRN_COL  = "prn"         # satellite id column inside each parquet  <-- confirm
# The receiver/station name is NOT a column: it is taken from the file name
# pattern  "{station}_dtecIPP_{year}{doy}.parquet".


# ==========================================================================
# 1. Load + spatio-temporal cut
# ==========================================================================
def _station_from_path(path: Path) -> str:
    """Extract the receiver/station code from the IPP file name."""
    # "{station}_dtecIPP_{year}{doy}.parquet" -> station
    return path.name.split("_dtecIPP_")[0]


def _detect_dtec_col(file: str, candidates: list[str] | None = None) -> str:
    """
    Return the first dtec-like column found in the parquet schema.
    Raises ValueError if none is found.
    """
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

    # fallback: any column whose name starts with "dtec_"
    dtec_cols = sorted(c for c in available if c.startswith("dtec_"))
    if dtec_cols:
        return dtec_cols[0]

    raise ValueError(
        f"No dtec column found in {file}.\n"
        f"Available columns: {sorted(available)}"
    )


def load_window(
    year: int,
    doy: int,
    lat_range: tuple,
    lon_range: tuple,
    time_range: tuple,
    dtec_col: str | None = None,        # None → auto-detect
    stations: list | None = None,
) -> pd.DataFrame:
    ipp_folder = Path(dtec_ipp_dir(year, doy))
    files = sorted(ipp_folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    if not files:
        raise FileNotFoundError(f"No IPP files found in {ipp_folder}")
    if stations is not None:
        wanted = set(stations)
        files = [f for f in files if _station_from_path(f) in wanted]
        if not files:
            raise FileNotFoundError(
                f"None of the requested stations {sorted(wanted)} were found."
            )

    # --- auto-detect dtec column from the first file's schema --------------
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
        raise ValueError(
            "No IPPs fall inside the requested lat/lon box. "
            "Widen the box or check the ranges."
        )

    # --- time cut + decimal UTC hour --------------------------------------
    t = pd.to_datetime(out[TIME_COL])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    out[TIME_COL] = t
    out["t_hours"] = t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0
    h0, h1 = time_range
    out = out[out["t_hours"].between(h0, h1)]
    if out.empty:
        raise ValueError(
            "No IPPs fall inside the requested time window. "
            "Widen the time range or check the hours."
        )

    out = out.reset_index(drop=True)
    out["pair"] = out["station"].astype(str) + " · " + out[PRN_COL].astype(str)
    return out


# ==========================================================================
# 2. Pair selection
# ==========================================================================
def select_pairs(
    df: pd.DataFrame,
    mode: str = "topN",        # "all" | "topN" | "list"
    n: int = 12,
    pairs: list | None = None,
    min_points: int = 5,
) -> list:
    """
    Pick which receiver-PRN pairs to plot.

      - "all"  : every pair with >= min_points samples
      - "topN" : the n pairs with the most samples (longest passes)
      - "list" : only the pairs explicitly given in `pairs`
                 (each item: "STATION · PRN xx" or a (station, prn) tuple)
    """
    counts = df.groupby("pair").size().sort_values(ascending=False)
    counts = counts[counts >= min_points]

    if mode == "all":
        chosen = counts.index.tolist()
    elif mode == "topN":
        chosen = counts.head(n).index.tolist()
    elif mode == "list":
        if not pairs:
            raise ValueError("mode='list' requires a non-empty `pairs` list.")
        norm = []
        for p in pairs:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                norm.append(f"{p[0]} · PRN {p[1]}")
            else:
                norm.append(str(p))
        chosen = [p for p in norm if p in set(df["pair"])]
        missing = set(norm) - set(chosen)
        if missing:
            print(f"[select] requested pairs not found / no data: {sorted(missing)}")
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")

    if not chosen:
        raise ValueError("No pairs survived the selection criteria.")

    print(f"[select] mode={mode} -> {len(chosen)} pair(s):")
    # for p in chosen:
    #     print(f"    {p:<28s}  {int(counts.get(p, 0)):>5d} pts")
    return chosen


# ==========================================================================
# 3a. dTEC line stack (X = time, one offset line per receiver-PRN pair)
# ==========================================================================
def plot_pairs_lines(
    df: pd.DataFrame,
    chosen_pairs: list,
    year: int,
    doy: int,
    dtec_col: str = "dtec_15min",
    offset: float = 1.0,           # vertical offset (TECU) between successive pairs
    trh: float | None = None,      # optional symmetric clip applied per curve
    cmap_name: str = "tab20",
    lw: float = 1.0,
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Stacked LINE plot of dTEC, one line per receiver-PRN pair, with each
    baseline shifted vertically by `offset` TECU (seismic-style cascade).

    X is UTC time (decimal hours). The order of `chosen_pairs` defines the
    vertical stacking from bottom up. A thin grey reference line marks each
    pair's zero, the pair name is shown as the y-tick at its baseline, and a
    `offset`-TECU scale bar is drawn in the upper-left corner.
    """
    if offset <= 0:
        raise ValueError(f"offset must be > 0 TECU (got {offset}).")

    movavg = (
        int(re.findall(r"\d+", dtec_col)[0]) if re.search(r"\d", dtec_col) else None
    )
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(chosen_pairs))]

    # height grows with the number of pairs so labels don't collide
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
        # pair's own zero
        ax.axhline(baseline, color="0.85", linewidth=0.6, zorder=1)
        ax.plot(sub["t_hours"], y + baseline,
                color=color, linewidth=lw, zorder=3)

    # --- Y axis: pair names at each baseline (no TECU ticks) ---
    ax.set_yticks(baselines)
    ax.set_yticklabels(chosen_pairs, fontsize=8)
    bot_pad = 0.6 * offset
    top_pad = 1.4 * offset      # leaves room for the scale bar
    ax.set_ylim(baselines[0] - bot_pad, baselines[-1] + top_pad)

    # --- X axis (time) ---
    sel = df["pair"].isin(chosen_pairs)
    h0 = float(df.loc[sel, "t_hours"].min())
    h1 = float(df.loc[sel, "t_hours"].max())
    ax.set_xlim(h0, h1)
    ax.set_xlabel("Time (UTC, hours)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)

    # --- offset scale bar (length = `offset` TECU) ---
    span = h1 - h0
    x_bar = h0 + 0.015 * span
    cap = 0.004 * span
    y_bot = baselines[-1] + 0.25 * offset
    y_top = y_bot + offset
    ax.plot([x_bar, x_bar], [y_bot, y_top],
            color="0.15", linewidth=2.0, zorder=4)
    ax.plot([x_bar - cap, x_bar + cap], [y_bot, y_bot],
            color="0.15", linewidth=2.0, zorder=4)
    ax.plot([x_bar - cap, x_bar + cap], [y_top, y_top],
            color="0.15", linewidth=2.0, zorder=4)
    ax.text(x_bar + 2 * cap, (y_bot + y_top) / 2, f"{offset:g} TECU",
            fontsize=8, va="center", ha="left", color="0.15")

    title = (
        f"dTEC line stack by receiver-PRN — {year} DOY {doy:03d}"
        f"  |  {len(chosen_pairs)} pair(s)  |  offset {offset:g} TECU"
    )
    if movavg is not None:
        title += f"  |  {movavg}-min moving average"
    ax.set_title(title)

    if out_path is None:
        out_path = (
            ttd_figs_dir()
            / f"linestack_{year}{doy:03d}_pairs.png"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path


# ==========================================================================
# 3b. Two-panel trajectory (top: time x lon, bottom: time x lat)
#     one curve per receiver-PRN pair  — same layout as the keogram figure
# ==========================================================================


def plot_pairs_trajectory(
    df: pd.DataFrame,
    chosen_pairs: list,
    year: int,
    doy: int,
    dtec_col: str = "dtec_15min",
    trh: float | None = None,      # symmetric dTEC color limit
    cmap: str = "seismic_r",
    s: float = 12.0,
    plot_storms: bool = True,      # overlay storm centroids as reference
    add_spectrogram: bool = True,  # extra panel: LFR wavelet spectrogram
    lat_range: tuple | None = None,   # falls back to global LAT_RANGE
    lon_range: tuple | None = None,   # falls back to global LON_RANGE
    time_range: tuple | None = None,  # falls back to global TIME_RANGE
    lightning_lat_range: tuple | None = None,  # area used ONLY for the LFR
    lightning_lon_range: tuple | None = None,  # panel; falls back to the
                                                # TTD's own lat/lon box above
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Stacked panels sharing the time (X) axis (same layout as the keogram):
        top     -> IPP longitude vs time
        middle  -> IPP latitude  vs time
        bottom  -> lightning flash rate (LFR) wavelet spectrogram
                   (only when add_spectrogram=True)

    Each point is colored by its dTEC on a shared colorbar. Optionally overlays
    storm centroids as a light, neutral reference layer (position only — the
    temperature analysis lives in the dedicated storm figure).

    The lightning panel searches for flashes inside `lightning_lat_range` /
    `lightning_lon_range` if given, which can be wider (or otherwise
    different) than the box used for the dTEC/IPP panels above -- useful
    since the storm driving the lightning may sit outside the narrow IPP
    window. If left as None, it defaults to the same box as the TTD.
    """
    # --- resolve effective lat/lon/time window (explicit arg > global) -----
    eff_lat_range  = lat_range  if lat_range  is not None else LAT_RANGE
    eff_lon_range  = lon_range  if lon_range  is not None else LON_RANGE
    eff_time_range = time_range if time_range is not None else TIME_RANGE

    # lightning-specific box: independent override, else same as the TTD
    eff_lightning_lat_range = (
        lightning_lat_range if lightning_lat_range is not None else eff_lat_range
    )
    eff_lightning_lon_range = (
        lightning_lon_range if lightning_lon_range is not None else eff_lon_range
    )

    # --- Storm reference marker styling --------------------------------
    storm_marker        = "o"
    storm_marker_s      = 35.
    storm_marker_alpha  = 1.

    sub = df.loc[df["pair"].isin(chosen_pairs)]

    vmax = trh if trh is not None else float(np.nanmax(np.abs(sub[dtec_col])))
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    # --- Storm centroids: load, clip to day + box (position only) ------
    storms_data = None
    if plot_storms:
        print("[storms] ON — centroid reference markers enabled. "
              "To turn OFF, call with plot_storms=False.")
        df_storms = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")

        # clip to the target day
        t0 = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        t1 = t0 + pd.Timedelta(days=1)
        df_storms = df_storms[
            (df_storms['datetime'] >= t0) & (df_storms['datetime'] < t1)
        ].copy()

        # clip to the lat/lon box (same global window used for the IPPs, padded)
        lat_lo, lat_hi = sorted(eff_lat_range)
        lon_lo, lon_hi = sorted(eff_lon_range)
        df_storms = df_storms[
            df_storms['centroid_lat'].between(lat_lo - 3, lat_hi + 3) &
            df_storms['centroid_lon'].between(lon_lo - 3, lon_hi + 3)
        ].copy()

        if not df_storms.empty:
            dt = pd.to_datetime(df_storms['datetime'])
            if dt.dt.tz is not None:
                dt = dt.dt.tz_convert(None)
            t_hours = (dt - t0).dt.total_seconds().to_numpy() / 3600.0

            storms_data = dict(
                t=t_hours,
                lon_c=df_storms['centroid_lon'].to_numpy(),
                lat_c=df_storms['centroid_lat'].to_numpy(),
            )
            print(f"[storms] {len(t_hours)} centroids in window.")
        else:
            print("[storms] no storms found for this day.")
    else:
        print("[storms] OFF — no centroid markers. "
              "To turn ON, call with plot_storms=True.")

    if add_spectrogram:
        fig, (ax_lon, ax_lat, ax_spec) = plt.subplots(
            3, 1, figsize=(14, 12), constrained_layout=True, sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1]},
        )
    else:
        fig, (ax_lon, ax_lat) = plt.subplots(
            2, 1, figsize=(14, 9), constrained_layout=True, sharex=True,
        )
        ax_spec = None

    absdtec = np.abs(sub[dtec_col])
    w      = np.clip(absdtec / vmax, 0.0, 1.0)
    gamma  = 2.0                            # 1 = linear, >1 = penaliza mais os fracos
    alpha  = 0.10 + 0.90 * (w ** gamma)

    s = absdtec * 25

    ax_lon.scatter(sub["t_hours"], sub[LON_COL], c=sub[dtec_col], s=s,
                   cmap=cmap, norm=norm, edgecolors="none", alpha=alpha)
    ax_lat.scatter(sub["t_hours"], sub[LAT_COL], c=sub[dtec_col], s=s,
                   cmap=cmap, norm=norm, edgecolors="none", alpha=alpha)

    # --- Storm centroids: neutral reference scatter --------------------
    if storms_data is not None:
            ax_lon.scatter(storms_data["t"], storms_data["lon_c"],
                           marker=storm_marker, s=storm_marker_s,
                           facecolors="cyan", edgecolors="black",
                           alpha=storm_marker_alpha, linewidths=1.0,
                           zorder=4, label="Cold spots positions")
            ax_lat.scatter(storms_data["t"], storms_data["lat_c"],
                           marker=storm_marker, s=storm_marker_s,
                           facecolors="cyan", edgecolors="black",
                           alpha=storm_marker_alpha, linewidths=1.0,
                           zorder=4, label="storm centroid")
            ax_lon.legend(loc="upper right", fontsize=8, framealpha=0.6)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    fig.colorbar(sm, ax=(ax_lon, ax_lat), location="right",
                 shrink=1.0, pad=0.02, label="dTEC (TECU)")

    ax_lon.set_ylabel("IPP longitude, deg")
    ax_lat.set_ylabel("IPP latitude, deg")

    # --- Panel 3: lightning flash rate (LFR) wavelet spectrogram ----------
    if add_spectrogram:
        spec_result = lfr_spectrogram(
            year, doy,
            lat_range=eff_lightning_lat_range, lon_range=eff_lightning_lon_range,
            time_range=eff_time_range,
        )

        if spec_result is not None:
            t_hours_spec, freq, power = spec_result

            cf = ax_spec.contourf(
                t_hours_spec, freq, power, cmap="nipy_spectral", levels=20,
            )
            fig.colorbar(cf, ax=ax_spec, location="right",
                         shrink=1.0, pad=0.02, label="log10(power)")
            ax_spec.set_title(
                "Wavelet power — Lightning Flash Rate (Morlet, "
                f"{WAV_FREQ_BOUNDS[0]}–{WAV_FREQ_BOUNDS[1]} mHz)",
                fontsize=9,
            )
        else:
            ax_spec.text(
                0.5, 0.5, "No lightning data / insufficient signal for wavelet",
                transform=ax_spec.transAxes, ha="center", va="center", fontsize=9,
            )
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])
        ax_spec.set_ylabel("Frequency (mHz)")

    bottom_ax = ax_spec if add_spectrogram else ax_lat
    bottom_ax.set_xlabel("Time (UTC, hours)")
    bottom_ax.set_xlim(float(sub["t_hours"].min()), float(sub["t_hours"].max()))
    bottom_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    bottom_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1/6))

    axes_all = (ax_lon, ax_lat, ax_spec) if add_spectrogram else (ax_lon, ax_lat)
    for ax in axes_all:
        ax.set_axisbelow(False)   # forca o grid a ficar por cima do scatter
        ax.grid(True, linestyle=":", alpha=1., zorder=5)
        ax.grid(True, which="minor", linestyle=":", alpha=1., zorder=5)
        

    movavg_srt = int(re.findall(r'\d+', dtec_col)[0])
    ax_lon.set_title(
        f"TRAVEL TIME DIAGRAM: {year}-{doy:03d} | MOV. AVG.: {movavg_srt} min"
        f"  |  {len(chosen_pairs)} pair(s)"
    )
    
    fig.text(0.9, -0.01, 'Script: gnss-tec-pipeline/visualize_scatter.py\nFunction: plot_pairs_trajectory\nStorage: database/ionosphere/keograms_TTDs/', 
             horizontalalignment='right', 
             verticalalignment='bottom', 
             fontsize=10, 
             color='black', 
             alpha=0.7)
    
    if out_path is None:
        out_path = (
            ttd_figs_dir()
            / f"ttd_{year}-{doy:03d}-{eff_time_range[0]}-{eff_time_range[1]}.png"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula a distância esférica (em km) entre dois pontos na Terra."""
    R = 6371.0 # Raio médio da Terra em km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def plot_pairs_distance(
    df: pd.DataFrame,
    chosen_pairs: list,
    year: int,
    doy: int,
    dtec_col: str = "dtec_15min",
    trh: float | None = None,
    cmap: str = "Spectral",        # Colormap atualizado para melhor contraste
    s: float = 12.0,
    plot_storms: bool = True,
    add_spectrogram: bool = True,
    lat_range: tuple | None = None,
    lon_range: tuple | None = None,
    time_range: tuple | None = None,
    lightning_lat_range: tuple | None = None,
    lightning_lon_range: tuple | None = None,
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Scatter plot da distância (em km) entre os IPPs e os centroides das nuvens, 
    cuja trajetória é modelada por uma aproximação polinomial.
    Compartilha o eixo X (tempo) com um possível espectrograma.
    """
    # --- Resolve os limites geográficos e temporais globais ---
    eff_lat_range  = lat_range  if lat_range  is not None else LAT_RANGE
    eff_lon_range  = lon_range  if lon_range  is not None else LON_RANGE
    eff_time_range = time_range if time_range is not None else TIME_RANGE

    eff_lightning_lat_range = lightning_lat_range if lightning_lat_range is not None else eff_lat_range
    eff_lightning_lon_range = lightning_lon_range if lightning_lon_range is not None else eff_lon_range

    sub = df.loc[df["pair"].isin(chosen_pairs)].copy()
    
    # Ordenar por tempo é essencial para que os arrays batam corretamente
    sub = sub.sort_values("t_hours")

    vmax = trh if trh is not None else float(np.nanmax(np.abs(sub[dtec_col])))
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    # --- Processamento e Aproximação Polinomial das Nuvens ------
    dist_array = np.full(len(sub), np.nan)
    
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
            # 1. Agrupar centroides do mesmo instante pela média 
            storm_agg = df_storms.groupby('datetime', as_index=False)[['centroid_lon', 'centroid_lat']].mean()
            storm_agg = storm_agg.sort_values('datetime')
            
            t_hours_storm = (storm_agg['datetime'] - t0).dt.total_seconds().to_numpy() / 3600.0
            lon_c_arr = storm_agg['centroid_lon'].to_numpy()
            lat_c_arr = storm_agg['centroid_lat'].to_numpy()

            # 2. Aproximação Polinomial (Suavização da trajetória)
            grau = 2  # Usando grau 2 para capturar a tendência parabólica suave do sistema
            coef_lon = np.polyfit(t_hours_storm, lon_c_arr, grau)
            coef_lat = np.polyfit(t_hours_storm, lat_c_arr, grau)

            # 3. Calcular a posição suave projetada para os exatos instantes dos IPPs
            sub_t = sub["t_hours"].to_numpy()
            interp_lon = np.polyval(coef_lon, sub_t)
            interp_lat = np.polyval(coef_lat, sub_t)

            # 4. Calcular distância de Haversine (em km)
            dist_array = haversine_distance(sub[LAT_COL].to_numpy(), sub[LON_COL].to_numpy(), interp_lat, interp_lon)
            print(f"[storms] Trajetória modelada com polinômio de grau {grau}.")
            print(f"[storms] Distância máx: {np.nanmax(dist_array):.1f} km, mín: {np.nanmin(dist_array):.1f} km")
        else:
            print("[storms] Nenhuma nuvem encontrada para este dia/janela.")

    sub["distance_km"] = dist_array

    # --- Configuração da Figura ---
    if add_spectrogram:
        fig, (ax_dist, ax_spec) = plt.subplots(
            2, 1, figsize=(14, 9), constrained_layout=True, sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig, ax_dist = plt.subplots(figsize=(14, 6), constrained_layout=True)
        ax_spec = None

    absdtec = np.abs(sub[dtec_col])
    w      = np.clip(absdtec / vmax, 0.0, 1.0)
    gamma  = 2.0                            
    alpha  = 0.10 + 0.90 * (w ** gamma)
    s_sizes = absdtec * 25

    # --- Painel 1: Scatter da Distância ---
    ax_dist.scatter(
        sub["t_hours"], sub["distance_km"], 
        c=sub[dtec_col], s=s_sizes,
        cmap=cmap, norm=norm, edgecolors="none", alpha=alpha
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_dist, location="right", shrink=1.0, pad=0.02, label="dTEC (TECU)")

    ax_dist.set_ylabel("Distância IPP à Nuvem (km)")

    # --- Painel 2: Espectrograma (Opcional) ---
    if add_spectrogram:
        spec_result = lfr_spectrogram(
            year, doy,
            lat_range=eff_lightning_lat_range, lon_range=eff_lightning_lon_range,
            time_range=eff_time_range,
        )

        if spec_result is not None:
            t_hours_spec, freq, power = spec_result
            cf = ax_spec.contourf(t_hours_spec, freq, power, cmap="nipy_spectral", levels=20)
            fig.colorbar(cf, ax=ax_spec, location="right", shrink=1.0, pad=0.02, label="log10(power)")
            ax_spec.set_title(f"Wavelet power — Lightning Flash Rate (Morlet, {WAV_FREQ_BOUNDS[0]}–{WAV_FREQ_BOUNDS[1]} mHz)", fontsize=9)
        else:
            ax_spec.text(0.5, 0.5, "No lightning data / insufficient signal for wavelet", transform=ax_spec.transAxes, ha="center", va="center", fontsize=9)
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])
        ax_spec.set_ylabel("Frequency (mHz)")

    # --- Configurações dos Eixos X e Grid ---
    bottom_ax = ax_spec if add_spectrogram else ax_dist
    bottom_ax.set_xlabel("Time (UTC, hours)")
    bottom_ax.set_xlim(float(sub["t_hours"].min()), float(sub["t_hours"].max()))
    bottom_ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    bottom_ax.xaxis.set_minor_locator(ticker.MultipleLocator(1/6))

    axes_all = (ax_dist, ax_spec) if add_spectrogram else (ax_dist,)
    for ax in axes_all:
        ax.set_axisbelow(False)
        ax.grid(True, linestyle=":", alpha=1., zorder=5)
        ax.grid(True, which="minor", linestyle=":", alpha=1., zorder=5)

    movavg_srt = int(re.findall(r'\d+', dtec_col)[0]) if re.search(r"\d", dtec_col) else "N/A"
    ax_dist.set_title(
        f"DISTANCE DIAGRAM: {year}-{doy:03d} | MOV. AVG.: {movavg_srt} min | {len(chosen_pairs)} pair(s) | Storm Fit: Poly(2)"
    )

    # Nota lateral semelhante à rotina original
    fig.text(0.9, -0.01, 'Script: gnss-tec-pipeline/visualize_scatter.py\nFunction: plot_pairs_distance\nStorage: database/ionosphere/keograms_TTDs/', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=10, color='black', alpha=0.7)

    # --- Salvar Arquivo ---
    if out_path is None:
        out_path = ttd_figs_dir() / f"dist_storm_{year}-{doy:03d}-{eff_time_range[0]}-{eff_time_range[1]}.png"
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path

# ==========================================================================
# 4. CLI parsing
# ==========================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scatter dTEC IPPs by receiver-PRN pair inside a "
                    "lat/lon/time window (no keogram, no aggregation)."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy",  type=int, required=True)
    parser.add_argument("--lat",  type=float, nargs=2, required=True,
                        metavar=("LAT_LO", "LAT_HI"))
    parser.add_argument("--lon",  type=float, nargs=2, required=True,
                        metavar=("LON_LO", "LON_HI"))
    parser.add_argument("--time", type=float, nargs=2, default=(0.0, 24.0),
                        metavar=("H_START", "H_END"))
    parser.add_argument("--dtec-col", type=str, default="dtec_15min")
    parser.add_argument("--offset", type=float, default=1.0,
                        help="Vertical offset in TECU between successive pairs "
                             "in the line stack (timeseries kind).")
    parser.add_argument("--mode", choices=["all", "topN", "list"], default="topN")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--trh", type=float, default=None)
    parser.add_argument("--kind", choices=["timeseries", "trajectory", "distance"],
                        default="timeseries",
                        help="timeseries = dTEC line stack; "
                             "trajectory = lon/lat vs time (two panels); "
                             "distance = dist(km) to storm vs time.")
    parser.add_argument("--color-by", choices=["dtec", "pair"], default="dtec",
                        help="Trajectory coloring (ignored for timeseries).")
    return parser.parse_args()


# ==========================================================================
# 5. Entry point: one-shot pipeline
# ==========================================================================

def run(
    year, doy,
    lat_range, lon_range, time_range,
    dtec_col=None,                # None → auto-detected inside load_window
    kind="timeseries",
    offset=1.0,
    color_by="dtec",
    mode="topN", n=12, pairs=None, min_points=5,
    trh=None,
    stations=None,
    add_spectrogram=True,         # trajectory only: LFR wavelet panel
    lightning_lat_range=None,     # trajectory only: independent LFR box
    lightning_lon_range=None,     # (defaults to lat_range/lon_range above)
    out_path=None,
):
    df = load_window(
        year, doy,
        lat_range=lat_range, lon_range=lon_range, time_range=time_range,
        dtec_col=dtec_col, stations=stations,
    )

    # resolve the actual column name (may have been auto-detected)
    dtec_col = next(c for c in df.columns if c.startswith("dtec_"))

    print(f"[load] {len(df)} IPPs in window | "
          f"{df['station'].nunique()} stations | "
          f"{df['pair'].nunique()} receiver-PRN pairs | "
          f"dtec_col='{dtec_col}'")

    chosen = select_pairs(df, mode=mode, n=n, pairs=pairs, min_points=min_points)

    if kind == "trajectory":
        return plot_pairs_trajectory(
            df, chosen, year=year, doy=doy,
            dtec_col=dtec_col, trh=trh,
            add_spectrogram=add_spectrogram,
            lat_range=lat_range, lon_range=lon_range, time_range=time_range,
            lightning_lat_range=lightning_lat_range,
            lightning_lon_range=lightning_lon_range,
            out_path=out_path,
        )
    elif kind == "timeseries":
        return plot_pairs_lines(
            df, chosen, year=year, doy=doy,
            dtec_col=dtec_col, offset=offset, trh=trh, out_path=out_path,
        )
        
    elif kind == "distance":
        return plot_pairs_distance(
            df, chosen, year=year, doy=doy,
            dtec_col=dtec_col, trh=trh,
            add_spectrogram=add_spectrogram,
            lat_range=lat_range, lon_range=lon_range, time_range=time_range,
            lightning_lat_range=lightning_lat_range,
            lightning_lon_range=lightning_lon_range,
            out_path=out_path,
        )
    else:
        raise ValueError(f"Unknown kind: {kind!r} (use 'timeseries', 'trajectory' or 'distance').")


# ==========================================================================
# 6. IDE / CLI entry point
# ==========================================================================
if __name__ == "__main__":
    RUN_MANUALLY = True
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")

        # --- Date/time windows to process -----------------------------------
        runs = [
            # (2021, 296, 14, 19),
            (2022, 279, 18, 22),
            (2022,  97, 17, 20),
            
            (2023, 347, 8, 10),
            (2023, 347,  15, 17),
            (2023, 347,  17, 20),
            (2023, 348,  17, 21),
            # (2023, 348, 18, 22),
            
            (2024, 264,  11, 13),
            (2024, 264,  13, 15),
            (2024, 264, 16, 18),
            (2024, 264, 18,  21),
            # (2024, 264,  9, 22),
            
            (2024, 333,  10, 12),
            (2024, 333,  12, 14),
            (2024, 333,  14, 16),
            (2024, 333,  16, 18),
            
            # (2025, 298,  7, 12),
            # (2025, 298, 18, 21),
        ]

        # --- Spatio-temporal cut --------------------------------------------
        LAT_RANGE = (-32.01, -17.99)    # deg
        LON_RANGE = (-62.01, -47.99)    # deg

        # Lightning panel: independent area (can be smaller OR larger than
        # LAT_RANGE/LON_RANGE above -- e.g. a tighter box around the storm
        # of interest, or a wider one to catch flashes outside the TEC box).
        # Leave both as None to just reuse LAT_RANGE / LON_RANGE above.
        LIGHTNING_LAT_RANGE = None    # e.g. (-27.0, -23.0)
        LIGHTNING_LON_RANGE = None    # e.g. (-55.0, -51.0)

        # --- Which figures to generate ---------------------------------------
        GENERATE_TTD    = True   # TTD (trajectory) figure -- the one we're using
        GENERATE_STORMS = False  # 4-panel storms figure -- disabled for now
        ADD_SPECTROGRAM = True   # extra LFR wavelet panel inside the TTD figure
        GENERATE_DIST   = True   # Nova figura de distância

        # --- Loop over all date/time windows --------------------------------
        for tyear, tdoy, tbeg, tend in runs:
            TIME_RANGE = (tbeg, tend)

            # --- Figure 1: TTD (ionosphere) --------------------------------
            if GENERATE_TTD:
                ttd_path = (ttd_figs_dir()
                            / f"ttd_{tyear}-{tdoy:03d}-{tbeg}-{tend}.png")
                if ttd_path.exists():
                    print(f"[skip] {ttd_path.name} already exists.")
                else:
                    print(f"[run ] {ttd_path.name}")
                    run(
                        year=tyear, doy=tdoy,
                        lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
                        dtec_col=None,          # auto-detect per file
                        kind="trajectory",
                        color_by="dtec",
                        mode="topN", n=500, min_points=10,
                        trh=.301,
                        add_spectrogram=ADD_SPECTROGRAM,
                        lightning_lat_range=LIGHTNING_LAT_RANGE,
                        lightning_lon_range=LIGHTNING_LON_RANGE,
                        out_path=ttd_path,
                    )

            # --- Figure 2: tracked storms (complementary) ------------------
            if GENERATE_STORMS:
                storms_path = (ttd_figs_dir()
                               / f"storms_{tyear}-{tdoy:03d}-{tbeg}-{tend}.png")
                if storms_path.exists():
                    print(f"[skip] {storms_path.name} already exists.")
                else:
                    print(f"[run ] {storms_path.name}")
                    plot_storms_tracked(
                        year=tyear, doy=tdoy,
                        lat_range=LAT_RANGE, lon_range=LON_RANGE,
                        time_range=TIME_RANGE,
                        max_link_dist_deg=2.0,
                        max_gap_frames=1,
                        min_track_len=3,
                        out_path=storms_path,
                    )
                    
            # --- Figure 1.5: Distance to Storm -----------------------------
            if GENERATE_DIST:
                dist_path = (ttd_figs_dir()
                            / f"dist_{tyear}-{tdoy:03d}-{tbeg}-{tend}.png")
                if dist_path.exists():
                    print(f"[skip] {dist_path.name} already exists.")
                else:
                    print(f"[run ] {dist_path.name}")
                    run(
                        year=tyear, doy=tdoy,
                        lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
                        dtec_col=None,          
                        kind="distance",     # Aqui chamamos o novo método!
                        color_by="dtec",
                        mode="topN", n=500, min_points=10,
                        trh=.301,
                        add_spectrogram=ADD_SPECTROGRAM,
                        lightning_lat_range=LIGHTNING_LAT_RANGE,
                        lightning_lon_range=LIGHTNING_LON_RANGE,
                        out_path=dist_path,
                    )

    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        run(
            year=args.year, doy=args.doy,
            lat_range=tuple(args.lat), lon_range=tuple(args.lon),
            time_range=tuple(args.time),
            dtec_col=getattr(args, "dtec_col", None),   # None if not passed via CLI
            kind=args.kind,
            offset=args.offset, color_by=args.color_by,
            mode=args.mode, n=args.n, trh=args.trh,
        )