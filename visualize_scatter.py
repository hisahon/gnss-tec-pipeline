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
import matplotlib.dates as mdates

# Import custom modules (same project layout as visualize_keo.py)
from shared.config import dtec_ipp_dir, dtec_keogram_figs_dir


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


def load_window(
    year: int,
    doy: int,
    lat_range: tuple,
    lon_range: tuple,
    time_range: tuple,          # (h_start, h_end) in UTC decimal hours
    dtec_col: str = "dtec_15min",
    stations: list | None = None,
) -> pd.DataFrame:
    """
    Read every station parquet for (year, doy) with DuckDB, keep only the IPPs
    inside the lat/lon box and the time window, and tag each row with its
    receiver name (parsed from the file path).

    Why DuckDB and not pandas/dask here:
      - the *input* is the whole day for many stations, the *output* is a tiny
        cut. DuckDB applies the lat/lon predicate as a parquet pushdown
        (skipping row groups via min/max zonemaps) and projects only the
        needed columns, so we never materialise the full day in Python RAM.
      - dask would parallelise the read but still has to scan everything and
        offers no win once the result fits in memory, plus parsing the station
        from the filename is clumsier than DuckDB's `filename=true`.

    Returns a single DataFrame with columns:
        [TIME_COL, LAT_COL, LON_COL, PRN_COL, dtec_col, "t_hours",
         "station", "pair"]
    """
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

    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    file_list = [str(f) for f in files]

    # Spatial cut + dropna + station tag are pushed down to DuckDB. The
    # time cut is done afterwards in pandas on the already-small result, to
    # stay robust to tz-aware vs tz-naive timestamps across files.
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

    # --- time cut + decimal UTC hour (pandas, on the small spatial cut) ---
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
    out["pair"] = out["station"].astype(str) + " · PRN " + out[PRN_COL].astype(str)
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
    for p in chosen:
        print(f"    {p:<28s}  {int(counts.get(p, 0)):>5d} pts")
    return chosen


# ==========================================================================
# 3a. Time-series scatter (X = time, one series per receiver-PRN pair)
# ==========================================================================
def plot_pairs_scatter(
    df: pd.DataFrame,
    chosen_pairs: list,
    year: int,
    doy: int,
    y_col: str = "dtec_15min",     # what goes on Y: "dtec_*" | LAT_COL | LON_COL
    trh: float | None = None,      # symmetric y-limit (only used for dtec)
    cmap_name: str = "tab20",
    s: float = 10.0,
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Draw the scatter. X axis is UTC time (decimal hours); Y axis defaults to
    dTEC but can be latitude or longitude to inspect the geometry of each pass.
    Every receiver-PRN pair gets its own color (and the same color is reused
    across panels if you extend this later).
    """
    movavg = (
        int(re.findall(r"\d+", y_col)[0]) if re.search(r"\d", y_col) else None
    )
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(chosen_pairs))]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    for color, pair in zip(colors, chosen_pairs):
        sub = df[df["pair"] == pair].sort_values("t_hours")
        ax.scatter(
            sub["t_hours"], sub[y_col],
            s=s, color=color, label=pair,
            edgecolors="none", alpha=0.85, zorder=3,
        )

    # --- Y axis ---
    if y_col == LAT_COL:
        ax.set_ylabel("IPP latitude, deg")
    elif y_col == LON_COL:
        ax.set_ylabel("IPP longitude, deg")
    else:
        ax.set_ylabel("dTEC (TECU)")
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="-", zorder=1)
        if trh is not None:
            ax.set_ylim(-trh, trh)

    # --- X axis (time) ---
    h0 = float(df["t_hours"].min())
    h1 = float(df["t_hours"].max())
    ax.set_xlim(h0, h1)
    ax.set_xlabel("Time (UTC, hours)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.grid(True, which="major", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    title = (
        f"dTEC scatter by receiver-PRN — {year} DOY {doy:03d}"
        f"  |  {len(chosen_pairs)} pair(s)"
    )
    if movavg is not None:
        title += f"  |  {movavg}-min moving average"
    ax.set_title(title)

    ncol = 1 if len(chosen_pairs) <= 16 else 2
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, ncol=ncol, frameon=False, title="receiver · PRN")

    if out_path is None:
        out_path = (
            Path(dtec_keogram_figs_dir())
            / f"scatter_{year}{doy:03d}_pairs.png"
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
    color_by: str = "dtec",        # "dtec" (shared cbar) | "pair" (categorical + legend)
    trh: float | None = None,      # symmetric dTEC limit (color_by="dtec")
    cmap_name: str = "tab20",      # categorical cmap (color_by="pair")
    dtec_cmap: str = "seismic",    # continuous cmap (color_by="dtec")
    s: float = 12.0,
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Two stacked panels sharing the time (X) axis, mirroring the keogram figure:
        top    -> IPP longitude vs time
        bottom -> IPP latitude  vs time

    Each receiver-PRN pair is one curve in BOTH panels, drawn in time order so
    its longitude/latitude evolution reads as a trajectory. The grouping key is
    the (receiver, PRN) pair — the same PRN seen by two receivers gives two
    distinct curves, never merged.

      - color_by="pair" : each pair has its own color (consistent across panels)
                          with a single shared legend.
      - color_by="dtec" : every point colored by dTEC on a shared colorbar.
    """
    movavg = (
        int(re.findall(r"\d+", dtec_col)[0]) if re.search(r"\d", dtec_col) else None
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             constrained_layout=True, sharex=True)

    def _draw(ax, y_col, color, sub):
        ax.plot(sub["t_hours"], sub[y_col],
                color=color if color_by == "pair" else "0.6",
                linewidth=0.7, alpha=0.6, zorder=2)

    if color_by == "pair":
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(len(chosen_pairs))]
        for color, pair in zip(colors, chosen_pairs):
            sub = df[df["pair"] == pair].sort_values("t_hours")
            # top: longitude
            _draw(axes[0], LON_COL, color, sub)
            axes[0].scatter(sub["t_hours"], sub[LON_COL], color=color, s=s,
                            edgecolors="none", alpha=0.9, label=pair, zorder=3)
            # bottom: latitude
            _draw(axes[1], LAT_COL, color, sub)
            axes[1].scatter(sub["t_hours"], sub[LAT_COL], color=color, s=s,
                            edgecolors="none", alpha=0.9, zorder=3)

        ncol = 1 if len(chosen_pairs) <= 18 else 2
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, -0.05),
                       fontsize=8, ncol=ncol, frameon=False,
                       title="receiver · PRN")

    elif color_by == "dtec":
        vmax = trh if trh is not None else float(
            np.nanmax(np.abs(df.loc[df["pair"].isin(chosen_pairs), dtec_col]))
        )
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.get_cmap(dtec_cmap)
        for pair in chosen_pairs:
            sub = df[df["pair"] == pair].sort_values("t_hours")
            # pure scatter (no connecting line); color encodes dTEC
            axes[0].scatter(sub["t_hours"], sub[LON_COL], c=sub[dtec_col],
                            cmap=cmap, norm=norm, s=s, edgecolors="none", zorder=3)
            axes[1].scatter(sub["t_hours"], sub[LAT_COL], c=sub[dtec_col],
                            cmap=cmap, norm=norm, s=s, edgecolors="none", zorder=3)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.7, pad=0.02)
        cb.set_label("dTEC (TECU)")
    else:
        raise ValueError(f"Unknown color_by: {color_by!r} (use 'pair' or 'dtec').")

    # --- Top: longitude ---
    axes[0].set_ylabel("IPP longitude, deg")
    title = (
        f"IPP trajectories by receiver-PRN — {year} DOY {doy:03d}"
        f"  |  {len(chosen_pairs)} pair(s)"
    )
    if movavg is not None:
        title += f"  |  {movavg}-min moving average"
    axes[0].set_title(title)
    axes[0].grid(True, which="major", linestyle=":", alpha=0.4)
    axes[0].grid(True, which="minor", linestyle=":", alpha=0.2)

    # --- Bottom: latitude ---
    axes[1].set_ylabel("IPP latitude, deg")
    axes[1].set_xlabel("Time (UTC, hours)")
    axes[1].grid(True, which="major", linestyle=":", alpha=0.4)
    axes[1].grid(True, which="minor", linestyle=":", alpha=0.2)

    # --- shared X (time) ---
    h0 = float(df.loc[df["pair"].isin(chosen_pairs), "t_hours"].min())
    h1 = float(df.loc[df["pair"].isin(chosen_pairs), "t_hours"].max())
    axes[1].set_xlim(h0, h1)
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    if out_path is None:
        out_path = (
            Path(dtec_keogram_figs_dir())
            / f"trajectory_{year}{doy:03d}_pairs.png"
        )
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
    parser.add_argument("--y-col", type=str, default=None,
                        help="Y axis column (defaults to --dtec-col).")
    parser.add_argument("--mode", choices=["all", "topN", "list"], default="topN")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--trh", type=float, default=None)
    parser.add_argument("--kind", choices=["timeseries", "trajectory"],
                        default="timeseries",
                        help="timeseries = time x dTEC; trajectory = lon x lat arcs.")
    parser.add_argument("--color-by", choices=["dtec", "pair"], default="dtec",
                        help="Trajectory coloring (ignored for timeseries).")
    return parser.parse_args()


# ==========================================================================
# 5. Entry point: one-shot pipeline
# ==========================================================================
def run(
    year, doy,
    lat_range, lon_range, time_range,
    dtec_col="dtec_15min",
    kind="timeseries",            # "timeseries" | "trajectory"
    y_col=None,                   # timeseries only
    color_by="dtec",              # trajectory only: "dtec" | "pair"
    mode="topN", n=12, pairs=None, min_points=5,
    trh=None,
    stations=None,
    out_path=None,
):
    y_col = y_col or dtec_col
    df = load_window(
        year, doy,
        lat_range=lat_range, lon_range=lon_range, time_range=time_range,
        dtec_col=dtec_col, stations=stations,
    )
    print(f"[load] {len(df)} IPPs in window | "
          f"{df['station'].nunique()} stations | "
          f"{df['pair'].nunique()} receiver-PRN pairs")

    chosen = select_pairs(df, mode=mode, n=n, pairs=pairs, min_points=min_points)

    if kind == "trajectory":
        return plot_pairs_trajectory(
            df, chosen, year=year, doy=doy,
            dtec_col=dtec_col, color_by=color_by, trh=trh, out_path=out_path,
        )
    elif kind == "timeseries":
        return plot_pairs_scatter(
            df, chosen, year=year, doy=doy,
            y_col=y_col, trh=trh, out_path=out_path,
        )
    else:
        raise ValueError(f"Unknown kind: {kind!r} (use 'timeseries' or 'trajectory').")


# ==========================================================================
# 6. IDE / CLI entry point
# ==========================================================================
if __name__ == "__main__":

    RUN_MANUALLY = True
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")

        year, doy = 2025, 298

        # --- spatio-temporal cut --------------------------------------------
        LAT_RANGE  = (-32.01, -16.99)     # deg
        LON_RANGE  = (-65.01, -39.99)     # deg
        TIME_RANGE = (12.0, 20.0)         # UTC decimal hours

        DTEC_COL = "dtec_15min"

        # --- which pairs to draw --------------------------------------------
        # "topN" -> longest N passes; "all" -> everything; "list" -> explicit
        #
        # kind="trajectory" -> each receiver-PRN pair is its OWN lon x lat arc
        #                      (same PRN from different receivers = different arc)
        # kind="timeseries" -> time x dTEC, one series per pair
        run(
            year=year, doy=doy,
            lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
            dtec_col=DTEC_COL,
            kind="trajectory",     # two panels: lon (top) & lat (bottom) vs time
            # kind="timeseries",     
            color_by="dtec",       # scatter colored by dTEC (seismic colorbar)
            mode="topN", n=12, min_points=10,
            trh=0.201,               # set None to autoscale
            # mode="list",
            # pairs=[("ALAR", 7), ("ALAR", 30), ("BRAZ", 9)],
        )
    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        run(
            year=args.year, doy=args.doy,
            lat_range=tuple(args.lat), lon_range=tuple(args.lon),
            time_range=tuple(args.time),
            dtec_col=args.dtec_col, kind=args.kind,
            y_col=args.y_col, color_by=args.color_by,
            mode=args.mode, n=args.n, trh=args.trh,
        )