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
    for p in chosen:
        print(f"    {p:<28s}  {int(counts.get(p, 0)):>5d} pts")
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
            Path(dtec_keogram_figs_dir())
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
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Two stacked panels sharing the time (X) axis (same layout as the keogram):
        top    -> IPP longitude vs time
        bottom -> IPP latitude  vs time
 
    Each point is colored by its dTEC on a shared colorbar. The receiver-PRN
    identity is preserved by the upstream selection (`chosen_pairs`) — the
    same PRN seen by two receivers contributes two distinct sets of points,
    never merged.
    """
    sub = df.loc[df["pair"].isin(chosen_pairs)]
 
    vmax = trh if trh is not None else float(np.nanmax(np.abs(sub[dtec_col])))

    
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
 
    fig, (ax_lon, ax_lat) = plt.subplots(
        2, 1, figsize=(14, 9), constrained_layout=True, sharex=True,
    )
    
    w     = np.clip(np.abs(sub[dtec_col]) / vmax, 0.0, 1.0)
    # alpha = 0.10 + 0.90 * w        # 0.15 (fraco, quase invisível) -> 1.0 (forte)
    gamma  = 2.0                            # 1 = linear, >1 = penaliza mais os fracos
    alpha = 0.10 + 0.90 * (w ** gamma)

    ax_lon.scatter(sub["t_hours"], sub[LON_COL], c=sub[dtec_col], s=abs(sub[dtec_col])*50,
                   cmap=cmap, norm=norm,  edgecolors="none", alpha=alpha)
    ax_lat.scatter(sub["t_hours"], sub[LAT_COL], c=sub[dtec_col], s=abs(sub[dtec_col])*50,
                   cmap=cmap, norm=norm,  edgecolors="none", alpha=alpha)
 
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    fig.colorbar(sm, ax=(ax_lon, ax_lat), location="right",
                 shrink=1.0, pad=0.02, label="dTEC (TECU)")
 
    ax_lon.set_ylabel("IPP longitude, deg")
    ax_lat.set_ylabel("IPP latitude, deg")
    ax_lat.set_xlabel("Time (UTC, hours)")
    ax_lat.set_xlim(float(sub["t_hours"].min()), float(sub["t_hours"].max()))
    ax_lat.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_lat.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    for ax in (ax_lon, ax_lat):
        ax.grid(True, linestyle=":", alpha=0.4)
 
    ax_lon.set_title(
        f"IPP trajectories by receiver-PRN — {year} DOY {doy:03d}"
        f"  |  {len(chosen_pairs)} pair(s)"
    )
 
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
    parser.add_argument("--offset", type=float, default=1.0,
                        help="Vertical offset in TECU between successive pairs "
                             "in the line stack (timeseries kind).")
    parser.add_argument("--mode", choices=["all", "topN", "list"], default="topN")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--trh", type=float, default=None)
    parser.add_argument("--kind", choices=["timeseries", "trajectory"],
                        default="timeseries",
                        help="timeseries = dTEC line stack; "
                             "trajectory = lon/lat vs time (two panels).")
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
    kind="timeseries",            # "timeseries" (dTEC line stack) | "trajectory"
    offset=1.0,                   # timeseries only: vertical offset (TECU)
    color_by="dtec",              # trajectory only: "dtec" | "pair"
    mode="topN", n=12, pairs=None, min_points=5,
    trh=None,
    stations=None,
    out_path=None,
):
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
            dtec_col=dtec_col,  trh=trh, out_path=out_path,
        )
    elif kind == "timeseries":
        return plot_pairs_lines(
            df, chosen, year=year, doy=doy,
            dtec_col=dtec_col, offset=offset, trh=trh, out_path=out_path,
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

        year, doy = 2021, 296

        # --- spatio-temporal cut --------------------------------------------
        LAT_RANGE  = (-30.01, -19.99)     # deg
        LON_RANGE  = (-60.01, -39.99)     # deg
        TIME_RANGE = (16, 18)         # UTC decimal hours

        DTEC_COL = "dtec_10min"

        # --- which pairs to draw --------------------------------------------
        # "topN" -> longest N passes; "all" -> everything; "list" -> explicit
        #
        # kind="timeseries" -> dTEC LINE STACK, one curve per pair offset
        #                      vertically by `offset` TECU from the previous
        # kind="trajectory" -> two panels: lon (top) & lat (bottom) vs time,
        #                      scatter colored by dTEC
        # run(
        #     year=year, doy=doy,
        #     lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
        #     dtec_col=DTEC_COL,
        #     kind="timeseries",     # dTEC line stack with offsets
        #     offset=.05,            # TECU between successive pairs
        #     # kind="trajectory",   # uncomment for lon/lat vs time panels
        #     # color_by="dtec",     # trajectory only
        #     mode="topN", n=30, min_points=10,
        #     # mode="list",
        #     # pairs=[("ALAR", 7), ("ALAR", 30), ("BRAZ", 9)],
        # )
        
        run(
            year=year, doy=doy,
            lat_range=LAT_RANGE, lon_range=LON_RANGE, time_range=TIME_RANGE,
            dtec_col=DTEC_COL,
            kind="trajectory",   # uncomment for lon/lat vs time panels
            color_by="dtec",     # trajectory only
            mode="topN", n=1000, min_points=10,
            trh=0.301,               # symmetric clip per curve (set None to disable)
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
            offset=args.offset, color_by=args.color_by,
            mode=args.mode, n=args.n, trh=args.trh,
        )