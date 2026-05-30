# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:42:57 2026

@author: hisas
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# project data + output locators (adjust import to your layout)
from shared.config import storms_SouthAmerica, dtec_keogram_figs_dir

def track_storms(
    df: pd.DataFrame,
    max_link_dist_deg: float = 2.0,   # max centroid jump between frames to link
    max_gap_frames: int = 1,          # tolerated consecutive missing frames
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
    time_col: str = "datetime",
) -> pd.DataFrame:
    """
    Greedy nearest-neighbour tracker for storm-cell detections that carry no
    native track identity. Detections in consecutive frames are linked when
    their centroids lie within `max_link_dist_deg` degrees; gaps of up to
    `max_gap_frames` empty frames are tolerated before a track is closed.
    One-to-one per frame: each track extends to at most one detection and
    vice-versa, assigning the closest pairs first.

    Returns the input dataframe (time-sorted) with an added int `track_id`.
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df["track_id"] = -1

    frames = list(df.groupby(time_col, sort=True).groups.items())  # (ts, index)

    active = []        # list of {id, lat, lon, last_frame}
    next_id = 0

    for frame_no, (_ts, idx) in enumerate(frames):
        det_idx = np.asarray(idx)
        det_lat = df.loc[det_idx, lat_col].to_numpy()
        det_lon = df.loc[det_idx, lon_col].to_numpy()

        # drop tracks idle for longer than the allowed gap
        active = [t for t in active
                  if frame_no - t["last_frame"] <= max_gap_frames + 1]

        # build all admissible (distance, track, detection) links
        candidates = []
        for ti, t in enumerate(active):
            d = np.hypot(det_lat - t["lat"], det_lon - t["lon"])
            for di, dist in enumerate(d):
                if dist <= max_link_dist_deg:
                    candidates.append((dist, ti, di))
        candidates.sort()                       # nearest first

        used_tracks, used_dets = set(), set()
        for dist, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            tid = active[ti]["id"]
            df.at[det_idx[di], "track_id"] = tid
            active[ti].update(lat=det_lat[di], lon=det_lon[di],
                              last_frame=frame_no)

        # unmatched detections start new tracks
        for di in range(len(det_idx)):
            if di in used_dets:
                continue
            df.at[det_idx[di], "track_id"] = next_id
            active.append(dict(id=next_id, lat=det_lat[di],
                               lon=det_lon[di], last_frame=frame_no))
            next_id += 1

    return df

def _distinct_track_colors(n: int):
    """Up to 60 qualitatively distinct colors, then continuous fallback."""
    base = list(plt.get_cmap("tab10").colors)
    if n <= len(base):
        return base[:n]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_storms_tracked(
    year: int,
    doy: int,
    lat_range: tuple,
    lon_range: tuple,
    time_range: tuple | None = None,
    temp_col: str = "temp_min_C",
    max_link_dist_deg: float = 2.0,   # tracker: max centroid jump per frame
    max_gap_frames: int = 1,          # tracker: tolerated missing frames
    min_track_len: int = 3,           # drop tracks shorter than this
    pad_deg: float = 3.0,             # spatial padding around the lat/lon box
    out_path: Path | None = None,
    dpi: int = 300,
):
    """
    Four stacked panels sharing the time (X) axis, complementary to the TTD:
        1  longitude of storm centroid vs time
        2  latitude  of storm centroid vs time
        3  cloud-top temperature       vs time
        4  dT/dt (temperature rate)    vs time

    Storms are tracked across frames (greedy nearest-neighbour) and each track
    gets a fixed color reused on all four panels, so a single storm can be
    followed by color throughout the figure.
    """
    # --- load + clip to day and (padded) box ---------------------------
    df_storms = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")

    t0 = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    t1 = t0 + pd.Timedelta(days=1)
    lat_lo, lat_hi = sorted(lat_range)
    lon_lo, lon_hi = sorted(lon_range)
    df_storms = df_storms[
        (df_storms["datetime"] >= t0) & (df_storms["datetime"] < t1) &
        df_storms["centroid_lat"].between(lat_lo - pad_deg, lat_hi + pad_deg) &
        df_storms["centroid_lon"].between(lon_lo - pad_deg, lon_hi + pad_deg)
    ].copy()

    if df_storms.empty:
        print("[storms] no detections in day + box — nothing to plot.")
        return None

    # normalize tz and build decimal-hour axis
    dt = pd.to_datetime(df_storms["datetime"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(None)
    df_storms["datetime"] = dt
    df_storms["t_hours"] = (dt - t0).dt.total_seconds().to_numpy() / 3600.0

    # --- cut to the requested time window BEFORE tracking --------------
    if time_range is not None:
        h0, h1 = time_range
        df_storms = df_storms[df_storms["t_hours"].between(h0, h1)].copy()
        if df_storms.empty:
            print("[storms] no detections inside the time window — nothing to plot.")
            return None

    # --- track + drop short tracks -------------------------------------
    tracked = track_storms(df_storms, max_link_dist_deg=max_link_dist_deg,
                            max_gap_frames=max_gap_frames)

    # --- track + drop short tracks -------------------------------------
    tracked = track_storms(df_storms, max_link_dist_deg=max_link_dist_deg,
                            max_gap_frames=max_gap_frames)
    counts = tracked["track_id"].value_counts()
    keep = counts[counts >= min_track_len].index
    tracked = tracked[tracked["track_id"].isin(keep)].copy()

    if tracked.empty:
        print(f"[storms] no track survives min_track_len={min_track_len}.")
        return None

    track_ids = sorted(tracked["track_id"].unique())
    palette   = _distinct_track_colors(len(track_ids))
    color_of  = {tid: palette[i] for i, tid in enumerate(track_ids)}
    print(f"[storms] {len(track_ids)} tracks "
          f"(>= {min_track_len} detections) from {len(tracked)} detections.")
    
        # --- font sizes ----------------------------------------------------
    plt.rcParams.update({
        "font.size":        14,   # base (tick labels, etc.)
        "axes.titlesize":   16,   # panel title
        "axes.labelsize":   14,   # x/y axis labels
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
    })

    # --- figure: 4 panels, shared X ------------------------------------
    fig, (ax_lon, ax_lat, ax_temp, ax_dt) = plt.subplots(
        4, 1, figsize=(14, 14), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 2]},
    )

    for tid in track_ids:
        g = tracked[tracked["track_id"] == tid].sort_values("datetime")
        t    = g["t_hours"].to_numpy()
        lon  = g["centroid_lon"].to_numpy()
        lat  = g["centroid_lat"].to_numpy()
        temp = g[temp_col].to_numpy()
        dTdt = np.gradient(temp, t) if len(t) > 1 else np.full(len(t), np.nan)
        c    = color_of[tid]

        ax_lon.plot(t, lon,  "-o", color=c, ms=5, lw=2.0)
        ax_lat.plot(t, lat,  "-o", color=c, ms=5, lw=2.0)
        ax_temp.plot(t, temp, "-o", color=c, ms=5, lw=2.0)
        ax_dt.plot(t, dTdt,  "-o", color=c, ms=5, lw=2.0)

        # genesis marker (open square) on the spatial panels
        ax_lon.scatter(t[0], lon[0], marker="s", s=70, facecolors="none",
                       edgecolors=c, linewidths=2.0, zorder=5)
        ax_lat.scatter(t[0], lat[0], marker="s", s=70, facecolors="none",
                       edgecolors=c, linewidths=2.0, zorder=5)
        
    # --- cosmetics -----------------------------------------------------
    ax_lon.set_ylabel("centroid longitude, deg")
    ax_lat.set_ylabel("centroid latitude, deg")
    ax_temp.set_ylabel("cloud-top T (°C)")
    ax_dt.set_ylabel("dT/dt (°C / h)")
    ax_dt.set_xlabel("Time (UTC, hours)")

    # fixed Y limits for temperature and derivative panels
    ax_temp.set_ylim(-95.0, -65.0)      # cloud-top T (°C)
    ax_dt.set_ylim(-40.0, 40.0)         # dT/dt (°C / h)

    ax_dt.axhline(0, color="0.4", lw=0.8, ls="--", zorder=1)

    if time_range is not None:
        ax_dt.set_xlim(float(time_range[0]), float(time_range[1]))
    else:
        ax_dt.set_xlim(float(tracked["t_hours"].min()),
                       float(tracked["t_hours"].max()))
    ax_dt.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_dt.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    for ax in (ax_lon, ax_lat, ax_temp, ax_dt):
        ax.grid(True, linestyle=":", alpha=0.4)

    ax_lon.set_title(
        f"STORM TRACKS: {year}-{doy:03d}  |  {len(track_ids)} tracks  "
        f"(>= {min_track_len} det., link <= {max_link_dist_deg}°)"
    )

    if out_path is None:
        lo = time_range[0] if time_range is not None else int(tracked['t_hours'].min())
        hi = time_range[1] if time_range is not None else int(tracked['t_hours'].max())
        out_path = (dtec_keogram_figs_dir()
                    / f"storms_{year}-{doy:03d}-{lo}-{hi}.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return out_path