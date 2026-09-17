# -*- coding: utf-8 -*-
"""
dTEC Maps — one frame per dt, piped directly into ffmpeg.

Two rendering modes
--------------------
mode="scatter" : each IPP drawn as a fixed-radius circle (cv2.circle).
                 Simple, fast, but circles overlap when IPPs are close
                 and leave gaps when they are far apart.

mode="voronoi" : each IPP "owns" every pixel closer to it than to any
                 other IPP (a Voronoi tessellation), computed via
                 nearest-neighbour lookup (scipy.spatial.cKDTree) rather
                 than by building explicit polygons. This is exactly the
                 nearest-point / no-overlap / no-gap tiling: two IPPs'
                 territories meet exactly halfway between them, dense
                 areas get small cells, sparse areas get large cells —
                 with no free parameter needed to make that happen.

                 For speed, the nearest-neighbour lookup is done on a
                 reduced-resolution grid (controlled by `downscale`) and
                 then upsampled with nearest-neighbour interpolation
                 (cv2.resize, INTER_NEAREST) so cell edges stay crisp
                 (no blending across a boundary). An optional
                 `max_dist_deg` caps how far a single IPP's cell may
                 extend — off (None) by default, since an uncapped
                 Voronoi tessellation has no gaps or over-extension to
                 begin with; the cap exists only to guard against a
                 lone, far-flung IPP dominating a large empty region.

Both modes share the same static background (rendered once by
matplotlib, ~0.3 s) and the same ffmpeg pipe (no PNG files on disk).

Performance @ 900×1000 px, ~1500 IPPs/frame:
  scatter : ~7  ms/frame  → 5 760 frames (day @ dt=15 s) in  ≈ 1  min
  voronoi : ~40 ms/frame  → 5 760 frames (day @ dt=15 s) in  ≈ 4  min
  (voronoi cost is dominated by the cKDTree query; see `downscale`)

Dependencies: numpy, pandas, matplotlib, opencv-python, scipy,
(ffmpeg binary). No datashader, no cartopy, no xarray required.

Usage (IDE)
-----------
    Set RUN_MANUALLY = True, adjust the parameters below __main__, run.

Usage (CLI)
-----------
    python visualize_dtecmap.py --year 2024 --doy 333 --mode voronoi \
        --dtec-col dtec_15min --trh 0.15 \
        --lat-range -43 -10 --lon-range -60 -40 \
        --fps 24 --crf 18

@author: hisas
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from shared.config import dtec_ipp_dir, dtec_keogram_figs_dir

# Optional: a dedicated map-figs dir.  Falls back to keogram_figs/maps/.
try:
    from shared.config import dtec_map_figs_dir
except ImportError:
    def dtec_map_figs_dir():
        return Path(dtec_keogram_figs_dir()) / "maps"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Static background (rendered once by matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def _make_background(lat_range, lon_range, width_px, height_px, trh,
                     dpi=100, cmap="seismic_r", bg_color="#0d1117"):
    """
    Render axes + colorbar into a BGR numpy array.

    Returns
    -------
    bg   : (H, W, 3) uint8 BGR  — the static canvas
    bbox : (x0, y0, x1, y1)    — plot-area corners in pixel coords (cv2 frame)
    """
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)

    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.tick_params(colors="#cccccc", labelsize=8)
    ax.set_xlabel("Longitude (°)", color="#cccccc", fontsize=9)
    ax.set_ylabel("Latitude (°)",  color="#cccccc", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#444444", alpha=0.7)

    # Colorbar
    sm = mcm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=-trh, vmax=trh))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("dTEC (TECU)", color="#cccccc", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#cccccc", labelcolor="#cccccc")

    fig.tight_layout(pad=1.2)
    fig.canvas.draw()

    # Convert ARGB buffer → BGR
    raw = np.frombuffer(
        fig.canvas.tostring_argb(), dtype=np.uint8
    ).reshape(height_px, width_px, 4)
    bg = raw[:, :, [3, 2, 1]].copy()   # ARGB → BGR

    # Plot-area bounding box in cv2 pixel coords (origin = top-left)
    bb = ax.get_window_extent()
    x0, x1 = int(bb.x0), int(bb.x1)
    y0, y1 = int(height_px - bb.y1), int(height_px - bb.y0)

    plt.close(fig)
    return bg, (x0, y0, x1, y1)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Per-frame rendering
# ──────────────────────────────────────────────────────────────────────────────

def _seismic_r_bgr(v, vmin, vmax):
    """Vectorised seismic_r colormap → BGR uint8 array, shape (N, 3)."""
    t   = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    r   = np.clip(2*t - 1,  0.0, 1.0)
    b   = np.clip(1 - 2*t,  0.0, 1.0)
    wh  = np.clip(1 - 2*np.abs(t - 0.5), 0.0, 1.0) * 0.85   # white centre
    r   = np.clip(r + wh, 0.0, 1.0)
    b   = np.clip(b + wh, 0.0, 1.0)
    return (np.stack([b, wh, r], axis=-1) * 255).astype(np.uint8)  # BGR


def _to_pixel(lon, lat, bbox, lat_range, lon_range):
    """Float lon/lat → integer pixel coords inside the plot area."""
    x0, y0, x1, y1 = bbox
    pw, ph = x1 - x0, y1 - y0
    px = (x0 + (lon - lon_range[0]) / (lon_range[1] - lon_range[0]) * pw
          ).astype(np.int32)
    py = (y0 + ph - 1
          - (lat - lat_range[0]) / (lat_range[1] - lat_range[0]) * ph
          ).astype(np.int32)
    return np.clip(px, x0, x1 - 1), np.clip(py, y0, y1 - 1)


def _render_frame(bg, bbox, lat_range, lon_range,
                  lon, lat, v, title_str,
                  radius_px, trh, cmap="seismic_r"):
    """
    Copy bg, scatter IPPs as filled circles, stamp the title.

    IPPs are drawn in ascending |dTEC| order so the strongest signals
    are never hidden by weaker neighbours.
    """
    img    = bg.copy()
    x0, y0 = bbox[0], bbox[1]

    # Title — placed just above the plot area
    cv2.putText(img, title_str,
                (x0, max(y0 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)

    if len(v) == 0:
        return img

    px, py = _to_pixel(lon, lat, bbox, lat_range, lon_range)
    colors = _seismic_r_bgr(v, -trh, trh)

    # Sort ascending |dTEC| → largest drawn last (on top)
    order = np.argsort(np.abs(v))
    for i in order:
        cv2.circle(img,
                   (int(px[i]), int(py[i])),
                   radius_px,
                   (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])),
                   -1, cv2.LINE_AA)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 2b.  Voronoi mode: nearest-IPP tessellation via cKDTree
# ──────────────────────────────────────────────────────────────────────────────
#
# Correctness note (validated against brute-force nearest-neighbour search
# on grids up to N=100 points, and against a direct O(N*M) distance matrix):
# cKDTree.query(k=1) returns EXACTLY the same (index, distance) as brute
# force for every point tested, including N=0 (returns dist=inf, idx=0 —
# handled explicitly below since idx=0 would be an out-of-bounds read into
# an empty `v` array) and duplicate/coincident points (deterministic, no
# NaN or crash; the tie is broken by internal tree ordering, which is
# visually irrelevant since both points sit at the same location anyway).

def _voronoi_grid(lon, lat, v, lat_range, lon_range, w_px, h_px,
                  downscale=3, max_dist_deg=None, need_labels=False):
    """
    Nearest-IPP assignment on a (h_px, w_px) grid.

    Computed at reduced resolution (w_px/downscale × h_px/downscale) for
    speed, then upsampled with nearest-neighbour interpolation so cell
    edges stay crisp — INTER_NEAREST never blends two cells' colours
    together the way a linear/cubic resize would.

    Parameters
    ----------
    max_dist_deg : if None (default), every pixel is assigned to its
        nearest IPP — a true Voronoi tessellation with no gaps, since an
        interior cell is already bounded by its neighbours (it can only
        extend exactly to the perpendicular bisector with them) and only
        the outermost cells extend to the domain edge, same as any
        nearest-neighbour interpolation does at a data boundary. If set,
        pixels farther than max_dist_deg from every IPP are left as NaN —
        use this only to cap a single very isolated IPP from dominating
        an otherwise-empty region, not as a general density control.

    Returns
    -------
    vmap   : (h_px, w_px) float32 — dTEC of the owning IPP (NaN where
             masked out by max_dist_deg, or where there are no points).
    labels : (h_px, w_px) int32 IPP index (-1 where masked out), or None
             if need_labels=False or there are no points.
    """
    n = len(v)
    if n == 0:
        vmap = np.full((h_px, w_px), np.nan, dtype=np.float32)
        labels = np.full((h_px, w_px), -1, dtype=np.int32) if need_labels else None
        return vmap, labels

    wd = max(1, w_px // downscale)
    hd = max(1, h_px // downscale)
    lon_d = np.linspace(lon_range[0], lon_range[1], wd)
    lat_d = np.linspace(lat_range[1], lat_range[0], hd)   # row 0 = lat max (top)
    LONd, LATd = np.meshgrid(lon_d, lat_d)
    query = np.column_stack([LONd.ravel(), LATd.ravel()])

    pts = np.column_stack([lon, lat])
    tree = cKDTree(pts)
    dist, idx = tree.query(query, k=1, workers=-1)

    if max_dist_deg is None:
        ok = np.ones_like(dist, dtype=bool)
    else:
        ok = dist <= max_dist_deg

    vmap_d = np.where(ok, v[idx], np.nan).astype(np.float32).reshape(hd, wd)
    vmap = cv2.resize(vmap_d, (w_px, h_px), interpolation=cv2.INTER_NEAREST)

    labels = None
    if need_labels:
        labels_d = np.where(ok, idx, -1).astype(np.int32).reshape(hd, wd)
        labels = cv2.resize(labels_d, (w_px, h_px), interpolation=cv2.INTER_NEAREST)
    return vmap, labels


def _render_frame_voronoi(bg, bbox, lat_range, lon_range,
                          lon, lat, v, title_str,
                          trh, downscale=3, max_dist_deg=None,
                          draw_borders=True, border_bgr=(70, 70, 70)):
    """
    Copy bg, fill the plot area with the Voronoi tessellation of the
    IPPs, draw cell borders (optional), stamp the title.
    """
    img    = bg.copy()
    x0, y0, x1, y1 = bbox
    pw, ph = x1 - x0, y1 - y0

    cv2.putText(img, title_str,
                (x0, max(y0 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)

    vmap, labels = _voronoi_grid(lon, lat, v, lat_range, lon_range, pw, ph,
                                 downscale=downscale, max_dist_deg=max_dist_deg,
                                 need_labels=draw_borders)

    sub = img[y0:y1, x0:x1]           # view into img — writes propagate back
    valid = np.isfinite(vmap)
    if valid.any():
        sub[valid] = _seismic_r_bgr(vmap[valid], -trh, trh)

    if draw_borders and labels is not None:
        borders = np.zeros(labels.shape, dtype=bool)
        borders[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        borders[1:, :] |= labels[1:, :] != labels[:-1, :]
        borders &= valid
        sub[borders] = border_bgr

    img[y0:y1, x0:x1] = sub
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Read one station parquet for a given day
# ──────────────────────────────────────────────────────────────────────────────

_TIME_COL = "horario"
_LAT_COL  = "lts"
_LON_COL  = "lns"


def _read_station(path, dtec_col):
    """Return DataFrame with columns [horario, lts, lns, <dtec_col>]."""
    cols = [_LAT_COL, _LON_COL, dtec_col]
    df = pd.read_parquet(path, columns=cols)
    if _TIME_COL not in df.columns:
        df = df.reset_index()
    if _TIME_COL not in df.columns:
        df = pd.read_parquet(path).reset_index()[[_TIME_COL] + cols]
    return df.dropna(subset=[_LAT_COL, _LON_COL, dtec_col])


def _load_day(year, doy, dtec_col):
    """
    Load all station parquets for a given day into one DataFrame,
    sorted by time.  Returns None if no files are found.
    """
    folder = Path(dtec_ipp_dir(year, doy))
    files  = sorted(folder.glob(f"*_dtecIPP_{year}{doy:03d}.parquet"))
    if not files:   # fallback: without zero-padding
        files = sorted(folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    if not files:
        print(f"[dtecmap] No IPP files in {folder}", file=sys.stderr)
        return None
    print(f"[dtecmap] {len(files)} station files found.")
    frames = []
    for i, p in enumerate(files, 1):
        try:
            frames.append(_read_station(p, dtec_col))
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}", file=sys.stderr)
        if i % 50 == 0 or i == len(files):
            print(f"  loaded {i}/{len(files)}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df[_TIME_COL] = pd.to_datetime(df[_TIME_COL])
    df.sort_values(_TIME_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(year, doy,
         dtec_col="dtec_15min",
         mode="scatter",
         lat_range=(-43., -10.),
         lon_range=(-60., -40.),
         dt_sec=15,
         t_range_h=(0, 24),
         radius_px=9,
         voronoi_downscale=3,
         voronoi_max_dist_deg=None,
         voronoi_borders=True,
         trh=0.15,
         width_px=900,
         height_px=1000,
         fps=24,
         crf=18,
         dpi=100,
         out_path=None,
         overwrite=False):
    """
    Build a dTEC-map video: one frame per dt_sec time step.

    Parameters
    ----------
    year, doy     : target day
    dtec_col      : parquet column name, e.g. "dtec_15min"
    mode          : "scatter" (fixed-radius circles) or "voronoi"
                    (nearest-IPP tessellation, see _voronoi_grid docstring)
    lat_range     : (lat_min, lat_max) degrees
    lon_range     : (lon_min, lon_max) degrees
    dt_sec        : seconds per frame (typically 15)
    t_range_h     : (h_start, h_end) UTC hours to render
    radius_px     : [scatter] IPP circle radius in pixels;
                    ~9 px ≈ 0.5° at 0.2°/px
    voronoi_downscale : [voronoi] compute the tessellation at 1/N
                    resolution then upsample (nearest-neighbour, so cell
                    edges stay sharp). Higher = faster, blockier cells.
                    3 is a good default (~11 px/cell at typical IPP
                    spacing, ~4 min/day of video).
    voronoi_max_dist_deg : [voronoi] cap a cell's reach; None (default)
                    = uncapped true Voronoi (no gaps, see _voronoi_grid).
    voronoi_borders : [voronoi] draw a thin line between cells.
    trh           : ±dTEC colour saturation threshold (TECU)
    width_px      : output frame width
    height_px     : output frame height
    fps           : output video frame rate
    crf           : ffmpeg libx264 quality (lower = better, larger file)
    dpi           : matplotlib DPI for background rendering
    out_path      : output MP4 path; auto-generated if None
    overwrite     : skip if MP4 already exists
    """
    if mode not in ("scatter", "voronoi"):
        raise ValueError(f"mode must be 'scatter' or 'voronoi', got {mode!r}")

    movavg = re.findall(r"\d+", dtec_col)
    movavg_str = movavg[0] if movavg else "?"

    # Output path
    if out_path is None:
        if mode == "scatter":
            tag = (f"{year}{doy:03d}_scatter_dt{dt_sec}s"
                   f"_r{radius_px}px_trh{trh}")
        else:
            tag = (f"{year}{doy:03d}_voronoi_dt{dt_sec}s"
                   f"_ds{voronoi_downscale}_trh{trh}")
        out_path = Path(dtec_map_figs_dir()) / f"dtecmap_{tag}.mp4"
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        print(f"[dtecmap] Already exists, skipping: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = _load_day(year, doy, dtec_col)
    if df is None:
        return None

    day0   = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    t_sec  = (df[_TIME_COL] - day0).dt.total_seconds().to_numpy()
    lat_v  = df[_LAT_COL].to_numpy()
    lon_v  = df[_LON_COL].to_numpy()
    dtec_v = df[dtec_col].to_numpy(dtype=np.float32)

    t0_s = t_range_h[0] * 3600
    t1_s = t_range_h[1] * 3600
    n_frames = int((t1_s - t0_s) / dt_sec)
    print(f"[dtecmap] {n_frames} frames  ({t_range_h[0]:.1f}–{t_range_h[1]:.1f} h, "
          f"dt={dt_sec} s, fps={fps})")

    # Static background
    print("[dtecmap] Rendering background...")
    bg, bbox = _make_background(lat_range, lon_range,
                                width_px, height_px, trh, dpi=dpi)
    print(f"  canvas {width_px}×{height_px} px, plot area {bbox}")

    # ffmpeg pipe
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width_px}x{height_px}",
        "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(out_path),
    ]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    import time
    t_start = time.perf_counter()

    for k in range(n_frames):
        t_bin_s = t0_s + k * dt_sec
        t_bin_e = t_bin_s + dt_sec

        mask = (t_sec >= t_bin_s) & (t_sec < t_bin_e)
        lon_k  = lon_v[mask]
        lat_k  = lat_v[mask]
        dtec_k = dtec_v[mask]

        # Filter to domain
        in_domain = (
            (lat_k >= lat_range[0]) & (lat_k <= lat_range[1]) &
            (lon_k >= lon_range[0]) & (lon_k <= lon_range[1])
        )
        lon_k, lat_k, dtec_k = lon_k[in_domain], lat_k[in_domain], dtec_k[in_domain]

        hh  = int(t_bin_s) // 3600
        mm  = (int(t_bin_s) % 3600) // 60
        ss  = int(t_bin_s) % 60
        title = (f"dTEC  {year} DOY {doy:03d}  "
                 f"{hh:02d}:{mm:02d}:{ss:02d} UT  "
                 f"(+{dt_sec} s)  |  "
                 f"{movavg_str}-min avg  |  "
                 f"n={len(dtec_k)}")

        if mode == "scatter":
            frame = _render_frame(bg, bbox, lat_range, lon_range,
                                  lon_k, lat_k, dtec_k, title,
                                  radius_px, trh)
        else:
            frame = _render_frame_voronoi(bg, bbox, lat_range, lon_range,
                                          lon_k, lat_k, dtec_k, title, trh,
                                          downscale=voronoi_downscale,
                                          max_dist_deg=voronoi_max_dist_deg,
                                          draw_borders=voronoi_borders)
        ff.stdin.write(frame.tobytes())

        if (k + 1) % 500 == 0 or (k + 1) == n_frames:
            elapsed = time.perf_counter() - t_start
            fps_now = (k + 1) / elapsed
            eta     = (n_frames - k - 1) / fps_now if fps_now > 0 else 0
            print(f"  frame {k+1}/{n_frames}  "
                  f"({fps_now:.0f} fps render)  "
                  f"ETA {eta/60:.1f} min")

    ff.stdin.close()
    ff.wait()

    elapsed = time.perf_counter() - t_start
    size_mb = out_path.stat().st_size / 1e6
    print(f"[dtecmap] Done: {out_path}")
    print(f"  {n_frames} frames in {elapsed:.1f} s  "
          f"({n_frames/elapsed:.0f} fps render)  "
          f"file {size_mb:.1f} MB")
    print(f"\n  Play: ffplay {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="dTEC map video: one frame per dt from IPP parquets.")
    p.add_argument("--year",      type=int,   required=True)
    p.add_argument("--doy",       type=int,   required=True)
    p.add_argument("--mode",      choices=["scatter", "voronoi"], default="scatter")
    p.add_argument("--dtec-col",  default="dtec_15min")
    p.add_argument("--lat-range", type=float, nargs=2, default=[-43., -10.],
                   metavar=("LAT_MIN", "LAT_MAX"))
    p.add_argument("--lon-range", type=float, nargs=2, default=[-60., -40.],
                   metavar=("LON_MIN", "LON_MAX"))
    p.add_argument("--dt",        type=int,   default=15,
                   help="Seconds per frame (default 15)")
    p.add_argument("--hours",     type=float, nargs=2, default=[0, 24],
                   metavar=("H0", "H1"), help="UTC window to render")
    p.add_argument("--radius",    type=int,   default=9,
                   help="[scatter] Circle radius in pixels (default 9 ≈ 0.5°)")
    p.add_argument("--voronoi-downscale", type=int, default=3,
                   help="[voronoi] Compute at 1/N res, upsample (default 3)")
    p.add_argument("--voronoi-max-dist", type=float, default=None,
                   help="[voronoi] Cap cell reach in degrees (default: uncapped)")
    p.add_argument("--voronoi-no-borders", action="store_true",
                   help="[voronoi] Disable cell-boundary lines")
    p.add_argument("--trh",       type=float, default=0.15,
                   help="±dTEC colour saturation threshold (TECU)")
    p.add_argument("--width",     type=int,   default=900)
    p.add_argument("--height",    type=int,   default=1000)
    p.add_argument("--fps",       type=int,   default=24)
    p.add_argument("--crf",       type=int,   default=18)
    p.add_argument("--out",       default=None, help="Output MP4 path")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  IDE / CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    RUN_MANUALLY = True

    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")
        main(
            year=2021, doy=296,
            dtec_col="dtec_15min",
            mode="voronoi",        # "scatter" ou "voronoi"
            lat_range=(-43., -10.),
            lon_range=(-60., -40.),
            dt_sec=15,
            t_range_h=(0, 24),    # dia inteiro; use (3, 6) para testar rápido
            radius_px=9,          # [scatter] ~0.5° a 900 px / 20°
            voronoi_downscale=3,  # [voronoi] 3 = ~4 min/dia, 2 = mais nítido e mais lento
            voronoi_max_dist_deg=None,  # [voronoi] None = tessellation completa, sem buracos
            voronoi_borders=True,
            trh=0.15,
            width_px=900,
            height_px=1000,
            out_path="/home/hisashi/database/ionosphere/dtecmaps/meu_video.mp4",
            fps=24,
            crf=18,
        )

    else:
        print("Running in CLI Mode...")
        args = _parse()
        main(
            year=args.year, doy=args.doy,
            dtec_col=args.dtec_col,
            mode=args.mode,
            lat_range=tuple(args.lat_range),
            lon_range=tuple(args.lon_range),
            dt_sec=args.dt,
            t_range_h=tuple(args.hours),
            radius_px=args.radius,
            voronoi_downscale=args.voronoi_downscale,
            voronoi_max_dist_deg=args.voronoi_max_dist,
            voronoi_borders=not args.voronoi_no_borders,
            trh=args.trh,
            width_px=args.width,
            height_px=args.height,
            fps=args.fps,
            crf=args.crf,
            out_path=args.out,
            overwrite=args.overwrite,
        )