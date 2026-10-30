# -*- coding: utf-8 -*-
"""
dtec_frames.py -- one dTEC map (PNG or JPEG) per 30-s frame, built for batch runs.

For every chunk of frames (CHUNK_FRAMES at a time):
  1. DuckDB reads the IPP parquets and, in SQL, drops every IPP into its 0.1 x 0.1 deg
     cell and averages the IPPs that share a cell. Only OCCUPIED cells come back:
     one row = (frame, cell row, cell column, mean dTEC).
  2. Worker processes take one frame each:
       - cells without IPP are 0 TECU;
       - each node = mean of the OCCUPIED cells inside a circle (or square) of radius
         RADIUS around it (empty cells are not counted); no occupied cell inside -> 0;
       - the grid is turned into colours and pasted under a ready-made template
         (axes, ticks, colour bar) that matplotlib drew only ONCE;
       - the time label is written on top and the image is saved.
Only the images are saved (IMG_FORMAT: "jpg" or "png").

With SHOW_CPTEC the CPTEC GOES map (IR channel 13, the .npz database of the detector
pipeline) nearest in time is drawn to the LEFT of every dTEC map, same lat/lon box and pixel size.
No detection / tracking: only the temperature map in its own colour scale.

Input : dtec_ipp_dir(year, doy) / "*_dtecIPP_{year}{doy}.parquet"
        columns `horario` (UTC), `lts` (IPP lat), `lns` (IPP lon), DTEC_COL (TECU)
        CPTEC: <database_numpy> / YYYY / MM / *_YYYYMMDDHHMM.npz  (read with shared.detector.read_npz_goes)
Output: OUT_DIR / year / doy / dtec_YYYYmmdd_HHMMSS.jpg or .png   (name = start of the frame, UT)
"""
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve, gaussian_filter

from shared.config import dtec_ipp_dir

# ---------------------------------------------------------------- parameters
DTEC_COL = "dtec_15min"            # dTEC column of the IPP parquets
TIME_COL, LAT_COL, LON_COL = "horario", "lts", "lns"
RES      = 0.1                     # grid spacing (deg)
CADENCE  = 30                      # one map every CADENCE seconds
RADIUS   = 0.5                     # kernel radius (deg); 0.1 + "square" = 3x3 block
SHAPE    = "circle"                # "circle" or "square"
# GAUSS_FWHM_DEG = 1.0
GAUSS_FWHM_DEG = None              # Gaussian smoothing of each grid before the colours: FWHM in deg
                                   # (None or 0 = off). It comes AFTER the kernel mean above, so with
                                   # both on the smoothing adds up (then use a small RADIUS).

VLIM, CMAP = 0.4, "bone_r"           # colour scale: -VLIM .. +VLIM TECU
CELL_PX  = 2                       # pixels per grid cell in the figure (integer) = THE RESOLUTION: it sets the size
                                   # (pixels and MB) of every image. 3 -> 2 gives PNGs ~40 % smaller, 3 -> 1 ~70 %.
                                   # Same look, smaller file: lower FONT_PT in the same proportion (3 / 30 -> 2 / 20)
DPI      = 75                      # only turns points into pixels for the TEXT (1 pt = DPI/72 px) and, with it,
                                   # the margins below. It does NOT change the size of the image: that is CELL_PX
                                   # (100 = the size the margins were drawn for; 50 = text half as large in pixels)
FONT_PT  = 30                      # size of ALL text in the figure (points); 10 = matplotlib default
LEFT, RIGHT, TOP, BOTTOM = 80, 40, 40, 65       # margins around the map (pixels, for 10 pt text at 100 dpi;
                                                # they grow in proportion to FONT_PT and DPI, i.e. to the text)
GAP      = 40                      # space between the dTEC map and the CPTEC map (same rule)
CBAR_W   = 0.4                     # colour bars sit inside the maps, lower-right corner: bar width / map width
CBAR_ALPHA = 0.85                  # opacity of the white box behind each colour bar (1 = hides the data under it)
IMG_FORMAT   = "jpg"               # "jpg": files ~45 % smaller and faster to save, but lossy (a faint halo around text
                                   # and lines, hard to see at normal zoom). "png": exact pixels. Changing it does not
                                   # delete the old files: with OVERWRITE = False the other format's frames are drawn again
JPEG_QUALITY = 70                  # jpg only, 1..95: 75 -> ~40 % smaller than 90 (more halo), 95 -> ~40 % larger

SHOW_CPTEC = True                  # True: CPTEC GOES map to the left of each dTEC map. False: dTEC only
CPTEC_DIR  = None                  # None = automatic (same folders as the detector pipeline).
                                   # Or the folder that contains YYYY/MM/*.npz, e.g. Path(r"D:\goes-data-pipeline\database_numpy")
CPTEC_VMIN, CPTEC_VMAX = -80, 40   # temperature range of the colour scale (deg C)
# Enhanced ("realçada") scale of channel 13 (10.35 um): 10 deg C colour bands between -80 and -30 deg C
# and a light gray -> black ramp from -30 to 40 deg C. Repeated temperatures make the colour "jumps"
# between the bands.
_TB_PONTOS = [
    (-80, (224, 131, 224)), (-70, (255, 209, 255)),  # magenta -> light pink
    (-70, (0, 0, 255)),     (-60, (82, 69, 255)),    # blue
    (-60, (47, 218, 255)),  (-50, (156, 227, 245)),  # cyan
    (-50, (255, 255, 97)),  (-40, (235, 255, 122)),  # yellow
    (-40, (255, 101, 1)),   (-30, (255, 158, 70)),   # orange
    (-30, (176, 176, 176)), (40, (0, 0, 0)),         # light gray -> black
]
CPTEC_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tb_realcada",
    [((t - CPTEC_VMIN) / (CPTEC_VMAX - CPTEC_VMIN), [v / 255 for v in c]) for t, c in _TB_PONTOS],
    N=1024,
)                                  # (any matplotlib colormap name, e.g. "gray_r", also works here)
CPTEC_NODATA = (210, 225, 245)     # colour of pixels without a GOES value (RGB)
CPTEC_MAX_GAP_MIN = 20             # frame with no GOES map within this many minutes -> empty panel

MAP_LINES = True                   # coastline, country borders, states (cartopy / Natural Earth) on both maps.
                                   # Drawn ONCE, in the template, on top of the data: no cost per frame.
                                   # False: no lines (cartopy is then not needed)
MAP_COLOR = "lime"                 # colour of the lines (any matplotlib colour). Softer ones: "#ffb7c5" pink,
                                   # "#a8e6a3" green, "#9fe8e0" mint
MAP_LW    = 0.6                    # coastline width (points, for 10 pt text; grows with FONT_PT)
MAP_FEATURES = (                   # (Natural Earth category, name, scale, width relative to MAP_LW)
    ("physical", "coastline",                      "50m", 1.0),    # scale: "110m", "50m" or "10m"
    ("cultural", "admin_0_boundary_lines_land",    "50m", 0.8),    # (the states of most countries
    ("cultural", "admin_1_states_provinces_lines", "50m", 0.5),    #  only exist at "10m"). Delete a line
)                                  # to drop that layer. The data are downloaded once by cartopy.

CHUNK_FRAMES = 480                 # frames queried at once (480 = 4 h)
WORKERS   = max(1, (os.cpu_count() or 2) - 10)
OVERWRITE = False                  # False: frames whose image file exists are skipped
IN_DIR    = None                   # None = automatic: dtec_ipp_dir(year, doy). Or a folder that
                                   # directly contains the parquets of the day (override)
OUT_DIR   = Path("dtec_frames")    # ROOT of the figures (relative paths start where you launch
                                   # the script, so prefer an absolute one), e.g.
                                   # Path(r"D:\database\ionosphere\dtecframes")


# ---------------------------------------------------------------- 1. reading (DuckDB)
def query_cells(files, t0, t1, lat_range, lon_range):
    """One row per occupied cell: (frame, iy, ix, m).
    frame = number of the CADENCE-second frame after t0, iy / ix = row / column of the
    0.1-deg cell (the cell is the square around the node; round half up), m = mean dTEC
    of the IPPs that fell in it. Only t0 <= t < t1 and the lat/lon box are used."""
    import duckdb                                        # here, so the workers do not load it

    file_list = "[" + ", ".join("'" + Path(f).as_posix().replace("'", "''") + "'"
                                for f in files) + "]"
    t0_epoch = (t0 - pd.Timestamp("1970-01-01")).total_seconds()
    sql = f"""
        WITH src AS (
            SELECT CAST({TIME_COL} AS TIMESTAMP) AS t,
                   CAST({LAT_COL}  AS DOUBLE)    AS lat,
                   CAST({LON_COL}  AS DOUBLE)    AS lon,
                   CAST({DTEC_COL} AS DOUBLE)    AS dtec
            FROM read_parquet({file_list})
        )
        SELECT CAST(floor((epoch(t) - {t0_epoch}) / {CADENCE}) AS INTEGER)    AS frame,
               CAST(floor((lat - ({lat_range[0]})) / {RES} + 0.5) AS INTEGER) AS iy,
               CAST(floor((lon - ({lon_range[0]})) / {RES} + 0.5) AS INTEGER) AS ix,
               avg(dtec)                                                      AS m
        FROM src
        WHERE t >= TIMESTAMP '{t0}' AND t < TIMESTAMP '{t1}'
          AND lat BETWEEN {lat_range[0]} AND {lat_range[1]}
          AND lon BETWEEN {lon_range[0]} AND {lon_range[1]}
          AND isfinite(dtec)
        GROUP BY 1, 2, 3
    """
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")                    # naive timestamps = UTC
    out = con.execute(sql).fetchnumpy()
    con.close()
    return (np.asarray(out["frame"]), np.asarray(out["iy"]),
            np.asarray(out["ix"]), np.asarray(out["m"]))


# ---------------------------------------------------------------- 2. gridding
def make_kernel(radius, res, shape):
    """2-D footprint of 0/1 with the centre node included.
    "circle": nodes whose distance to the centre is <= radius.
    "square": nodes with |dlat| <= radius and |dlon| <= radius.
    The 1e-9 only protects against 0.3/0.1 = 2.9999999999999996."""
    r = radius / res                                     # radius in grid steps
    n = int(np.floor(r + 1e-9))
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
    if shape == "circle":
        inside = dy ** 2 + dx ** 2 <= r ** 2 + 1e-9
    elif shape == "square":
        inside = np.ones(dy.shape, dtype=bool)
    else:
        raise ValueError(f"SHAPE must be 'circle' or 'square', got {shape!r}")
    return inside.astype(float)


def grid_frame(iy, ix, m, shape, kernel, sigma_px=0.0):
    """One frame. iy, ix = row / column of each OCCUPIED cell, m = mean dTEC of its IPPs,
    shape = (n_lat, n_lon), sigma_px = Gaussian sigma in grid steps (0 = no smoothing).
    Returns the (n_lat, n_lon) grid."""
    # cells without IPP stay 0
    occupied = np.zeros(shape, dtype=bool)
    cell_mean = np.zeros(shape)
    occupied[iy, ix] = True
    cell_mean[iy, ix] = m

    # each node: add up the cell means inside the kernel and divide by how many
    # OCCUPIED cells are inside the kernel
    total = convolve(cell_mean, kernel, mode="constant", cval=0.0)
    n_occupied = convolve(occupied.astype(float), kernel, mode="constant", cval=0.0)
    grid = np.zeros(shape)
    has_data = n_occupied > 0
    grid[has_data] = total[has_data] / n_occupied[has_data]

    # optional Gaussian smoothing, before the colours. Same idea as the kernel mean: only nodes
    # WITH data are averaged (the zeros of empty nodes do not enter) and empty nodes stay 0
    if sigma_px > 0:
        num = gaussian_filter(grid, sigma_px, mode="constant", cval=0.0)       # grid is 0 where no data
        den = gaussian_filter(has_data.astype(float), sigma_px, mode="constant", cval=0.0)
        grid = np.where(has_data, num / np.maximum(den, 1e-12), 0.0)
    return grid


# ---------------------------------------------------------------- 2b. CPTEC GOES maps
def cptec_root():
    """Folder of the GOES .npz database, as in the detector pipeline:
    <root>/YYYY/MM/S10635346_YYYYMMDDHHMM.npz"""
    if os.name == "nt":
        return Path("D:/goes-data-pipeline/database_numpy")
    return Path("~/database/cptec/database_numpy").expanduser()


def list_cptec(root, t0, t1):
    """GOES files whose time (read from the file name) lies within CPTEC_MAX_GAP_MIN of [t0, t1].
    Returns (times, paths), sorted by time (times = DatetimeIndex, UTC without tz)."""
    gap = pd.Timedelta(minutes=CPTEC_MAX_GAP_MIN)
    found = []
    for month in pd.period_range(t0 - gap, t1 + gap, freq="M"):
        for f in (Path(root) / f"{month.year}" / f"{month.month:02d}").glob("*.npz"):
            hit = re.search(r"_(\d{12})", f.name)          # YYYYMMDDHHMM, as in the detector pipeline
            if hit:
                t = pd.to_datetime(hit.group(1), format="%Y%m%d%H%M")
                if t0 - gap <= t <= t1 + gap:
                    found.append((t, f))
    found.sort(key=lambda x: x[0])
    return pd.DatetimeIndex([t for t, _ in found]), [str(f) for _, f in found]


def match_cptec(times, paths, frame_times):
    """For every frame: (path, time) of the GOES map nearest in time (the earlier one if both
    neighbours are equally near), or (None, None) if it is farther than CPTEC_MAX_GAP_MIN."""
    t = times.values.astype("datetime64[s]").astype("int64")           # seconds
    f = frame_times.values.astype("datetime64[s]").astype("int64")
    j = np.searchsorted(t, f)                                          # first map at or after the frame
    lo, hi = np.clip(j - 1, 0, t.size - 1), np.clip(j, 0, t.size - 1)
    best = np.where(np.abs(f - t[lo]) <= np.abs(t[hi] - f), lo, hi)
    near = np.abs(t[best] - f) <= CPTEC_MAX_GAP_MIN * 60
    return [(paths[b], times[b]) if ok else (None, None) for b, ok in zip(best, near)]


def read_npz_goes(file_path):
    """
    Load a saved GOES .npz and return (temp, lats, lons) exactly as stored.

    Expected keys in the archive: 'temp', 'lats', 'lons'.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(fp)

    with np.load(fp, allow_pickle=False) as npz:
        try:
            temp = npz["temp"]/100.
            lats = npz["lats"]
            lons = npz["lons"]
        except KeyError as e:
            raise KeyError(f"Missing key in {fp.name}: {e}")

    return temp, lats, lons

def read_cptec(path):
    """One GOES map through your own reader. Returns temp (n_lat, n_lon) float32 in deg C
    with NaN where there is no value, lat (n_lat,), lon (n_lon,) -- regular 1-D grids."""
    # from shared.detector import read_npz_goes             # here: only the processes that need it
    temp, lat, lon = read_npz_goes(Path(path))
    temp = np.ma.filled(np.ma.asarray(temp, dtype=np.float32), np.nan)
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if lat.ndim == 2: lat = lat[:, 0]                      # (in case they come as 2-D meshes)
    if lon.ndim == 2: lon = lon[0, :]
    if temp.shape != (lat.size, lon.size):
        raise ValueError(f"{path}: temp {temp.shape} does not match lat {lat.size} x lon {lon.size}")
    for name, a in (("lat", lat), ("lon", lon)):
        step = (a[-1] - a[0]) / (a.size - 1)
        if not np.allclose(np.diff(a), step, atol=0.05 * abs(step)):
            raise ValueError(f"{path}: {name} is not a regular grid")
    return temp, lat, lon


def _nearest_index(src, first_edge, step, n):
    """For n output pixels (the k-th one is centred at first_edge + (k + 0.5) * step) the index of
    the nearest node of the regular grid `src` (either order). A pixel up to one grid step beyond
    the last node takes that node (the edge of the box is not a thin empty line); farther out: -1."""
    d = (src[-1] - src[0]) / (src.size - 1)                # grid step, with its sign
    i = np.floor((first_edge + (np.arange(n) + 0.5) * step - src[0]) / d + 0.5).astype(int)
    outside = (i < -1) | (i > src.size)
    i = np.clip(i, 0, src.size - 1)
    i[outside] = -1
    return i


def cptec_panel(path, lut, lat_top, lon_left, h, w):
    """Colour image (h, w, 3) uint8 of one GOES map on the pixels of the dTEC map: row 0 = north,
    the pixel k of a row is centred at lon_left + (k + 0.5) * RES / CELL_PX, and the same in lat.
    Nearest GOES node per pixel (temperatures are not interpolated); `lut` has n + 1 colours:
    n levels between CPTEC_VMIN and CPTEC_VMAX (colder / warmer: the end colours) + the no-data colour."""
    n = len(lut) - 1                                       # number of colour levels; index n = no data
    temp, lat, lon = read_cptec(path)
    level = np.full(temp.shape, n, dtype=np.uint16)
    ok = np.isfinite(temp)
    level[ok] = np.clip((temp[ok] - CPTEC_VMIN) / (CPTEC_VMAX - CPTEC_VMIN) * n, 0, n - 1)
    rows = _nearest_index(lat, lat_top, -RES / CELL_PX, h)
    cols = _nearest_index(lon, lon_left, RES / CELL_PX, w)
    out = level[np.ix_(np.maximum(rows, 0), np.maximum(cols, 0))]
    out[rows < 0, :] = n                                   # pixels outside the GOES grid
    out[:, cols < 0] = n
    return lut[out]


# ---------------------------------------------------------------- 2c. coastlines, borders, states
def _line_arrays(geom):
    """The (n, 2) lon/lat arrays of a (multi) line geometry; a polygon gives its outlines."""
    kind = geom.geom_type
    if kind in ("LineString", "LinearRing"):
        return [np.asarray(geom.coords)[:, :2]]
    if kind in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        return [a for g in geom.geoms for a in _line_arrays(g)]
    if kind == "Polygon":
        return [np.asarray(r.coords)[:, :2] for r in (geom.exterior, *geom.interiors)]
    return []


def map_lines(lat_range, lon_range):
    """The Natural Earth lines of MAP_FEATURES that touch the lat/lon box (1 deg margin).
    Returns [(name, [(n, 2) arrays of lon/lat], relative width), ...].
    cartopy only supplies the data (download on first use, then cached); the drawing is plain matplotlib."""
    try:
        import cartopy.feature as cfeature
        x0, x1 = lon_range[0] - 1, lon_range[1] + 1
        y0, y1 = lat_range[0] - 1, lat_range[1] + 1
        out = []
        for category, name, scale, width in MAP_FEATURES:
            arrays = []
            for geom in cfeature.NaturalEarthFeature(category, name, scale).geometries():
                gx0, gy0, gx1, gy1 = geom.bounds
                if gx1 >= x0 and gx0 <= x1 and gy1 >= y0 and gy0 <= y1:       # touches the box
                    arrays += _line_arrays(geom)
            out.append((name, arrays, width))
        return out
    except Exception as e:
        raise RuntimeError(f"Could not get the map lines ({type(e).__name__}: {e}). cartopy installed? Internet "
                           f"on the first run (it downloads Natural Earth)? Or set MAP_LINES = False.") from e


# ---------------------------------------------------------------- 3. figure
def make_template(lats, lons):
    """Everything that is the same in every figure -- axes, ticks, grid lines, colour bars --
    drawn ONCE with matplotlib, with exact pixel positions: each map occupies
    (nx * CELL_PX) x (ny * CELL_PX) pixels, one grid cell = CELL_PX x CELL_PX pixels.
    With SHOW_CPTEC the CPTEC map goes on the LEFT (it carries the latitude numbers) and the dTEC map
    on the right: same size, same lat/lon box.
    Returns a transparent RGBA image of the whole figure + where the maps go."""
    s = FONT_PT / 10                                     # line widths (points) are defined for 10 pt text
    k = s * DPI / 100                                    # margins (pixels) are defined for 10 pt text at 100 dpi
    left, right, top, bottom, gap = (round(v * k) for v in (LEFT, RIGHT, TOP, BOTTOM, GAP))
    ny, nx = lats.size, lons.size
    w, h = nx * CELL_PX, ny * CELL_PX                    # map size (pixels)
    x_cptec = left                                       # left edge of the CPTEC map (when shown)
    x_dtec = left + w + gap if SHOW_CPTEC else left      # left edge of the dTEC map
    W, H = x_dtec + w + right, top + h + bottom          # figure size (pixels)

    lines = map_lines((lats[0], lats[-1]), (lons[0], lons[-1])) if MAP_LINES else []
    # Every text gets FONT_PT explicitly. Do NOT rely on matplotlib's "font.size": a matplotlibrc,
    # a style or an imported module that sets axes.labelsize / xtick.labelsize ... ignores it.
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

    bars = []                                            # (colour bar axes, left edge of its map)

    def add_panel(x, show_lat, vmin, vmax, cmap, cbar_label, ticks):
        """One map: axes (ticks, grid lines) with its left edge at pixel x, and its colour bar
        (placed later, in the lower-right corner of the map)."""
        ax = fig.add_axes([x / W, bottom / H, w / W, h / H])
        for _, arrays, width in lines:                   # coastline, borders, states (on top of the data)
            ax.add_collection(LineCollection(arrays, colors=MAP_COLOR, linewidths=MAP_LW * s * width))
        ax.set_xlim(lons[0] - RES / 2, lons[-1] + RES / 2)   # cell edges, not node centres (fixes the box)
        ax.set_ylim(lats[0] - RES / 2, lats[-1] + RES / 2)
        ax.set_xlabel("Longitude, deg", fontsize=FONT_PT)
        if show_lat:                                     # only the left-most map has latitude name + numbers
            ax.set_ylabel("Latitude, deg", fontsize=FONT_PT)
        ax.tick_params(labelsize=FONT_PT, labelleft=show_lat)   # numbers of the axes
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.patch.set_alpha(0)                            # transparent: the map goes underneath
        cax = fig.add_axes([x / W, bottom / H, CBAR_W * w / W, 16 * k / H])   # (provisional position)
        sm = mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), cmap)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
        cbar.set_label(cbar_label, fontsize=FONT_PT)
        cbar.ax.tick_params(labelsize=FONT_PT)           # numbers of the colour bar
        bars.append((cax, x))

    if SHOW_CPTEC:
        add_panel(x_cptec, True, CPTEC_VMIN, CPTEC_VMAX, CPTEC_CMAP, "Tb GOES Ch13 (°C)",
                  np.arange(CPTEC_VMIN, CPTEC_VMAX + 1, 20))
    add_panel(x_dtec, not SHOW_CPTEC, -VLIM, VLIM, CMAP, "dTEC (TECU)", np.linspace(-VLIM, VLIM, 5))

    # Colour bars: lower-right corner INSIDE each map, on a white box so that the numbers can be read
    # over the data. The box is as large as bar + numbers + name (measured), so it fits any FONT_PT.
    fig.canvas.draw()                                    # (the sizes of the texts are needed)
    pad = round(0.4 * FONT_PT * DPI / 72)                # box <-> text, and box <-> edge of the map
    for cax, x in bars:
        bb = cax.get_tightbbox(fig.canvas.get_renderer())      # bar + numbers + name, in figure pixels
        dx, dy = (x + w - 2 * pad) - bb.x1, (bottom + 2 * pad) - bb.y0   # move its lower-right corner there
        p = cax.get_position()
        cax.set_position([p.x0 + dx / W, p.y0 + dy / H, p.width, p.height])
        cax.set_zorder(2)                                # above the box, above the map lines
        fig.add_artist(mpl.patches.Rectangle((bb.x0 + dx - pad, bb.y0 + dy - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                                             transform=mpl.transforms.IdentityTransform(), facecolor="white",
                                             alpha=CBAR_ALPHA, edgecolor="0.4", linewidth=0.6 * s, zorder=1))

    fig.patch.set_alpha(0)                               # transparent: the maps go underneath
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba()).copy()  # (H, W, 4)
    plt.close(fig)

    font_px = round(1.2 * FONT_PT * DPI / 72)            # labels above the maps: bold, 1.2 x the text size
    bold = str(font_manager.findfont(font_manager.FontProperties(weight="bold")))  # str: must be picklable
    return dict(image=image, box=(x_dtec, top), title_xy=(x_dtec + w / 2, top / 2),
                box_cptec=(x_cptec, top) if SHOW_CPTEC else None, title_cptec_xy=(x_cptec + w / 2, top / 2),
                geometry=(lats[-1] + RES / 2, lons[0] - RES / 2, h, w),   # lat of the top edge, lon of the left edge, size
                map_info=", ".join(f"{name} {len(arrays)}" for name, arrays, _ in lines),   # lines found in the box
                font_path=bold, font_px=font_px)


_W = {}                                                  # per-worker copy of the shared data


def _init_worker(kernel, shape, template, sigma_px=0.0):
    cmap = CPTEC_CMAP if isinstance(CPTEC_CMAP, mpl.colors.Colormap) else mpl.colormaps[CPTEC_CMAP]
    cptec_lut = np.vstack([cmap(np.linspace(0, 1, 1024), bytes=True)[:, :3],
                           np.array(CPTEC_NODATA, dtype=np.uint8)])         # 1024 levels (0.12 deg C each) + no data
    _W.update(kernel=kernel, shape=shape, sigma_px=sigma_px, box=template["box"],
              title_xy=template["title_xy"], template=Image.fromarray(template["image"]),
              lut=mpl.colormaps[CMAP](np.arange(256), bytes=True)[:, :3],
              font=ImageFont.truetype(template["font_path"], template["font_px"]),
              box_cptec=template["box_cptec"], title_cptec_xy=template["title_cptec_xy"], geometry=template["geometry"],
              cptec_lut=cptec_lut, cptec_file=None, cptec_image=None)


def cptec_image(gfile):
    """PIL image of the GOES map `gfile` for the CPTEC panel. The last one is kept: the 30-s frames
    of a worker are consecutive and share the same GOES map (one every ~10 min), so it is read once."""
    if _W["cptec_file"] != gfile:
        _W["cptec_image"] = Image.fromarray(cptec_panel(gfile, _W["cptec_lut"], *_W["geometry"]))
        _W["cptec_file"] = gfile
    return _W["cptec_image"]


def render_frame(task):
    """grid -> colours -> pasted under the template -> labels -> image file (JPEG if the path ends in .jpg, else PNG).
    task = (image path, label, iy, ix, m, (GOES file, GOES time) or (None, None))."""
    path, label, iy, ix, m, (gfile, gtime) = task
    grid = grid_frame(iy, ix, m, _W["shape"], _W["kernel"], _W["sigma_px"])

    # same value -> colour mapping as matplotlib (256 levels between -VLIM and +VLIM)
    level = np.clip(((grid + VLIM) / (2 * VLIM) * 256).astype(int), 0, 255)
    rgb = _W["lut"][level][::-1]                         # row 0 is south: flip, north on top
    rgb = np.repeat(np.repeat(rgb, CELL_PX, axis=0), CELL_PX, axis=1)

    canvas = Image.new("RGBA", _W["template"].size, "white")
    canvas.paste(Image.fromarray(np.ascontiguousarray(rgb)), _W["box"])
    label2 = None
    if _W["box_cptec"] is not None:                      # CPTEC map, to the left of the dTEC map
        if gfile is None:                                # nothing near enough in time: empty panel
            h, w = _W["geometry"][2:]
            canvas.paste(Image.new("RGB", (w, h), CPTEC_NODATA), _W["box_cptec"])
            label2 = f"GOES: no map within {CPTEC_MAX_GAP_MIN} min"
        else:
            canvas.paste(cptec_image(gfile), _W["box_cptec"])
            label2 = f"GOES Ch13  {gtime:%H:%M} UT"
    canvas = Image.alpha_composite(canvas, _W["template"])   # axes, ticks, colour bars on top
    draw = ImageDraw.Draw(canvas)
    draw.text(_W["title_xy"], label, fill="black", font=_W["font"], anchor="mm")
    if label2:
        draw.text(_W["title_cptec_xy"], label2, fill="black", font=_W["font"], anchor="mm")
    image = canvas.convert("RGB")
    if str(path).lower().endswith(".jpg"):
        image.save(path, quality=JPEG_QUALITY, subsampling=0)     # subsampling=0: no colour blur on thin coloured lines
    else:
        image.save(path, compress_level=1)


# ---------------------------------------------------------------- run
def out_folder(root, year, doy):
    """Folder that receives the figures of one day: <root>/<year>/<doy>.
    The folder layout is defined here and only here."""
    return Path(root).expanduser() / str(year) / f"{doy:03d}"


def main(year, doy, lat_range, lon_range, ut_range, in_dir=None, out_dir=None, cptec_dir=None):
    """in_dir / out_dir / cptec_dir override IN_DIR / OUT_DIR / CPTEC_DIR for this call
    (None = use the constants)."""
    t_start = time.perf_counter()
    if IMG_FORMAT not in ("jpg", "png"):
        raise ValueError(f"IMG_FORMAT must be 'jpg' or 'png', got {IMG_FORMAT!r}")
    in_dir = in_dir or IN_DIR
    folder = Path(in_dir).expanduser() if in_dir else Path(dtec_ipp_dir(year, doy))
    files = sorted(set(folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
                   | set(folder.glob(f"*_dtecIPP_{year}{doy:03d}.parquet")))
    if not files:
        raise FileNotFoundError(f"No IPP files in {folder}")

    day = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    t0 = day + pd.Timedelta(hours=ut_range[0])
    t1 = day + pd.Timedelta(hours=ut_range[1])
    n_frames = int(round((t1 - t0).total_seconds() / CADENCE))

    lats = lat_range[0] + RES * np.arange(int(round((lat_range[1] - lat_range[0]) / RES)) + 1)
    lons = lon_range[0] + RES * np.arange(int(round((lon_range[1] - lon_range[0]) / RES)) + 1)
    shape = (lats.size, lons.size)
    kernel = make_kernel(RADIUS, RES, SHAPE)
    sigma_px = GAUSS_FWHM_DEG / (2 * np.sqrt(2 * np.log(2))) / RES if GAUSS_FWHM_DEG else 0.0
    template = make_template(lats, lons)
    out_dir = out_folder(out_dir or OUT_DIR, year, doy)
    out_dir.mkdir(parents=True, exist_ok=True)

    goes = [(None, None)] * n_frames                     # per frame: (GOES file, its time)
    if SHOW_CPTEC:
        root = Path(cptec_dir or CPTEC_DIR).expanduser() if (cptec_dir or CPTEC_DIR) else cptec_root()
        times, paths = list_cptec(root, t0, t1)
        if not paths:
            raise FileNotFoundError(f"No CPTEC maps (YYYY/MM/*_YYYYMMDDHHMM.npz) in {root} within "
                                    f"{CPTEC_MAX_GAP_MIN} min of {t0} .. {t1}. "
                                    f"Check CPTEC_DIR, or set SHOW_CPTEC = False for dTEC only.")
        goes = match_cptec(times, paths, t0 + pd.to_timedelta(np.arange(n_frames) * CADENCE, unit="s"))
        temp, glat, glon = read_cptec(paths[0])          # one read here: fail early, show the grid
        lat_top, lon_left, h, w = template["geometry"]                       # any map pixel without a GOES node?
        outside = ((_nearest_index(glat, lat_top, -RES / CELL_PX, h) < 0).any() or
                   (_nearest_index(glon, lon_left, RES / CELL_PX, w) < 0).any())

    fmt = IMG_FORMAT.upper() + (f" quality {JPEG_QUALITY}" if IMG_FORMAT == "jpg" else "")
    print(f"input : {folder}\noutput: {out_dir.resolve()}  ({fmt})")
    print(f"{len(files)} files | {n_frames} frames of {CADENCE} s | grid {shape[0]} x {shape[1]} "
          f"@ {RES} deg | kernel {SHAPE} {RADIUS} deg ({int(kernel.sum())} nodes) | "
          f"{WORKERS} workers")
    if MAP_LINES:
        print(f"map lines in the box ({MAP_COLOR}): {template['map_info']}")
    if SHOW_CPTEC:
        print(f"CPTEC : {root} | {len(paths)} maps {times[0]:%H:%M}..{times[-1]:%H:%M} UT | "
              f"grid {temp.shape[0]} x {temp.shape[1]}, lat {glat.min():.2f}..{glat.max():.2f}, "
              f"lon {glon.min():.2f}..{glon.max():.2f} | {sum(g[0] is None for g in goes)} frames "
              f"without a map within {CPTEC_MAX_GAP_MIN} min | colours {CPTEC_VMIN}..{CPTEC_VMAX} deg C")
        if outside:
            print("WARNING: the requested lat/lon box is larger than the GOES grid: "
                  "the part outside is drawn in the no-data colour")
    if sigma_px > 0:
        sigma_km = sigma_px * RES * 111.2                # amplitude response to a 180 km wave
        print(f"Gaussian smoothing: FWHM {GAUSS_FWHM_DEG} deg = sigma {sigma_px:.2f} grid steps | "
              f"amplitude response at 180 km: {np.exp(-2 * (np.pi * sigma_km / 180) ** 2):.2f}")
    done = skipped = 0
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker,
                             initargs=(kernel, shape, template, sigma_px)) as pool:
        for c0 in range(0, n_frames, CHUNK_FRAMES):
            c1 = min(c0 + CHUNK_FRAMES, n_frames)        # this chunk = frames c0 .. c1-1
            ta = t0 + pd.Timedelta(seconds=c0 * CADENCE)
            tb = t0 + pd.Timedelta(seconds=c1 * CADENCE)
            frame, iy, ix, m = query_cells(files, ta, tb, lat_range, lon_range)

            order = np.argsort(frame, kind="stable")     # one contiguous slice per frame
            frame, iy, ix, m = frame[order], iy[order], ix[order], m[order]
            edges = np.searchsorted(frame, np.arange(c1 - c0 + 1))

            tasks = []
            for k in range(c1 - c0):
                ts = ta + pd.Timedelta(seconds=k * CADENCE)
                path = out_dir / f"dtec_{ts:%Y%m%d_%H%M%S}.{IMG_FORMAT}"
                if path.exists() and not OVERWRITE:
                    skipped += 1
                    continue
                i0, i1 = edges[k], edges[k + 1]
                label = f"{ts:%m/%d/%Y}  UT = {ts:%H:%M:%S}"
                tasks.append((str(path), label, iy[i0:i1], ix[i0:i1], m[i0:i1], goes[c0 + k]))
            list(pool.map(render_frame, tasks, chunksize=8))   # list(): re-raises worker errors
            done += len(tasks)
            print(f"frames {c0}-{c1 - 1}: {len(tasks)} drawn | "
                  f"{m.size / max(c1 - c0, 1):.0f} occupied cells per frame")

    seconds = time.perf_counter() - t_start
    print(f"{done} figures drawn, {skipped} skipped (already exist) in {seconds:.1f} s "
          f"-> {out_dir}")
    return out_dir


if __name__ == "__main__":                              # required for multiprocessing on Windows
    main(
        # year=2021, doy=296,
        year=2023, doy=347,
        # year=2026, doy=203,
        # year=2026, doy=204,
         lat_range=(-55.0, 10.0),
         lon_range=(-82.0, -34.0),
         ut_range=(0, 24),                              # UT hours: window [20h, 23h)
         out_dir=r"D:\database\ionosphere\dtecframes",  # optional: overrides OUT_DIR
         # in_dir=None,                                   # optional: overrides IN_DIR
         # cptec_dir=None,                                # optional: overrides CPTEC_DIR
         )