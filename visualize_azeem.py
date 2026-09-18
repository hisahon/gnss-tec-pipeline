# -*- coding: utf-8 -*-
"""
dTEC maps and wavelet spectrograms following Azeem, Walterscheid & Crowley
(2018), GRL, 45, 8014-8021, doi:10.1029/2018GL078107  (Section 2 + Section 4).

Processing chain of the paper (Section 2, "GPS TEC Data")
---------------------------------------------------------
  (1) slant TEC -> vertical TEC                          [upstream pipeline]
  (2) 10-min running mean detrend, per receiver-satellite pair
                                                        [upstream: `dtec_10min`]
  (3) nearest-neighbour interpolation of the scattered IPPs onto a
      0.1 deg x 0.1 deg lat/lon grid, one map every 30 s  [grid_nearest]
  (4) 2-D Gaussian smoothing                             [gaussian_nan]
  (5) adaptive complex diffusion despeckling filter      [despeckle_acd]
      -> ON/OFF switch (`despeckle=True/False`)
Analysis (Section 4)
  (6) dTEC maps, bone colour scale, +-0.2 TECU, red cross on the storm (Fig. 3)
  (7) normalized wavelet spectrograms, period x longitude at a fixed latitude
      and period x latitude at a fixed longitude, at a given UT (Fig. 4a/4b)
  (8) wavelet spectrogram period x UT at a point near the storm centre
      (described in Section 5 of the paper, "not shown here")

The numerical model of Section 3 / Fig. 5 (Walterscheid's 2-D nonlinear
cylindrical model) is NOT part of this script.

Reads the same per-station IPP parquets as `visualize_keo_datashader.py`:
    dtec_ipp_dir(year, doy) / "*_dtecIPP_{year}{doy}.parquet"
    index  : `horario` (datetime, UTC)
    columns: `lts` (IPP lat), `lns` (IPP lon), `dtec_10min` (TECU)

NOTES ON WHAT THE PAPER DOES NOT FULLY SPECIFY  (all exposed as parameters)
---------------------------------------------------------------------------
* Gaussian FWHM. The 2018 paper says "3 deg". Vadas & Azeem (2021, JGR,
  doi:10.1029/2020JA028275) describe the SAME scheme as "0.1 x 0.1 binning
  ... 2D Gaussian filter with a full width at half maximum of 0.3 deg", i.e.
  3 grid pixels. A literal 3 deg FWHM attenuates a 180-km wave by ~1e-5 and
  would erase the very rings shown in Fig. 3, so the default here is
  FWHM = 0.3 deg (3 px). Set GAUSS_FWHM_DEG = 3.0 for the literal reading.
* Despeckle filter. "Adaptive complex diffusion despeckling filter" is the
  filter of Bernardes et al. (2010, Opt. Express 18, 24048), built on the
  nonlinear complex diffusion of Gilboa et al. (2004, IEEE TPAMI 26, 1020).
  It is implemented here from the published equations with the published
  constants (theta = pi/30, kmin = 2, kmax = 28, alpha = 4, a = .25, b = .75).
  The total diffusion time T and the TECU -> 0..255 grey-level mapping are
  not given by Azeem; see `despeckle_acd`.
* Maximum nearest-neighbour distance. Fig. 3 has white (no-data) holes, so
  the NN interpolation was range limited, but the radius is not given.
* Wavelet. Mother wavelet not given; Morlet (omega0 = 6) is used
  (Torrence & Compo 1998), power normalized to its maximum.

@author: hisas
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

try:                       # NetCDF output, same stack as the keogram routine
    import xarray as xr
except ImportError:        # falls back to .npz
    xr = None

try:                       # optional coastlines / borders
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    ccrs = cfeature = None

# Import custom modules (same as visualize_keo_datashader.py)
try:
    from shared.config import (dtec_ipp_dir, dtec_keogram_dir,
                               dtec_keogram_figs_dir, storms_SouthAmerica)
except ImportError:        # core functions remain usable stand-alone
    dtec_ipp_dir = dtec_keogram_dir = None
    dtec_keogram_figs_dir = storms_SouthAmerica = None


# ==========================================
# 0. Parameters of Azeem et al. (2018)
# ==========================================
DTEC_COL         = "dtec_15min"  # paper (S2) uses 10-min; your IPP parquets
                                 # only carry dtec_15min (see note below and
                                 # in _resolve_dtec_col). Auto-detected/warned
                                 # anyway if this doesn't match what's on disk.
GRID_RES_DEG     = 0.1           # 0.1 x 0.1 deg grid               (paper, S2)
CADENCE_S        = 30            # one map every 30 s               (paper, S2)
GAUSS_FWHM_DEG   = 0.3           # 3 px; paper text says "3 deg" -> see header
NN_MAX_DIST_DEG  = 1.0           # not given in the paper (white holes, Fig. 3)
DESPECKLE        = True          # ON/OFF switch of step (5)
DESPECKLE_T      = 2.0           # total diffusion time, not given in the paper
DESPECKLE_SCALE  = 0.2           # +-TECU mapped to grey levels 0..255
MAP_VLIM_TECU    = 0.2           # colour scale of Fig. 3
MAP_CMAP         = "bone"        # colour map of Fig. 3
WAVELET_PERIODS  = (1.0, 10.0)   # min; paper: 1 < tau < 10 min
WAVELET_OMEGA0   = 6.0           # Morlet

TIME_COL, LAT_COL, LON_COL = "horario", "lts", "lns"


# ==========================================
# 1. Core: nearest-neighbour gridding
# ==========================================
def build_grid_axes(lat_range, lon_range, res=GRID_RES_DEG):
    """Grid node coordinates (1-D lat and lon vectors) at `res` degrees."""
    n_lat = int(round((lat_range[1] - lat_range[0]) / res)) + 1
    n_lon = int(round((lon_range[1] - lon_range[0]) / res)) + 1
    lats = lat_range[0] + res * np.arange(n_lat)
    lons = lon_range[0] + res * np.arange(n_lon)
    return lats, lons


def grid_nearest(lon_p, lat_p, val_p, grid_nodes, shape, max_dist_deg):
    """Nearest-neighbour interpolation of scattered IPPs onto the grid.

    Same as MATLAB griddata(..., 'nearest') / scipy NearestNDInterpolator,
    but nodes farther than `max_dist_deg` from any IPP are left as NaN
    (the white holes of Fig. 3). max_dist_deg=None -> unlimited.
    """
    out = np.full(grid_nodes.shape[0], np.nan, dtype=np.float32)
    tree = cKDTree(np.column_stack([lon_p, lat_p]))
    ub = np.inf if max_dist_deg is None else float(max_dist_deg)
    dist, idx = tree.query(grid_nodes, k=1, distance_upper_bound=ub, workers=-1)
    ok = np.isfinite(dist)
    out[ok] = val_p[idx[ok]]
    return out.reshape(shape)


# ==========================================
# 2. Core: 2-D Gaussian filter (NaN aware)
# ==========================================
def fwhm_to_sigma_px(fwhm_deg, res=GRID_RES_DEG):
    return fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))) / res


def gaussian_attenuation(wavelength_km, fwhm_deg, lat_deg=0.0, along="lat"):
    """Theoretical amplitude response of the Gaussian filter to a plane wave
    of the given horizontal wavelength. Used only for the log message."""
    km_per_deg = 111.2 * (np.cos(np.deg2rad(lat_deg)) if along == "lon" else 1.0)
    sigma_km = fwhm_deg / 2.3548 * km_per_deg
    return float(np.exp(-2.0 * (np.pi * sigma_km / wavelength_km) ** 2))


def gaussian_nan(img, sigma_px):
    """2-D Gaussian smoothing as a normalized convolution, so that empty
    (NaN) nodes neither bleed zeros into the data nor get filled in."""
    if sigma_px is None or sigma_px <= 0:
        return img
    valid = np.isfinite(img)
    if not valid.any():
        return img
    num = gaussian_filter(np.where(valid, img, 0.0).astype(np.float64),
                          sigma_px, mode="nearest")
    den = gaussian_filter(valid.astype(np.float64), sigma_px, mode="nearest")
    out = np.full(img.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=valid & (den > 1e-6), casting="unsafe")
    return out


# ==========================================
# 3. Core: adaptive complex diffusion despeckling filter
# ==========================================
_G3 = np.exp(-(np.arange(-1, 2)[:, None] ** 2 + np.arange(-1, 2)[None, :] ** 2)
             / (2.0 * 10.0 ** 2))
_G3 /= _G3.sum()           # local-mean kernel G(N=3, sigma=10) of Bernardes+2010


def _fill_nearest(img, valid):
    """Fill NaN nodes with the nearest valid node (only to run the PDE)."""
    if valid.all():
        return img.astype(np.float64)
    idx = distance_transform_edt(~valid, return_distances=False,
                                 return_indices=True)
    return img[tuple(idx)].astype(np.float64)


def despeckle_acd(img, T=DESPECKLE_T, scale_tecu=DESPECKLE_SCALE,
                  theta=np.pi / 30.0, kmin=2.0, kmax=28.0,
                  alpha=4.0, a=0.25, b=0.75, max_iter=1000):
    """Improved adaptive nonlinear complex diffusion despeckling filter.

    Bernardes et al. (2010), Opt. Express 18(23), 24048; diffusion process of
    Gilboa, Sochen & Zeevi (2004), IEEE TPAMI 26(8), 1020:

        dI/dt = div( D grad I ),   D = exp(i theta) / (1 + (Im(I)/(k theta))^2)

    For small theta, Im(I)/theta behaves as a smoothed Laplacian of the image
    (edge detector), so diffusion is strong in flat/noisy areas and weak
    across wave fronts. "Adaptive" =
        k  = kmax + (kmin - kmax) (g - min g)/(max g - min g),  g = G * Re(I)
        dt = (1/alpha) [a + b exp(-max(|Re(dI/dt)| / Re(I)))]

    The constants were tuned for 8-bit images, so dTEC is mapped linearly,
    -scale_tecu -> 0 and +scale_tecu -> 255 (fixed mapping, identical for all
    frames so the time series stay consistent; values are NOT clipped) and
    mapped back afterwards. NaN nodes are filled only to integrate the PDE
    and are restored as NaN in the output.

    T (total diffusion time) controls the strength: T=0 returns the input.
    """
    valid = np.isfinite(img)
    if T is None or T <= 0 or not valid.any():
        return img
    g2l = 255.0 / (2.0 * scale_tecu)
    I = ((_fill_nearest(img, valid) + scale_tecu) * g2l).astype(np.complex128)
    rot = np.exp(1j * theta)

    t, it = 0.0, 0
    while t < T - 1e-9 and it < max_iter:
        re = I.real
        g = convolve(re, _G3, mode="nearest")
        gmin, gmax = g.min(), g.max()
        k = kmax + (kmin - kmax) * (g - gmin) / (gmax - gmin) if gmax > gmin \
            else np.full_like(g, kmax)
        D = rot / (1.0 + (I.imag / (k * theta)) ** 2)

        # div(D grad I): 4-neighbour scheme, D averaged at the half points,
        # zero-flux (mirrored) boundaries
        P, Q = np.pad(I, 1, mode="edge"), np.pad(D, 1, mode="edge")
        dI = 0.5 * ((Q[:-2, 1:-1] + D) * (P[:-2, 1:-1] - I)
                    + (Q[2:, 1:-1] + D) * (P[2:, 1:-1] - I)
                    + (Q[1:-1, :-2] + D) * (P[1:-1, :-2] - I)
                    + (Q[1:-1, 2:] + D) * (P[1:-1, 2:] - I))

        ratio = np.abs(dI.real) / np.maximum(np.abs(re), 1.0)
        dt = min((a + b * np.exp(-ratio.max())) / alpha, T - t)
        I = I + dt * dI
        t += dt
        it += 1

    out = (I.real / g2l - scale_tecu).astype(np.float32)
    out[~valid] = np.nan
    return out


# ==========================================
# 4. Container + IO
# ==========================================
@dataclass
class TecCube:
    """(time, lat, lon) dTEC maps. `dtec` = steps 3+4 (despeckle OFF);
    `dtec_despeckled` = steps 3+4+5 (despeckle ON), None if not computed."""
    times: np.ndarray                  # datetime64[s], frame start
    lats: np.ndarray
    lons: np.ndarray
    dtec: np.ndarray
    dtec_despeckled: np.ndarray = None
    n_ipp: np.ndarray = None           # IPPs used per frame
    attrs: dict = field(default_factory=dict)

    def get(self, despeckle: bool):
        """Select the ON/OFF product."""
        if not despeckle:
            print("[despeckle] OFF - Gaussian-smoothed maps. "
                  "To turn ON, pass despeckle=True.")
            return self.dtec
        if self.dtec_despeckled is None:
            raise ValueError("Cube was built with despeckle=False. "
                             "Run main_build(..., despeckle=True, overwrite=True).")
        print("[despeckle] ON - adaptive complex diffusion filter applied. "
              "To turn OFF, pass despeckle=False.")
        return self.dtec_despeckled

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if xr is None:
            path = path.with_suffix(".npz")
            np.savez_compressed(
                path, times=self.times, lats=self.lats, lons=self.lons,
                dtec=self.dtec, n_ipp=self.n_ipp,
                **({} if self.dtec_despeckled is None
                   else {"dtec_despeckled": self.dtec_despeckled}))
            print(f"Cube saved (xarray not found -> npz): {path}")
            return path
        dims = ("time", "lat", "lon")
        data = {"dtec": (dims, self.dtec, {"units": "TECU"}),
                "n_ipp": (("time",), self.n_ipp)}
        if self.dtec_despeckled is not None:
            data["dtec_despeckled"] = (dims, self.dtec_despeckled,
                                       {"units": "TECU"})
        ds = xr.Dataset(data,
                        coords={"time": self.times.astype("datetime64[ns]"),
                                "lat": self.lats, "lon": self.lons},
                        attrs={k: (v if isinstance(v, (int, float, str)) else str(v))
                               for k, v in self.attrs.items()})
        try:
            enc = {v: {"zlib": True, "complevel": 4} for v in data if v != "n_ipp"}
            ds.to_netcdf(str(path), encoding=enc)
        except Exception:              # backend without compression (scipy)
            ds.to_netcdf(str(path))
        print(f"Cube saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=False)
            return cls(z["times"], z["lats"], z["lons"], z["dtec"],
                       z["dtec_despeckled"] if "dtec_despeckled" in z else None,
                       z["n_ipp"])
        ds = xr.open_dataset(path)
        return cls(ds["time"].values.astype("datetime64[s]"),
                   ds["lat"].values, ds["lon"].values, ds["dtec"].values,
                   ds["dtec_despeckled"].values if "dtec_despeckled" in ds else None,
                   ds["n_ipp"].values, dict(ds.attrs))

    # -- helpers -----------------------------------------------------------
    def itime(self, when):
        """Index of the frame nearest to `when` ('HH:MM[:SS]' or datetime)."""
        if isinstance(when, str) and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", when):
            when = f"{str(self.times[0])[:10]}T{when}"
        return int(np.argmin(np.abs(self.times - np.datetime64(when, "s"))))

    def ilat(self, lat):
        return int(np.argmin(np.abs(self.lats - lat)))

    def ilon(self, lon):
        return int(np.argmin(np.abs(self.lons - lon)))


def day_start(year, doy):
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


def find_ipp_files(year, doy):
    folder = Path(dtec_ipp_dir(year, doy))
    files = set(folder.glob(f"*_dtecIPP_{year}{doy}.parquet"))
    files |= set(folder.glob(f"*_dtecIPP_{year}{doy:03d}.parquet"))
    return sorted(files)


def inspect_schema(year, doy):
    """Print index/columns of the first IPP parquet (sanity check)."""
    files = find_ipp_files(year, doy)
    if not files:
        print("No IPP files found.")
        return
    df = pd.read_parquet(files[0])
    print(files[0].name, "| index:", df.index.name, df.index.dtype)
    print(df.dtypes)


def _resolve_dtec_col(parquet_file, requested):
    """Checks which `dtec_<N>min` column is actually in the parquet schema
    (cheap: reads only metadata, not the data) and falls back to the
    closest available one if `requested` is missing, instead of crashing.

    Azeem et al. (2018), Section 2, detrend the VTEC with a 10-min running
    mean ("dtec_10min"). If your upstream IPP pipeline only produces a
    different window (e.g. "dtec_15min", as in visualize_keo_datashader.py),
    the analysis still runs, but the effective detrend differs from the
    paper by that amount -- worth knowing, not necessarily worth blocking on.
    """
    import pyarrow.parquet as pq
    names = pq.ParquetFile(parquet_file).schema.names
    if requested in names:
        return requested
    candidates = sorted(
        (n for n in names if re.fullmatch(r"dtec_\d+min", n)),
        key=lambda n: int(re.findall(r"\d+", n)[0]))
    if not candidates:
        raise ValueError(f"No 'dtec_*min' column found in {parquet_file}. "
                         f"Columns present: {names}")
    chosen = candidates[0]
    print(f"[dtec_col] WARNING: '{requested}' not found in the parquet "
         f"schema (available: {candidates}). Using '{chosen}' instead. "
         f"Azeem et al. (2018) use a 10-min running-mean detrend of VTEC "
         f"(Section 2) -- if your IPP pipeline only generates '{chosen}', "
         f"the detrend window differs from the paper by that amount.")
    return chosen


def load_ipp_points(parquet_files, t0, t1, lat_range, lon_range,
                    dtec_col=DTEC_COL, cadence_s=CADENCE_S):
    """Read every station parquet and return the IPP samples inside the
    time window / region as flat arrays (frame index, lon, lat, dTEC).

    Frame k holds the samples with  t0 + k*cadence <= t < t0 + (k+1)*cadence.
    """
    dtec_col = _resolve_dtec_col(parquet_files[0], dtec_col)
    frames, lons, lats, vals = [], [], [], []
    for f in parquet_files:
        col = dtec_col
        try:
            df = pd.read_parquet(f, columns=[LAT_COL, LON_COL, col])
        except Exception:
            col = _resolve_dtec_col(f, dtec_col)   # station file with a
            df = pd.read_parquet(f, columns=[LAT_COL, LON_COL, col])  # different schema
        t = df[TIME_COL] if TIME_COL in df.columns else df.index
        t = pd.DatetimeIndex(t)
        if t.tz is not None:
            t = t.tz_convert(None)
        lat = df[LAT_COL].to_numpy(np.float64)
        lon = df[LON_COL].to_numpy(np.float64)
        val = df[col].to_numpy(np.float32)
        keep = ((t >= t0) & (t < t1)
                & np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
                & (lat >= lat_range[0]) & (lat <= lat_range[1])
                & (lon >= lon_range[0]) & (lon <= lon_range[1]))
        if not keep.any():
            continue
        sec = (t[keep] - t0).total_seconds().to_numpy()
        frames.append(np.floor(sec / cadence_s).astype(np.int32))
        lons.append(lon[keep].astype(np.float32))
        lats.append(lat[keep].astype(np.float32))
        vals.append(val[keep])
    if not frames:
        raise ValueError("No IPP samples inside the requested window/region.")
    return (np.concatenate(frames), np.concatenate(lons),
            np.concatenate(lats), np.concatenate(vals))


# ==========================================
# 5. Build the cube (steps 3-4-5)
# ==========================================
def build_tec_cube(frame, lon, lat, val, t0, n_frames, lat_range, lon_range,
                   res=GRID_RES_DEG, cadence_s=CADENCE_S,
                   nn_max_dist_deg=NN_MAX_DIST_DEG,
                   gauss_fwhm_deg=GAUSS_FWHM_DEG,
                   despeckle=DESPECKLE, despeckle_T=DESPECKLE_T,
                   despeckle_scale=DESPECKLE_SCALE, min_points=3,
                   verbose=True):
    lats, lons = build_grid_axes(lat_range, lon_range, res)
    shape = (lats.size, lons.size)
    glon, glat = np.meshgrid(lons, lats)
    nodes = np.column_stack([glon.ravel(), glat.ravel()])
    sigma_px = fwhm_to_sigma_px(gauss_fwhm_deg, res) if gauss_fwhm_deg else 0.0

    if verbose:
        print(f"Grid {shape[0]} x {shape[1]} @ {res} deg | {n_frames} frames "
              f"@ {cadence_s} s | NN max dist = {nn_max_dist_deg} deg")
        if sigma_px > 0:
            att = gaussian_attenuation(180.0, gauss_fwhm_deg)
            print(f"[gaussian] FWHM = {gauss_fwhm_deg} deg ({sigma_px:.2f} px "
                  f"sigma). Amplitude response at 180 km: {att:.2e}"
                  + ("  <-- WAVE ERASED, see header note" if att < 0.05 else ""))
        print(f"[despeckle] {'ON' if despeckle else 'OFF'}"
              + (f" - T = {despeckle_T}, scale = +-{despeckle_scale} TECU"
                 if despeckle else ""))

    order = np.argsort(frame, kind="stable")
    frame, lon, lat, val = frame[order], lon[order], lat[order], val[order]
    edges = np.searchsorted(frame, np.arange(n_frames + 1))

    cube = np.full((n_frames,) + shape, np.nan, dtype=np.float32)
    cube_d = np.full_like(cube, np.nan) if despeckle else None
    n_ipp = np.diff(edges).astype(np.int32)

    for k in range(n_frames):
        i0, i1 = edges[k], edges[k + 1]
        if i1 - i0 < min_points:
            continue
        g = grid_nearest(lon[i0:i1], lat[i0:i1], val[i0:i1],
                         nodes, shape, nn_max_dist_deg)          # step 3
        g = gaussian_nan(g, sigma_px)                            # step 4
        cube[k] = g
        if despeckle:                                            # step 5
            cube_d[k] = despeckle_acd(g, T=despeckle_T,
                                      scale_tecu=despeckle_scale)
        if verbose and (k % 100 == 0 or k == n_frames - 1):
            print(f"  frame {k + 1}/{n_frames}  ({i1 - i0} IPPs)")

    times = (np.datetime64(t0, "s")
             + (np.arange(n_frames) * cadence_s).astype("timedelta64[s]"))
    attrs = dict(reference="Azeem et al. (2018) doi:10.1029/2018GL078107",
                 grid_res_deg=res, cadence_s=cadence_s,
                 nn_max_dist_deg=str(nn_max_dist_deg),
                 gauss_fwhm_deg=gauss_fwhm_deg, despeckle=str(despeckle),
                 despeckle_T=despeckle_T, despeckle_scale_tecu=despeckle_scale)
    return TecCube(times, lats, lons, cube, cube_d, n_ipp, attrs)


# ==========================================
# 6. Morlet CWT (Torrence & Compo 1998)
# ==========================================
def _fill_gaps_1d(x, max_nan_frac):
    """Linear interpolation of NaN gaps along the last axis. Rows with more
    than `max_nan_frac` missing become all-NaN."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = x.shape[-1]
    idx = np.arange(n)
    flat = x.reshape(-1, n)
    for row in flat:
        bad = ~np.isfinite(row)
        if bad.mean() > max_nan_frac or (~bad).sum() < 4:
            row[:] = np.nan
        elif bad.any():
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return flat.reshape(x.shape)


def wavelet_periods(pmin=WAVELET_PERIODS[0], pmax=WAVELET_PERIODS[1], dj=0.05):
    return pmin * 2.0 ** np.arange(0, np.log2(pmax / pmin) + dj / 2, dj)


def morlet_power(x, dt, periods, omega0=WAVELET_OMEGA0, at_index=None):
    """Morlet wavelet power |W|^2 along the last axis of `x`.

    x : (..., nt)   dt, periods : same time unit
    Returns (..., nper, nt), or (..., nper) when `at_index` is given
    (only that time sample is kept -> cheap for the Fig. 4 cuts).
    Rows containing NaN return NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    nt = x.shape[-1]
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        x = x - np.nanmean(x, axis=-1, keepdims=True)  # all-NaN row -> NaN (fine)
    npad = int(2 ** np.ceil(np.log2(nt) + 1))          # zero padding
    xf = np.fft.fft(x, n=npad, axis=-1)
    omega = 2.0 * np.pi * np.fft.fftfreq(npad, dt)
    scales = periods * (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4.0 * np.pi)

    out_shape = x.shape[:-1] + (periods.size,) + (() if at_index is not None else (nt,))
    out = np.empty(out_shape, dtype=np.float64)
    for j, s in enumerate(scales):
        psi = (np.pi ** -0.25) * np.sqrt(2.0 * np.pi * s / dt) \
            * np.exp(-0.5 * (s * omega - omega0) ** 2) * (omega > 0)
        W = np.fft.ifft(xf * psi, axis=-1)[..., :nt]
        if at_index is not None:
            out[..., j] = np.abs(W[..., at_index]) ** 2
        else:
            out[..., j, :] = np.abs(W) ** 2
    return out


def coi_period(nt, dt, omega0=WAVELET_OMEGA0):
    """Cone of influence, as the largest reliable period at each sample."""
    ff = 4.0 * np.pi / (omega0 + np.sqrt(2.0 + omega0 ** 2))
    edge = np.minimum(np.arange(nt), np.arange(nt)[::-1]) * dt
    return ff * edge / np.sqrt(2.0)


def wavelet_cuts(cube: TecCube, when, fixed_lat, fixed_lon, despeckle=DESPECKLE,
                 periods=None, max_nan_frac=0.2):
    """Fig. 4: for every node along the fixed-latitude row (and along the
    fixed-longitude column) the dTEC time series is wavelet transformed and
    the power at time `when` is kept. Each panel is normalized to its maximum.
    """
    data = cube.get(despeckle)
    periods = wavelet_periods() if periods is None else np.asarray(periods)
    dt_min = float((cube.times[1] - cube.times[0]) / np.timedelta64(1, "s")) / 60.0
    it, iy, ix = cube.itime(when), cube.ilat(fixed_lat), cube.ilon(fixed_lon)

    reach = coi_period(cube.times.size, dt_min)[it]
    if reach < periods.max():
        print(f"[wavelet] WARNING: {when} is close to the edge of the cube; "
              f"periods > {reach:.1f} min are inside the cone of influence. "
              f"Build a wider TIME_RANGE.")

    res = {}
    for name, series in (("lon", data[:, iy, :].T), ("lat", data[:, :, ix].T)):
        p = morlet_power(_fill_gaps_1d(series, max_nan_frac), dt_min, periods,
                         at_index=it)                   # (npix, nper)
        res[name] = (p / np.nanmax(p)).T                # (nper, npix)
    return dict(periods=periods, time=cube.times[it],
                lat_used=float(cube.lats[iy]), lon_used=float(cube.lons[ix]),
                power_vs_lon=res["lon"], power_vs_lat=res["lat"])


# ==========================================
# 7. Plots
# ==========================================
def _footer(fig, func):
    fig.text(0.99, -0.03, f"Script: gnss-tec-pipeline/azeem_tec_maps.py\n"
             f"Function: {func}", ha="right", va="top", fontsize=7,
             color="red")


def _map_axes(fig, pos, use_cartopy):
    if use_cartopy:
        ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        return ax
    return fig.add_subplot(*pos)


def _draw_map(ax, cube, img, vlim, cmap, storm_center, use_cartopy, zoom=None):
    cm = mpl.colormaps[cmap].copy()
    cm.set_bad("white")                      # no-data holes, as in Fig. 3
    r = (cube.lats[1] - cube.lats[0]) / 2
    extent = [cube.lons[0] - r, cube.lons[-1] + r,
              cube.lats[0] - r, cube.lats[-1] + r]
    kw = dict(transform=ccrs.PlateCarree()) if use_cartopy else {}
    im = ax.imshow(img, extent=extent, origin="lower", cmap=cm, vmin=-vlim,
                   vmax=vlim, interpolation="nearest",
                   aspect=None if use_cartopy else "auto", **kw)
    if storm_center is not None:
        ax.plot(storm_center[1], storm_center[0], "x", color="red", ms=8,
                mew=2, **kw)
    if use_cartopy:
        ax.set_extent(zoom or extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linestyle=":", alpha=0.4)
        gl.top_labels = gl.right_labels = False
    else:
        if zoom:
            ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
        lat_mid = np.deg2rad(np.mean(ax.get_ylim()))
        ax.set_aspect(1.0 / np.cos(lat_mid))
        ax.set_xlabel("Longitude, deg"); ax.set_ylabel("Latitude, deg")
        ax.grid(True, linestyle=":", alpha=0.4)
    return im


def plot_map_panels(cube: TecCube, whens, out_path, despeckle=DESPECKLE,
                    storm_center=None, vlim=MAP_VLIM_TECU, cmap=MAP_CMAP,
                    zoom=None, use_cartopy="auto", dpi=200):
    """Fig. 3: sequence of dTEC maps (default 2 x 2 for four times)."""
    data = cube.get(despeckle)
    use_cartopy = (ccrs is not None) if use_cartopy == "auto" else bool(use_cartopy)
    n = len(whens)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(6.5 * ncol, 5.2 * nrow + 0.8), layout="constrained")
    axs = []
    for i, when in enumerate(whens):
        it = cube.itime(when)
        ax = _map_axes(fig, (nrow, ncol, i + 1), use_cartopy)
        axs.append(ax)
        im = _draw_map(ax, cube, data[it], vlim, cmap, storm_center,
                       use_cartopy, zoom)
        ts = pd.Timestamp(cube.times[it])
        ax.set_title(f"({'abcdefghijkl'[i % 12]}) {ts:%m/%d/%Y}  UT = {ts:%H:%M:%S}",
                     fontweight="bold")
    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.7, aspect=45,
                 pad=0.03).set_label(
        f"dTEC (TECU)  |  despeckle {'ON' if despeckle else 'OFF'}")
    _footer(fig, "plot_map_panels")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")


def export_frames(cube: TecCube, out_dir, t_start=None, t_end=None,
                  despeckle=DESPECKLE, storm_center=None, vlim=MAP_VLIM_TECU,
                  cmap=MAP_CMAP, zoom=None, use_cartopy="auto", dpi=120):
    """One PNG per 30-s frame, for people who want to assemble the movie
    themselves (or inspect individual frames). For an actual movie file,
    use `export_movie` instead -- it's what main_plot(movie=True) calls."""
    data = cube.get(despeckle)
    use_cartopy = (ccrs is not None) if use_cartopy == "auto" else bool(use_cartopy)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    i0 = 0 if t_start is None else cube.itime(t_start)
    i1 = cube.times.size - 1 if t_end is None else cube.itime(t_end)
    for it in range(i0, i1 + 1):
        fig = plt.figure(figsize=(7, 6))
        ax = _map_axes(fig, (1, 1, 1), use_cartopy)
        im = _draw_map(ax, cube, data[it], vlim, cmap, storm_center,
                       use_cartopy, zoom)
        ts = pd.Timestamp(cube.times[it])
        ax.set_title(f"{ts:%m/%d/%Y}  UT = {ts:%H:%M:%S}", fontweight="bold")
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1,
                     shrink=0.8).set_label("dTEC (TECU)")
        fig.savefig(out_dir / f"frame_{ts:%Y%m%d_%H%M%S}.png", dpi=dpi,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"{i1 - i0 + 1} frames saved: {out_dir}")


def export_movie(cube: TecCube, out_path, t_start=None, t_end=None,
                 despeckle=DESPECKLE, storm_center=None, vlim=MAP_VLIM_TECU,
                 cmap=MAP_CMAP, zoom=None, use_cartopy="auto", fps=10, dpi=110):
    """Movie S1 equivalent: an actual animation of the dTEC maps, one frame
    per 30-s sample. Tries to write an .mp4 via ffmpeg (matplotlib's
    FFMpegWriter); if ffmpeg isn't on PATH, falls back automatically to an
    animated .gif via Pillow (always available, no extra install) with the
    same file stem. This is what `main_plot(..., movie=True)` calls -- it's
    the reason you weren't finding a video before: the previous version
    only wrote loose PNG frames (`export_frames`, still available below) and
    left assembling them into a video as a manual ffmpeg step for you."""
    from matplotlib import animation

    data = cube.get(despeckle)
    use_cartopy = (ccrs is not None) if use_cartopy == "auto" else bool(use_cartopy)
    i0 = 0 if t_start is None else cube.itime(t_start)
    i1 = cube.times.size - 1 if t_end is None else cube.itime(t_end)
    idx = range(i0, i1 + 1)

    fig = plt.figure(figsize=(7, 6))
    ax = _map_axes(fig, (1, 1, 1), use_cartopy)
    im = _draw_map(ax, cube, data[i0], vlim, cmap, storm_center, use_cartopy, zoom)
    title = ax.set_title("", fontweight="bold")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1,
                shrink=0.8).set_label("dTEC (TECU)")
    _footer(fig, "export_movie")

    def update(it):
        im.set_data(data[it])
        ts = pd.Timestamp(cube.times[it])
        title.set_text(f"{ts:%m/%d/%Y}  UT = {ts:%H:%M:%S}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=idx, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest = out_path.with_suffix(".mp4")
        anim.save(str(dest), writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
                  dpi=dpi)
    except Exception as exc:
        print(f"[movie] ffmpeg unavailable or failed ({exc}); "
             f"saving an animated GIF instead (no extra install needed).")
        dest = out_path.with_suffix(".gif")
        anim.save(str(dest), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Movie saved: {dest}  ({len(idx)} frames @ {fps} fps)")
    return dest


def plot_wavelet_cuts(cube: TecCube, when, fixed_lat, fixed_lon, out_path,
                      despeckle=DESPECKLE, lon_lim=None, lat_lim=None, dpi=200):
    """Fig. 4: normalized wavelet spectrograms, (a) fixed latitude and
    (b) fixed longitude, at UT = `when`."""
    w = wavelet_cuts(cube, when, fixed_lat, fixed_lon, despeckle)
    ts = pd.Timestamp(w["time"])
    levels = np.linspace(0, 1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    panels = (
        (axes[0], cube.lons, w["power_vs_lon"], "Lon. [Deg]", lon_lim,
         f"(a)  LATITUDE = {w['lat_used']:.2f}"),
        (axes[1], cube.lats, w["power_vs_lat"], "Lat. [Deg]", lat_lim,
         f"(b)  LONGITUDE = {w['lon_used']:.2f}"))
    for ax, x, p, xlabel, lim, tag in panels:
        cs = ax.contourf(x, w["periods"], np.nan_to_num(p), levels=levels,
                         cmap="jet")
        ax.set_facecolor(mpl.colormaps["jet"](0.0))
        ax.set_ylim(0, WAVELET_PERIODS[1])
        if lim:
            ax.set_xlim(lim)
        ax.set_xlabel(xlabel); ax.set_ylabel("Period [min]")
        ax.set_title(f"Normalized Wavelet Spectrogram\n{tag}   UT = {ts:%H:%M:%S}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.colorbar(cs, ax=axes, ticks=[0, .25, .5, .75, 1])
    _footer(fig, "plot_wavelet_cuts")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")
    return w


def plot_wavelet_time(cube: TecCube, lat, lon, out_path, despeckle=DESPECKLE,
                      box_deg=0.0, dpi=200):
    """Wavelet spectrogram (period x UT) of the dTEC sampled at one point
    near the storm centre (Section 5 of the paper, 'not shown here').
    box_deg > 0 averages the nodes within +-box_deg before the transform."""
    data = cube.get(despeckle)
    iy, ix = cube.ilat(lat), cube.ilon(lon)
    h = int(round(box_deg / (cube.lats[1] - cube.lats[0])))
    with np.errstate(all="ignore"):
        series = np.nanmean(data[:, max(iy - h, 0):iy + h + 1,
                                 max(ix - h, 0):ix + h + 1], axis=(1, 2))
    periods = wavelet_periods()
    dt_min = float((cube.times[1] - cube.times[0]) / np.timedelta64(1, "s")) / 60.0
    p = morlet_power(_fill_gaps_1d(series, 0.5), dt_min, periods)
    p = p / np.nanmax(p)
    tnum = pd.DatetimeIndex(cube.times)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                             constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 2]))
    axes[0].plot(tnum, series, "k", lw=0.8)
    axes[0].set_ylabel("dTEC (TECU)")
    axes[0].set_title(f"dTEC at lat = {cube.lats[iy]:.2f}, lon = {cube.lons[ix]:.2f}"
                      f"  |  despeckle {'ON' if despeckle else 'OFF'}")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    cs = axes[1].contourf(tnum, periods, np.nan_to_num(p),
                          levels=np.linspace(0, 1, 13), cmap="jet")
    coi = coi_period(tnum.size, dt_min)
    axes[1].set_facecolor(mpl.colormaps["jet"](0.0))
    axes[1].fill_between(tnum, np.clip(coi, periods.min(), periods.max()),
                         periods.max(),
                         facecolor="white", alpha=0.6, hatch="//",
                         edgecolor="gray", linewidth=0)
    axes[1].set_ylim(0, WAVELET_PERIODS[1])
    axes[1].set_ylabel("Period [min]"); axes[1].set_xlabel("Time (UTC)")
    axes[1].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.colorbar(cs, ax=axes[1], ticks=[0, .25, .5, .75, 1]).set_label(
        "Normalized power")
    _footer(fig, "plot_wavelet_time")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_path}")


# ==========================================
# 8. Storm centre from the storms catalogue (optional)
# ==========================================
def storm_center_from_catalog(t_start, t_end, lat_range, lon_range):
    """(lat, lon) of the coldest cloud top in the window, from the same
    catalogue used by the keogram overlay. Returns None if unavailable."""
    try:
        df = pd.read_parquet(storms_SouthAmerica(), engine="fastparquet")
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert(None)
        df = df[(dt >= t_start) & (dt < t_end)
                & df["centroid_lat"].between(*lat_range)
                & df["centroid_lon"].between(*lon_range)]
        if df.empty:
            return None
        row = df.loc[df["temp_min_C"].idxmin()]
        print(f"[storm] coldest top {row['temp_min_C']:.1f} C at "
              f"({row['centroid_lat']:.2f}, {row['centroid_lon']:.2f}) "
              f"{row['datetime']}")
        return float(row["centroid_lat"]), float(row["centroid_lon"])
    except Exception as exc:
        print(f"[storm] catalogue not used ({exc}).")
        return None


# ==========================================
# 9. Entry points
# ==========================================
def dtecmaps_dir(year, doy):
    """Single output folder for everything this script produces: the cube
    (.nc/.npz), the Fig.3/Fig.4-style PNGs and the movie. Mirrors the same
    DB root your other scripts already use -- takes dtec_keogram_dir(year,
    doy) (e.g. .../database/ionosphere/keograms/2024/333) and swaps the
    "keograms" path segment for "dtecmaps", so everything from this script
    lands in a sibling tree instead of mixing with the keogram outputs:

        .../database/ionosphere/dtecmaps/<year>/<doy:03d>/

    Set DTECMAPS_ROOT below to an absolute path to override this outright
    (e.g. if "keograms" isn't a literal path segment in your setup)."""
    if DTECMAPS_ROOT is not None:
        d = Path(DTECMAPS_ROOT) / str(year) / f"{doy:03d}"
    else:
        keo_dir = Path(dtec_keogram_dir(year, doy))
        parts = list(keo_dir.parts)
        if "keograms" in parts:
            parts[parts.index("keograms")] = "dtecmaps"
            d = Path(*parts)
        else:                          # unknown layout: nest alongside it
            d = keo_dir.parent.parent / "dtecmaps" / str(year) / f"{doy:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Absolute override for dtecmaps_dir()'s root (folder, no year/doy), e.g.
# Path(r"D:\database\ionosphere\dtecmaps"). None -> auto-derive from
# dtec_keogram_dir by swapping "keograms" -> "dtecmaps" (see docstring above).
DTECMAPS_ROOT = None


def cube_path(year, doy, time_range):
    h0, h1 = time_range
    return (dtecmaps_dir(year, doy)
            / f"tecmap_azeem_{year}_{doy:03d}_{h0:05.2f}-{h1:05.2f}UT.nc")


def main_build(year, doy, lat_range, lon_range, time_range=(0, 24),
               dtec_col=DTEC_COL, despeckle=DESPECKLE, overwrite=False,
               **build_kwargs):
    """Stage 1 - steps 3-4-5 of the paper for every 30-s frame in
    `time_range` (UT hours). Both products are stored: `dtec` (despeckle OFF)
    and, when despeckle=True, `dtec_despeckled` (ON)."""
    path = cube_path(year, doy, time_range)
    if (path.exists() or path.with_suffix(".npz").exists()) and not overwrite:
        print(f"Cube already exists, skipping build: {path.name}")
        return path
    files = find_ipp_files(year, doy)
    if not files:
        print(f"No IPP files found in {dtec_ipp_dir(year, doy)}")
        return None
    print(f"Found {len(files)} station files. Reading '{dtec_col}'...")

    cadence = build_kwargs.get("cadence_s", CADENCE_S)
    t0 = day_start(year, doy) + pd.Timedelta(hours=time_range[0])
    t1 = day_start(year, doy) + pd.Timedelta(hours=time_range[1])
    n_frames = int(round((t1 - t0).total_seconds() / cadence))
    frame, lon, lat, val = load_ipp_points(files, t0, t1, lat_range, lon_range,
                                           dtec_col, cadence)
    print(f"{val.size:,} IPP samples in window/region.")
    cube = build_tec_cube(frame, lon, lat, val, t0, n_frames, lat_range,
                          lon_range, despeckle=despeckle, **build_kwargs)
    cube.attrs["dtec_col"] = dtec_col
    return cube.save(path)


def main_plot(year, doy, time_range, map_times, wavelet_time, storm_center=None,
              fixed_lat=None, fixed_lon=None, despeckle=DESPECKLE, zoom=None,
              movie=True, frames=False, fps=10):
    """Stage 2 - Fig. 3, Fig. 4, the period x UT spectrogram and, by
    default, the Movie S1-style animation (movie=True -> .mp4, or .gif if
    ffmpeg isn't installed). Set frames=True too if you also want the
    loose per-frame PNGs (export_frames)."""
    cube = TecCube.load(cube_path(year, doy, time_range))
    if storm_center is None:
        storm_center = storm_center_from_catalog(
            pd.Timestamp(cube.times[0]), pd.Timestamp(cube.times[-1]),
            (cube.lats[0], cube.lats[-1]), (cube.lons[0], cube.lons[-1]))
    if fixed_lat is None or fixed_lon is None:
        if storm_center is None:
            raise ValueError("Give storm_center or fixed_lat/fixed_lon.")
        fixed_lat = storm_center[0] if fixed_lat is None else fixed_lat
        fixed_lon = storm_center[1] if fixed_lon is None else fixed_lon

    tag = "dspkON" if despeckle else "dspkOFF"
    out = dtecmaps_dir(year, doy)
    plot_map_panels(cube, map_times, out / f"tecmaps_{year}{doy:03d}_{tag}.png",
                    despeckle, storm_center, zoom=zoom)
    plot_wavelet_cuts(cube, wavelet_time, fixed_lat, fixed_lon,
                      out / f"wavelet_cuts_{year}{doy:03d}_{tag}.png", despeckle)
    plot_wavelet_time(cube, fixed_lat, fixed_lon,
                      out / f"wavelet_time_{year}{doy:03d}_{tag}.png", despeckle)
    if movie:
        export_movie(cube, out / f"movie_{year}{doy:03d}_{tag}", despeckle=despeckle,
                    storm_center=storm_center, zoom=zoom, fps=fps)
    if frames:
        export_frames(cube, out / f"frames_{tag}", despeckle=despeckle,
                      storm_center=storm_center, zoom=zoom)


def parse_arguments():
    p = argparse.ArgumentParser(
        description="dTEC maps + wavelet spectrograms, Azeem et al. (2018).")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    p.add_argument("--lat", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--lon", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--ut", type=float, nargs=2, default=(0, 24), metavar=("H0", "H1"))
    p.add_argument("--storm", type=float, nargs=2, default=None, metavar=("LAT", "LON"))
    p.add_argument("--map-times", nargs="+", required=True, help="HH:MM[:SS] ...")
    p.add_argument("--wavelet-time", required=True, help="HH:MM[:SS]")
    p.add_argument("--no-despeckle", action="store_true",
                   help="turn the despeckling filter OFF")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ==========================================
# 10. IDE / CLI entry point
# ==========================================
if __name__ == "__main__":

    RUN_MANUALLY = True
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")
        year, doy = 2021, 296

        # Region / event taken from visualize_keo_datashader.py ------------
        LAT_RANGE = (-43.0, -10.0)
        LON_RANGE = (-65.0, -45.0)
        STORM_CENTER = (-30.23, -53.57)      # RS (lat, lon); None -> catalogue

        # A few hours around the event is enough (Fig. 3/4 need only the
        # MAP_TIMES/WAVELET_TIME below, with some margin so the wavelet's
        # cone of influence doesn't eat into them). No need for 24 h: that
        # reads every station file in full and grids 2880 frames for
        # nothing. Widen only if your event's storm/TID window is longer.
        TIME_RANGE = (13, 17)            # UT hours

        # ---- ON / OFF ----------------------------------------------------
        DESPECKLE_ON = True                  # False = pipeline without step 5

        # Times of the Fig. 3 panels and of the Fig. 4 spectrograms --------
        MAP_TIMES = ["21:47:00", "21:48:00", "21:49:00", "21:50:00"]
        WAVELET_TIME = "21:25:00"

        # inspect_schema(year, doy)          # check index / column names

        # Stage 1 - run once. Always build with despeckle=True: both the ON
        # and OFF products are stored and the switch below picks one.
        main_build(year, doy, LAT_RANGE, LON_RANGE, TIME_RANGE,
                   dtec_col=DTEC_COL, despeckle=True, overwrite=False,
                   # cadence_s=30, nn_max_dist_deg=1.0, gauss_fwhm_deg=0.3,
                   # despeckle_T=2.0, despeckle_scale=0.2,
                   )
        # Stage 2
        main_plot(year, doy, TIME_RANGE, MAP_TIMES, WAVELET_TIME,
                  storm_center=STORM_CENTER, despeckle=DESPECKLE_ON,
                  zoom=None,                 # [lon0, lon1, lat0, lat1]
                  movie=True,                # Movie S1 equivalent (.mp4, or
                                             # .gif if ffmpeg isn't on PATH)
                  frames=False)              # True -> ALSO dump loose PNG frames
        print(f"Everything saved under: {dtecmaps_dir(year, doy)}")
    else:
        print("Running in CLI Mode...")
        a = parse_arguments()
        main_build(a.year, a.doy, tuple(a.lat), tuple(a.lon), tuple(a.ut),
                   despeckle=True, overwrite=a.overwrite)
        main_plot(a.year, a.doy, tuple(a.ut), a.map_times, a.wavelet_time,
                  storm_center=tuple(a.storm) if a.storm else None,
                  despeckle=not a.no_despeckle)