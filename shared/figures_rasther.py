#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:07:41 2026

@author: hisashi
"""

# --- Visualization (matplotlib) -------------------------------------
# --- CRITICAL SETTING TO PREVENT PLOT DISPLAY ---
import matplotlib
matplotlib.use('Agg')  # Set the non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
# ------------------------------------------------
import matplotlib.dates as mdates # Time-axis formatting (HH:MM, etc.)

import numpy as np
import pandas as pd
from pathlib import Path          # OS-independent path manipulation

# ---------------------------------------------------------------------
# PLOT (v4): matplotlib scatter with storms overlay
# ---------------------------------------------------------------------
def plot_ipp_trajectory(
    data_pack: dict,
    output_dir: str = "imagens",
    cmap: str = "turbo",
    trh: float = 0.5
    ):
    """
    Generates ONE figure with two panels (Time x Lat and Time x Lon)
    using `plt.scatter`, and OVERLAYS the detected convective storms
    (CPTEC c13) as `vlines` whose COLOR depends on the minimum cloud
    top temperature (cmap 'Blues_r' between -95 and -65 C).

    This is the "rich" version used for combined dTEC x convective
    analysis. For a SIMPLE version (no storms), see
    `plot_lat_lon_scatter_mpl` below.
    """
    # --- Unpack the precomputed data pack ---------------------------
    prn = data_pack["prn"]
    df_lt = data_pack["df_lt"]
    df_ln = data_pack["df_ln"]
    df_dtec = data_pack["df_dtec"]
    t_index = data_pack["time_index"]
    date_string = data_pack["date_string"]
    t_start = data_pack["t_start"]
    t_end = data_pack["t_end"]

    df_storms = data_pack["df_storms"]

    # --- Convert to "long" format (one point per row) ---------------
    # Each row of the original DataFrame becomes N_stations points.
    # `np.repeat` replicates each timestamp once per station.
    t_rep = np.repeat(t_index, df_lt.shape[1])

    lt_flat = df_lt.to_numpy().ravel()
    ln_flat = df_ln.to_numpy().ravel()
    dtec_flat = df_dtec.to_numpy().ravel()

    # Masks: points with valid dTEC AND valid coordinate -------------
    mask_common = ~np.isnan(dtec_flat)
    mask_lat = mask_common & ~np.isnan(lt_flat)
    mask_lon = mask_common & ~np.isnan(ln_flat)

    # --- Storm centroid attributes ----------------------------------
    centroid_time = df_storms['datetime']
    centroid_lat = df_storms['centroid_lat']
    centroid_lon = df_storms['centroid_lon']
    centroid_radius = df_storms['radius_deg']
    centroid_temp = df_storms['temp_min_C']

    # Time formatter (HH:MM) on the X axis ---------------------------
    date_fmt = mdates.DateFormatter('%H:%M')

    # ---------------------------------------------------------------
    # Map storm color from the minimum cloud top temperature
    # (colder -> darker/bluer). 'Blues_r' makes -95 C dark navy
    # and -65 C light.
    # ---------------------------------------------------------------
    norm_storm = plt.Normalize(vmin=-95, vmax=-65)
    cmap_storm = plt.get_cmap('Blues_r')
    storm_colors = cmap_storm(norm_storm(centroid_temp.values))

    # --- Figure with 2 stacked panels (lat top, lon bottom) ---------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    cmap_use = cmap if cmap else "viridis"

    # PANEL 1 - Time x IPP latitude, colored by dTEC -----------------
    sc1 = axes[0].scatter(
        t_rep[mask_lat], lt_flat[mask_lat], c=dtec_flat[mask_lat],
        s=1, alpha=0.8, marker='|', cmap=cmap_use, vmin=-trh, vmax=trh,
        zorder=1
    )
    # Overlay: each storm becomes a vertical line whose height equals
    # the effective radius (in degrees), centered on the centroid latitude
    axes[0].vlines(
        x=centroid_time,
        ymin=centroid_lat - (centroid_radius/2),
        ymax=centroid_lat + (centroid_radius/2),
        colors=storm_colors,   # color encodes temperature (-95..-65 C)
        linewidth=3.,
        alpha=0.2,
        zorder=2,
        label='Storms'
    )

    axes[0].set_title(f"{date_string} [{prn}] Storm colorscale from -65 to -95 ºC")
    axes[0].set_ylabel("Latitude (°)")
    axes[0].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    cb1 = plt.colorbar(sc1, ax=axes[0])
    cb1.set_label("dTEC, TECU")

    # PANEL 2 - Time x IPP longitude, colored by dTEC ----------------
    sc2 = axes[1].scatter(
        t_rep[mask_lon], ln_flat[mask_lon], c=dtec_flat[mask_lon],
        s=1, alpha=0.8, marker='|', cmap=cmap_use, vmin=-trh, vmax=trh,
        zorder=1
        )

    axes[1].vlines(
        x=centroid_time,
        ymin=centroid_lon - (centroid_radius/2),
        ymax=centroid_lon + (centroid_radius/2),
        colors=storm_colors,
        linewidth=3.,
        alpha=0.2,
        zorder=2,
        label='Storms'
    )

    axes[1].set_ylabel("Longitude (°)")
    axes[1].set_xlabel(f"[{date_string}] Time (UTC)")
    axes[1].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    axes[1].set_xlim(t_start, t_end)
    axes[1].xaxis.set_major_formatter(date_fmt)

    cb2 = plt.colorbar(sc2, ax=axes[1])
    cb2.set_label("dTEC, TECU")

    # --- Save output ------------------------------------------------
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    save_path = Path(output_dir) / f"{date_string}_{prn}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {prn}")
    
def plot_ipp_trajectory2(
    df: pd.DataFrame,
    station: str,
    date_string: str,
    output_dir: str = "imagens",
    cmap: str = "turbo",
    trh: float = 0.5,
):
    """
    Two-panel figure (Time x Lat, Time x Lon) for all PRNs of one station.
    Expects a DataFrame with datetime index and columns: prn, lts, lns, els, dtec.

    Parameters
    ----------
    df          : DataFrame with datetime index + [prn, lts, lns, els, dtec]
    station     : station name, used in the output filename
    date_string : label for the plot title / filename (e.g. "2024-001")
    output_dir  : folder to save the figure
    cmap        : colormap for dTEC scatter
    trh         : symmetric dTEC color limits [-trh, +trh]
    """
    df      = df[df["els"] > 0].copy()
    t       = df.index
    t_hours = (t - t[0]).total_seconds() / 3600   # datetime → decimal hours
    lts     = df["lts"].to_numpy()
    lns     = df["lns"].to_numpy()

    dtec_col = next(col for col in df.columns if col.startswith("dtec"))
    dtec     = df[dtec_col].to_numpy()
    running_average = dtec_col.split('_')[1]

    mask_lat = ~np.isnan(lts) & ~np.isnan(dtec)
    mask_lon = ~np.isnan(lns) & ~np.isnan(dtec)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)

    # Panel 1 — Time x Lat
    sc1 = axes[0].scatter(
        t_hours[mask_lat], lts[mask_lat], c=dtec[mask_lat],
        s=1, alpha=0.8, marker="|", cmap=cmap, vmin=-trh, vmax=trh
    )
    axes[0].set_title(f"dtec [{date_string}] [{station}] (avg. window = {running_average})")
    axes[0].set_ylabel("Latitude (°)")
    axes[0].grid(True, linestyle=":", color="gray", alpha=0.7)
    plt.colorbar(sc1, ax=axes[0]).set_label("dTEC (TECU)")

    # Panel 2 — Time x Lon
    sc2 = axes[1].scatter(
        t_hours[mask_lon], lns[mask_lon], c=dtec[mask_lon],
        s=1, alpha=0.8, marker="|", cmap=cmap, vmin=-trh, vmax=trh
    )
    axes[1].set_ylabel("Longitude (°)")
    axes[1].set_xlabel(f"[{date_string}] Time (UTC)")
    axes[1].grid(True, linestyle=":", color="gray", alpha=0.7)
    axes[1].set_xlim(t_hours.min(), t_hours.max())
    axes[1].set_xlim(-0.5, 24.5)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(3))
    # axes[1].xaxis.set_major_locator(plt.MultipleLocator(1))   # tick a cada 1h
    # axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):02d}:00"))

    plt.colorbar(sc2, ax=axes[1]).set_label("dTEC (TECU)")

    # Save
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(output_dir) / f"{station}_{date_string}.png", dpi=150)
    plt.close(fig)
    # print(f"Saved: {station}")


# ---------------------------------------------------------------------
# PLOT (v5 - NEW): SIMPLE matplotlib scatter
# ---------------------------------------------------------------------
# This routine was added in v5 on request: for EACH .parquet file
# (each PRN), produces ONE figure with 2 stacked panels - Latitude
# vs Time on top and Longitude vs Time on the bottom - using ONLY a
# matplotlib scatter, with no storm overlays. Useful when the focus
# is on inspecting the dTEC signature without visual distractions.
# ---------------------------------------------------------------------
def plot_lat_lon_scatter_mpl(
    data_pack: dict,
    output_dir: str = "imagens_simple_scatter",
    cmap: str = "turbo",
    trh: float = 0.5,
    marker_size: float = 1.0,
    marker: str = "|",
    alpha: float = 0.8,
    dpi: int = 300,
    ):
    """
    For a single PRN/file, generates ONE figure with:
        - TOP panel    : IPP Latitude  vs time
        - BOTTOM panel : IPP Longitude vs time
    both as matplotlib scatter plots, with each point colored by
    the dTEC (TECU) on the symmetric scale [-trh, +trh].

    Parameters
    ----------
    data_pack : dict
        Output of `process_prn_file` (same structure used by
        `plot_ipp_trajectory`).
    output_dir : str
        Folder where the figure will be saved. Created if missing.
    cmap : str
        Matplotlib colormap applied to dTEC (e.g. "turbo",
        "seismic", "RdBu_r").
    trh : float
        Symmetric color-scale limit (vmin=-trh, vmax=+trh).
    marker_size : float
        Marker size (`s` of scatter).
    marker : str
        Marker style. Default '|' mimics the look of v4 figures
        (thin vertical lines that highlight passes).
    alpha : float
        Point opacity.
    dpi : int
        Resolution of the saved figure.

    Notes
    -----
    This function does NOT use `df_storms` - it is deliberately
    simple, for quick inspection of the pure scatter. For the
    version with storms, use `plot_ipp_trajectory`.
    """
    # --- 1. Unpack data ---------------------------------------------
    prn = data_pack["prn"]
    df_lt = data_pack["df_lt"]
    df_ln = data_pack["df_ln"]
    df_dtec = data_pack["df_dtec"]
    t_index = data_pack["time_index"]
    date_string = data_pack["date_string"]
    t_start = data_pack["t_start"]
    t_end = data_pack["t_end"]

    # --- 2. Flatten to per-point format -----------------------------
    # `t_rep` repeats each timestamp once per station, aligning with
    # the flattened latitudes/longitudes/dtec arrays.
    t_rep = np.repeat(t_index, df_lt.shape[1])
    lt_flat = df_lt.to_numpy().ravel()
    ln_flat = df_ln.to_numpy().ravel()
    dtec_flat = df_dtec.to_numpy().ravel()

    # Masks: drop entries with NaN in dTEC or in the coordinate
    mask_common = ~np.isnan(dtec_flat)
    mask_lat = mask_common & ~np.isnan(lt_flat)
    mask_lon = mask_common & ~np.isnan(ln_flat)

    # If there are no valid points on either axis, abort
    if not mask_lat.any() and not mask_lon.any():
        print(f"[plot_lat_lon_scatter_mpl] no valid points for {prn}; figure not generated.")
        return None

    # --- 3. Build the figure with 2 stacked panels (sharex) --------
    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 6),
        constrained_layout=True,
        sharex=True
    )
    date_fmt = mdates.DateFormatter('%H:%M')  # HH:MM on the X axis

    # --- 4. TOP PANEL: latitude vs time -----------------------------
    sc_lat = axes[0].scatter(
        t_rep[mask_lat],
        lt_flat[mask_lat],
        c=dtec_flat[mask_lat],
        s=marker_size,
        alpha=alpha,
        marker=marker,
        cmap=cmap,
        vmin=-trh,
        vmax=trh,
    )
    axes[0].set_title(f"{date_string} [{prn}] - simple scatter (matplotlib)")
    axes[0].set_ylabel("IPP Latitude (°)")
    axes[0].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    cb_lat = plt.colorbar(sc_lat, ax=axes[0])
    cb_lat.set_label("dTEC (TECU)")

    # --- 5. BOTTOM PANEL: longitude vs time -------------------------
    sc_lon = axes[1].scatter(
        t_rep[mask_lon],
        ln_flat[mask_lon],
        c=dtec_flat[mask_lon],
        s=marker_size,
        alpha=alpha,
        marker=marker,
        cmap=cmap,
        vmin=-trh,
        vmax=trh,
    )
    axes[1].set_ylabel("IPP Longitude (°)")
    axes[1].set_xlabel(f"[{date_string}] Time (UTC)")
    axes[1].grid(True, which='major', linestyle=':', color='gray', alpha=0.7)
    # Lock the X axis exactly to the processed day and format as HH:MM
    axes[1].set_xlim(t_start, t_end)
    axes[1].xaxis.set_major_formatter(date_fmt)
    cb_lon = plt.colorbar(sc_lon, ax=axes[1])
    cb_lon.set_label("dTEC (TECU)")

    # --- 6. Save the figure -----------------------------------------
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    save_path = Path(output_dir) / f"{date_string}_{prn}_scatter_mpl.png"
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved (simple scatter): {save_path.name}")
    return save_path