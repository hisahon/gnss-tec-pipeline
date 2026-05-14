#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:01:49 2026

@author: hisashi
"""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from datetime import datetime
# from typing import Optional


from shared.config import ephemeris_parquet_dir
GNSS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

def interpolate_orbits(year, doy) -> dict:
    """
    Cria interpoladores PCHIP de coordenadas para cada PRN.

    Parâmetros:
        df_orb      : DataFrame com colunas 'prn', 'horario' (GPS Time), 'x_km', 'y_km', 'z_km'
        common_prns : array com os PRNs a processar

    Retorna:
        dict → {prn: {'horario_sec': [...], 'fx': interpolador, 'fy': interpolador, 'fz': interpolador}}
    """
    
    file_ephemeris = ephemeris_parquet_dir(year, doy) / f"ORBIT_{year}{doy:03d}.parquet"
    df_orb = pd.read_parquet(file_ephemeris)
        
    prns = df_orb['prn'].unique()
        
    result = {}

    for prn in prns:
        group_orb = (df_orb[df_orb['prn'] == prn]
             .drop_duplicates(subset='horario')
             .sort_values('horario'))

        orb_t = (group_orb["horario"] - GNSS_EPOCH).dt.total_seconds().to_numpy()
        orb_x = group_orb['x_km'].to_numpy()
        orb_y = group_orb['y_km'].to_numpy()
        orb_z = group_orb['z_km'].to_numpy()

        result[prn] = {
            # 'horario_sec': orb_t,
            'fx_km'         : PchipInterpolator(orb_t, orb_x),
            'fy_km'         : PchipInterpolator(orb_t, orb_y),
            'fz_km'         : PchipInterpolator(orb_t, orb_z),
        }

    return result

# def satellite_xyz(df_rinex, df_orbit):
#     """
#     Add satellite ECEF coordinates (x_km, y_km, z_km) to the GNSS observation DataFrame
#     using interpolated SP3 orbit data.

#     Parameters:
#         df_rinex (pd.DataFrame): GNSS observation DataFrame with 'prn' and 'time_j' columns.
#         df_orbit (pd.DataFrame): SP3 orbit DataFrame used for interpolation.

#     Returns:
#         pd.DataFrame: Updated DataFrame with satellite coordinates.
#     """
#     prns = sorted(df_rinex['prn'].unique())
#     coord_cols = ['x_km', 'y_km', 'z_km']
#     df_rinex[coord_cols] = None  # Initialize columns
#     dropped = []

#     for prn in prns:
#         idxs = df_rinex['prn'] == prn
#         try:
#             Inter_prn, keep_info = data_interpol(df_orbit, prn)
#             if isinstance(Inter_prn, str):
#                 # print(f"[SKIP] {prn}: {keep_info}")
#                 dropped.append(prn)
#                 df_rinex = df_rinex[~idxs]
#                 continue

#             time_j = df_rinex.loc[idxs, 'time_j']
#             prn_xyz = coord_cartesian(Inter_prn, time_j)

#             df_rinex.loc[idxs, coord_cols] = prn_xyz.values

#         except Exception as e:
#             print(f"[ERROR] {prn}: {e}")
#             dropped.append(prn)
#             df_rinex = df_rinex[~idxs]

#     # if dropped:
#     #     print(f"[INFO] Dropped PRNs: {dropped} (total: {len(dropped)})")

#     return df_rinex, dropped

def satellite_xyz_v2(df_tec_dtec, orbit_dict):
    
    # df_tec_dtec = pd.read_parquet(dtec_file)
    # Coordinate columns to be filled
    coord_cols = ['x_km', 'y_km', 'z_km']
    df_tec_dtec[coord_cols] = np.nan

    # Find PRNs present in both the dataframe and the orbit dictionary
    prns_in_data = df_tec_dtec['prn'].unique()
    common_prns = np.intersect1d(prns_in_data, list(orbit_dict))

    dropped = []
    for prn in common_prns:
        # Boolean mask for rows belonging to this PRN
        mask = df_tec_dtec['prn'] == prn
        prn_slice = df_tec_dtec[mask]

        # Convert timestamps to seconds since GNSS epoch
        times = (prn_slice["horario"] - GNSS_EPOCH).dt.total_seconds().to_numpy()

        try:
            # Interpolate satellite coordinates using precomputed spline functions
            df_tec_dtec.loc[mask, 'x_km'] = orbit_dict[prn]['fx_km'](times)
            df_tec_dtec.loc[mask, 'y_km'] = orbit_dict[prn]['fy_km'](times)
            df_tec_dtec.loc[mask, 'z_km'] = orbit_dict[prn]['fz_km'](times)
        except Exception as e:
            print(f"[ERROR] PRN {prn}: {e}")
            dropped.append(prn)

    # Remove rows belonging to PRNs that raised errors
    if dropped:
        df_tec_dtec = df_tec_dtec[~df_tec_dtec['prn'].isin(dropped)]
        print(f"[INFO] Dropped PRNs: {dropped} (total: {len(dropped)})")

    # Drop rows with missing coordinates (PRNs absent from orbit_dict)
    df_tec_dtec.dropna(subset=coord_cols, inplace=True)
    
    return df_tec_dtec

# ---------------------------------------------------------------------
# GEOMETRY: IPP (THIN-SHELL)
# ---------------------------------------------------------------------
def ipp_calc(prn_coordinates, station_coordinates, H_km, RE_km=6371.0):
    """
    Computes the Ionospheric Pierce Point (IPP) and the elevation at
    the receiver using the classical thin-shell approach (thin
    spherical shell), starting from the satellite and station ECEF
    coordinates.

    This version uses a trigonometric formulation based on ENU plus
    the local zenith angle (chi). See `ipp_geometric` for the
    alternative formulation via ray/sphere intersection (radius RE+H).

    Parameters
    ----------
    prn_coordinates : tuple of arrays
        (x_p, y_p, z_p) - satellite ECEF in km, vectors of length N.
    station_coordinates : tuple
        (x_s, y_s, z_s, lon_s_deg, lat_s_deg, h_s) - station ECEF +
        geodetic lon/lat in DEGREES + height.
    H_km : float
        Height of the ionospheric shell above the Earth's surface.
    RE_km : float
        Mean Earth radius in km.

    Returns
    -------
    (lat_pp_deg, lon_pp_deg, El_deg) : tuple of arrays
        IPP latitude and longitude in degrees, and geometric
        satellite elevation in degrees. Points with El<0 or
        outside the clipping masks become NaN.
    """
    # Unpack inputs
    x_s, y_s, z_s, ln_s, lt_s, h_s = station_coordinates
    x_p, y_p, z_p = prn_coordinates

    # ECEF difference vector (satellite - station) -------------------
    dx = x_p - x_s
    dy = y_p - y_s
    dz = z_p - z_s

    # Convert station lat/lon to radians ----------------------------
    lat = np.radians(lt_s)
    lon = np.radians(ln_s)

    sphi, cphi = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    # ECEF -> ENU (East, North, Up) rotation at the station ---------
    e = -slon*dx +  clon*dy
    n = -clon*sphi*dx - slon*sphi*dy + cphi*dz
    u =  clon*cphi*dx + slon*cphi*dy + sphi*dz

    # Geometric elevation and azimuth at the receiver ---------------
    hor = np.hypot(e, n)
    El  = np.arctan2(u, hor)          # elevation in [-pi/2, pi/2]
    Az  = np.arctan2(e, n)            # azimuth in [-pi, pi]

    # Thin-shell geometry -------------------------------------------
    # chi = zenith angle at the IPP; psi = geocentric arc station->IPP
    cosE  = np.cos(El)
    schi  = (RE_km/(RE_km + H_km)) * cosE
    schi  = np.clip(schi, -1.0, 1.0)  # numerical robustness
    chi   = np.arcsin(schi)

    psi = (np.pi/2.0) - El - chi      # arc between station and IPP

    spsi, cpsi = np.sin(psi), np.cos(psi)

    # IPP latitude via spherical navigation formula -----------------
    sin_lat_pp = sphi*cpsi + cphi*spsi*np.cos(Az)
    sin_lat_pp = np.clip(sin_lat_pp, -1.0, 1.0)
    lat_pp = np.arcsin(sin_lat_pp)

    # IPP longitude (delta longitude relative to the station) -------
    num = spsi*np.sin(Az)
    den = cphi*cpsi - sphi*spsi*np.cos(Az)
    dlon = np.arctan2(num, den)
    lon_pp = lon + dlon

    # Wrap longitude to the (-pi, pi] interval ----------------------
    lon_pp = (lon_pp + np.pi) % (2.0*np.pi) - np.pi

    # Mask: drop points with negative elevation (satellite below
    # the horizon - IPP has no physical meaning in that case) ------
    mask = El < 0.0
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)

    # ---------------------------------------------------------------
    # ADDITIONAL GEOGRAPHIC CLIPPING MASKS
    # (kept from the original draft - adjust to match your area of
    # interest; here IPPs north of 20N and east of -20E are dropped
    # as a focus on the South American sector)
    # ---------------------------------------------------------------
    
    # latitude > 20 deg mask
    mask = np.degrees(lat_pp) > 20
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)
    # longitude > -20 deg mask
    mask = np.degrees(lon_pp) > -20
    lat_pp = np.where(mask, np.nan, lat_pp)
    lon_pp = np.where(mask, np.nan, lon_pp)

    # Output in degrees ---------------------------------------------
    return np.degrees(lat_pp), np.degrees(lon_pp), np.degrees(El)


def ipp_geometric(prn_coordinates, station_coordinates, H_km, RE_km=6371.0):
    """
    Computes the Ionospheric Pierce Point (IPP) on a thin spherical
    shell and the geometric elevation at the receiver, for N
    satellite positions.

    DIFFERENCE FROM `ipp_calc`:
    This version directly solves the quadratic equation for the
    intersection between the receiver->satellite ray and the
    ionospheric sphere of radius RE+H. It is numerically more
    robust at large angles and easier to verify independently.

    Parameters
    ----------
    prn_coordinates : (x_p, y_p, z_p)
        Each is a 1D array of length N (km), satellite ECEF
        coordinates.
    station_coordinates : (x_s, y_s, z_s, lon_s, lat_s, h_s)
        Station coordinates:
        - x_s, y_s, z_s: ECEF (km)
        - lon_s, lat_s: longitude/latitude (assumed in rad)
        - h_s: height (not used here)
    H_km : float
        Height of the ionospheric shell above the surface (km).
    RE_km : float, optional
        Mean Earth radius (km).

    Returns
    -------
    lat_pp_deg : array (N,)
        Geocentric latitude of the IPP, in degrees.
    lon_pp_deg : array (N,)
        Geocentric longitude of the IPP, in degrees.
    El_deg : array (N,)
        Geometric elevation at the receiver, in degrees.
    """

    # Unpack inputs
    x_s, y_s, z_s, lon_s, lat_s, h_s = station_coordinates
    x_p, y_p, z_p = prn_coordinates

    # Receiver position vector: r (station in ECEF)
    r = np.array([x_s, y_s, z_s], dtype=float)  # (3,)

    # Satellite position vector: s (N,3) - stacks the three 1D vectors
    s = np.vstack((x_p, y_p, z_p)).T           # (N,3)

    # LOS vector receiver->satellite: v = s - r
    v = s - r                                  # (N,3)

    # Ionospheric shell radius
    R_ion = RE_km + H_km

    # ---------------------------------------------------------------
    # Intersection equation: ||r + lambda v||^2 = R_ion^2
    # leads to the quadratic a*lambda^2 + b*lambda + c = 0
    # ---------------------------------------------------------------
    a = np.sum(v * v, axis=1)                  # (N,)  ||v||^2
    b = 2.0 * np.dot(v, r)                     # (N,)  2 r dot v
    r2 = np.dot(r, r)
    c = r2 - R_ion**2                          # scalar

    # Discriminant
    Delta = b**2 - 4.0 * a * c

    # sqrt(Delta), clipping numerical negatives to avoid spurious NaNs
    sqrtD = np.sqrt(np.maximum(Delta, 0.0))

    # Roots
    lambda1 = (-b + sqrtD) / (2.0 * a)
    lambda2 = (-b - sqrtD) / (2.0 * a)

    # ---------------------------------------------------------------
    # Pick the physical root: the SMALLEST POSITIVE lambda (first
    # forward intersection along the satellite direction). If both
    # are positive, take the smaller; otherwise take whichever is
    # positive.
    # ---------------------------------------------------------------
    lambda_star = np.full_like(lambda1, np.nan, dtype=float)

    cond1 = (lambda1 > 0.0) & (lambda2 > 0.0)
    lambda_star[cond1] = np.minimum(lambda1[cond1], lambda2[cond1])

    cond2 = (lambda1 > 0.0) & ~cond1
    lambda_star[cond2] = lambda1[cond2]

    cond3 = (lambda2 > 0.0) & ~cond1 & ~cond2
    lambda_star[cond3] = lambda2[cond3]

    # Pierce point: p* = r + lambda* v
    p_star = r + lambda_star[:, None] * v      # (N,3)
    x_pp, y_pp, z_pp = p_star.T

    # Geocentric IPP latitude/longitude
    lon_pp = np.arctan2(y_pp, x_pp)
    lat_pp = np.arctan2(z_pp, np.hypot(x_pp, y_pp))

    # ---------------------------------------------------------------
    # Geometric elevation via the dot product between the LOS unit
    # direction and the local normal (zenith) at the receiver.
    # ---------------------------------------------------------------
    # Local normal at the receiver (zenith): n_r = r / ||r||
    n_r = r / np.sqrt(r2)                      # (3,)

    # Unit LOS direction: d = v / ||v||
    v_norm = np.sqrt(a)                        # (N,)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = v / v_norm[:, None]                # (N,3)

    # sin(El) = d dot n_r
    dn = np.dot(d, n_r)                        # (N,)
    dn = np.clip(dn, -1.0, 1.0)
    El = np.arcsin(dn)                         # (N,) rad

    # Mask cases with no physical intersection or below the horizon
    mask_bad = (Delta < 0.0) | (~np.isfinite(lambda_star)) | (El <= 0.0)

    lat_pp[mask_bad] = np.nan
    lon_pp[mask_bad] = np.nan
    El[mask_bad] = np.nan

    # Convert to degrees
    return np.degrees(lat_pp), np.degrees(lon_pp), np.degrees(El)