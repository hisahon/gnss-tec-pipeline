# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:27:46 2026

@author: hisas
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# Import custom modules
from shared.geometry import ipp_geometric
from shared.prn_store import load_tec_with_neighbors_v2
import shared.config as cfg

# ==========================================
# 1. Processing: Interpolate & Calculate IPP
# ==========================================
def compute_station_ipp(df_dtec, sat_filepath, rx_coords_tuple, H_km=350.0):
    """
    Synchronizes high-res dTEC with low-res satellite orbits and calculates IPP.
    """
    if df_dtec.empty:
        print("Warning: Input dTEC DataFrame is empty.")
        return pd.DataFrame()

    print("Loading satellite dataset...")
    df_sat = pd.read_parquet(sat_filepath)
    
    # Dynamically identify time column
    time_col = 'horario' if 'horario' in df_dtec.columns else df_dtec.columns[0]
    
    # Standardize satellite file columns
    if 'time' in df_sat.columns and time_col != 'time':
        df_sat = df_sat.rename(columns={'time': time_col})
    if 'PRN' in df_sat.columns:
        df_sat = df_sat.rename(columns={'PRN': 'prn'})

    # Convert datetimes to POSIX seconds for interpolation
    df_dtec['time_sec'] = pd.to_datetime(df_dtec[time_col]).astype('int64') // 10**9
    df_sat['time_sec'] = pd.to_datetime(df_sat[time_col]).astype('int64') // 10**9
    
    df_dtec = df_dtec.sort_values(by=['prn', 'time_sec'])
    df_sat = df_sat.sort_values(by=['prn', 'time_sec'])

    print("Interpolating satellite coordinates & computing IPP...")
    sat_coords = ['sat_x', 'sat_y', 'sat_z'] 
    processed_dfs = []
    
    for prn_val in df_dtec['prn'].unique():
        dtec_prn = df_dtec[df_dtec['prn'] == prn_val].copy()
        sat_prn = df_sat[df_sat['prn'] == prn_val].copy()
        
        if len(sat_prn) < 2:
            continue
            
        # Linear interpolation for satellite ECEF coordinates
        for col in sat_coords:
            interpolator = interp1d(
                sat_prn['time_sec'], 
                sat_prn[col], 
                kind='linear', 
                bounds_error=False, 
                fill_value=np.nan
            )
            dtec_prn[col] = interpolator(dtec_prn['time_sec'])
            
        dtec_prn = dtec_prn.dropna(subset=sat_coords)
        if dtec_prn.empty:
            continue
            
        # Vectorized geometric IPP calculation
        x_p = dtec_prn[sat_coords[0]].values
        y_p = dtec_prn[sat_coords[1]].values
        z_p = dtec_prn[sat_coords[2]].values
        
        lat_pp, lon_pp, el_deg = ipp_geometric(
            prn_coordinates=(x_p, y_p, z_p),
            station_coordinates=rx_coords_tuple,
            H_km=H_km
        )
        
        dtec_prn['ipp_lat'] = lat_pp
        dtec_prn['ipp_lon'] = lon_pp
        dtec_prn['elevation'] = el_deg
        
        # Mask points where satellite is below the horizon
        dtec_prn = dtec_prn.dropna(subset=['ipp_lat', 'ipp_lon'])
        
        if not dtec_prn.empty:
            processed_dfs.append(dtec_prn)
        
    if not processed_dfs:
        print("Warning: No valid data left after IPP calculation.")
        return pd.DataFrame()

    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.drop(columns=['time_sec'])
    
    print(f"Integration complete! {len(final_df)} IPP points generated.")
    return final_df

# ==========================================
# 2. Visualization: Generate the Plot
# ==========================================
def plot_ipp_map(df_ipp, rx_coords_tuple, station_name="Station", save_path=None):
    """
    Generates a 2D geographical scatter plot of dTEC values at their Piercing Points.
    """
    if df_ipp.empty:
        print("Cannot plot: DataFrame is empty.")
        return

    # Dynamically find the dTEC column
    dtec_col = [col for col in df_ipp.columns if 'dtec' in col][0]

    rx_lon_deg = np.degrees(rx_coords_tuple[3])
    rx_lat_deg = np.degrees(rx_coords_tuple[4])

    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        df_ipp['ipp_lon'], df_ipp['ipp_lat'], 
        c=df_ipp[dtec_col], 
        cmap='coolwarm', 
        s=10, 
        vmin=-0.5, vmax=0.5,
        alpha=0.8
    )

    plt.scatter(
        rx_lon_deg, rx_lat_deg, 
        c='black', marker='^', s=150, edgecolor='white', label=f'Receiver ({station_name})'
    )

    plt.colorbar(scatter, label=f'{dtec_col} (TECU)')
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.title(f'dTEC Ionospheric Piercing Points - {station_name}')
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

# ==========================================
# 3. Execution Logic & CLI Parsing
# ==========================================
def parse_arguments():
    """Handles command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Calculate and plot IPP from TEC data.")
    
    parser.add_argument("--root", type=str, default="tec_pc/", help="Root directory of TEC data")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g., 2023)")
    parser.add_argument("--doy", type=int, required=True, help="Day of Year (e.g., 1)")
    parser.add_argument("--station", type=str, required=True, help="Station name (e.g., Desconhecida)")
    parser.add_argument("--sat_file", type=str, required=True, help="Path to satellite coordinates parquet")
    parser.add_argument("--height", type=float, default=350.0, help="Ionospheric shell height in km (default: 350.0)")
    parser.add_argument("--save", action="store_true", help="Save the plot as a PNG instead of displaying it")

    return parser.parse_args()


def main(root_dir, year, doy, height=350.0, save_fig=False):
    """Core workflow that can be called from CLI or IDE."""

    #Loading Files
    folder_ephemeris = cfg.ephemeris_parquet_dir(year, doy)
    folder_dtec = cfg.dtec_pc_dir(year, doy)
    folder_receivers = cfg.station_positions_dir()
    
    file_ephemeris = folder_ephemeris / f"ORBIT_{year}{doy:03d}.parquet"
    df_ephemeris = pd.read_parquet(file_ephemeris)
    
    ...
    
    
    
    
    




    # # 1. Load the dTEC data
    # print(f"Loading data for {station} on {year}/{doy:03d}...")
    # df_raw = load_tec_with_neighbors_v2(
    #     root=root_path, 
    #     year=year, 
    #     doy=doy, 
    #     station=station
    # )

    # # 2. Interpolate satellites and compute IPP
    # df_processed = compute_station_ipp(
    #     df_dtec=df_raw, 
    #     sat_filepath=sat_file, 
    #     rx_coords_tuple=rx_coord_tuple, 
    #     H_km=height
    # )

    # # 3. Generate the Figure
    # save_path = f"{station}_ipp_map_{year}_{doy:03d}.png" if save_fig else None
    
    # plot_ipp_map(
    #     df_ipp=df_processed, 
    #     rx_coords_tuple=rx_coord_tuple, 
    #     station_name=station,
    #     save_path=save_path
    # )


if __name__ == "__main__":
    # ==============================================================
    # SPYDER / IDE MANUAL OVERRIDE
    # Set to True to run inside your IDE without terminal args
    # ==============================================================
    RUN_MANUALLY = True 
    
    if RUN_MANUALLY:
        print("Running in IDE Manual Mode...")
        main(
            root_dir="tec_pc/",
            year=2025,
            doy=316,
            height=350.0,
            save_fig=True 
        )
    else:
        print("Running in CLI Mode...")
        args = parse_arguments()
        main(
            root_dir=args.root,
            year=args.year,
            doy=args.doy,
            station=args.station,
            sat_file=args.sat_file,
            height=args.height,
            save_fig=args.save
        )