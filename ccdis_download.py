# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:44:06 2026

@author: hisas
"""

import earthaccess
import os
from datetime import datetime
from dotenv import load_dotenv

from shared.config import ephemeris_dir

# Securely loads variables from the .env file
load_dotenv()

def download_precise_orbits(target_date: datetime):
    """
    Attempts to download Final SP3 files in priority order: COD, WUM, SHA, IAC.
    target_date: datetime object
    """
    # 1. GNSS Time Calculations
    epoch_gps = datetime(1980, 1, 6)
    gps_week = (target_date - epoch_gps).days // 7
    doy = target_date.strftime("%j")
    year = target_date.strftime("%Y")
    
    # 2. Directory Setup (2-digit year format: YY-MM-DD)
    target_directory = ephemeris_dir(int(year), int(doy))
    os.makedirs(target_directory, exist_ok=True)
    
    # 3. Restricted Agency List (Priority: COD, followed by the 3 alternatives)
    agencies = ["COD0MGXFIN", "WUM0MGXFIN", "SHA0MGXFIN", "IAC0MGXFIN"]
    
    print(f"--- Processing: {target_date.strftime('%Y-%m-%d')} ---")
    print(f"GPS Week: {gps_week} | DOY: {doy}")
    
    # 4. Secure Login
    # earthaccess will automatically read EARTHDATA_USERNAME and EARTHDATA_PASSWORD from your environment
    try:
        earthaccess.login(strategy="environment")
    except Exception as e:
        print("Authentication Error. Check if the .env file is configured correctly.")
        return
    
    success = False
    
    # 5. Download Loop
    for prefix in agencies:
        file_name = f"{prefix}_{year}{doy}0000_01D_05M_ORB.SP3.gz"
        url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}/{file_name}"
        
        print(f"Trying {prefix}...")
        
        try:
            earthaccess.download([url], local_path=target_directory)
            print(f"OK: File successfully saved in ./{target_directory}/{file_name}")
            success = True
            break # Exits the loop as soon as the first one is downloaded
        except Exception:
            print(f"[{prefix}] Unavailable.")
            continue

    if not success:
        print(f"CRITICAL: None of the files (COD, WUM, SHA, IAC) were found for this date.")

# Example usage integrating with datetime:
# desired_date = datetime(2021, 7, 11)
# download_precise_orbits(desired_date)