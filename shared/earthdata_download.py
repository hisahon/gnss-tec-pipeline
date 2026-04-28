# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:44:06 2026

@author: hisas


"""

import earthaccess
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from shared.config import ephemeris_dir

# Securely loads variables from the .env file
load_dotenv()

# Silencia os logs internos do earthaccess
logging.getLogger('earthaccess').setLevel(logging.CRITICAL)

def download_precise_orbits(target_date: datetime):
    """
    Downloads SP3 files for a 3-day window (target_date - 1, target_date, target_date + 1).
    Prioritizes FIN, falling back to RAP or ULT.
    Cleans up (deletes) lower-priority files if a higher-priority one is secured.
    """
    
    try:
        earthaccess.login(strategy="environment")
        print("Login to Earthdata successful.\n")
    except Exception as e:
        print("Authentication Error. Check if the .env file is configured correctly.")
        return

    dates_to_process = [
        target_date - timedelta(days=1),
        target_date,                     
        target_date + timedelta(days=1)  
    ]
    
    epoch_gps = datetime(1980, 1, 6)
    agencies_base = ["COD0MGX", "WUM0MGX", "SHA0MGX", "IAC0MGX"]
    orbit_priorities = ["FIN", "RAP", "ULT"] 
    
    for current_date in dates_to_process:
        gps_week = (current_date - epoch_gps).days // 7
        doy = current_date.strftime("%j")
        year = current_date.strftime("%Y")
        
        target_directory = ephemeris_dir(int(year), int(doy))
        os.makedirs(target_directory, exist_ok=True)
        
        print(f"==================================================")
        print(f"--- Processing Date: {current_date.strftime('%Y-%m-%d')} ---")
        print(f"GPS Week: {gps_week} | DOY: {doy} | Output Dir: {target_directory}")
        print(f"==================================================")
        
        day_success = False
        
        for base in agencies_base:
            agency_downloaded = False
            
            # O enumerate nos dá o índice (i) para sabermos em qual nível de prioridade estamos
            for i, orb_type in enumerate(orbit_priorities):
                prefix = f"{base}{orb_type}"
                file_name = f"{prefix}_{year}{doy}0000_01D_05M_ORB.SP3.gz"
                file_path = os.path.join(target_directory, file_name)
                
                file_secured = False
                
                if os.path.exists(file_path):
                    print(f"  [SKIPPED] Best available file already exists locally: {file_name}")
                    file_secured = True
                else:
                    url = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}/{file_name}"
                    print(f"  > Trying to download {prefix}...")
                    try:
                        earthaccess.download([url], local_path=target_directory)
                        print(f"  [OK] File successfully saved: {file_name}")
                        file_secured = True
                    except Exception:
                        pass # Falhou, o loop vai continuar para tentar a próxima prioridade
                
                # Se garantimos o arquivo (seja local ou via download), fazemos a limpeza
                if file_secured:
                    day_success = True
                    agency_downloaded = True
                    
                    # --- ROTINA DE LIMPEZA (CLEANUP) ---
                    # Pega apenas os tipos de órbita com prioridade MENOR que a atual
                    lower_priorities = orbit_priorities[i+1:] 
                    for lower_orb_type in lower_priorities:
                        lower_file_name = f"{base}{lower_orb_type}_{year}{doy}0000_01D_05M_ORB.SP3.gz"
                        lower_file_path = os.path.join(target_directory, lower_file_name)
                        
                        if os.path.exists(lower_file_path):
                            try:
                                os.remove(lower_file_path)
                                print(f"  [CLEANUP] Deleted outdated lower-priority file: {lower_file_name}")
                            except Exception as e:
                                print(f"  [WARNING] Could not delete {lower_file_name}: {e}")
                    
                    break # Sai do loop de prioridades, pois já temos a melhor versão desta agência
            
            if not agency_downloaded:
                 print(f"  [X] {base} unavailable in any format (FIN, RAP, ULT).")

        if not day_success:
            print(f"  [CRITICAL] None of the agencies had data for {current_date.strftime('%Y-%m-%d')}.")
        print("\n")
        
    print("Finished checking, downloading, and cleaning up data for the 3-day window.")