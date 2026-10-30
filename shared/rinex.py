# -*- coding: utf-8 -*-
"""Leitura e decodificação de arquivos RINEX (v2 e v3).

Autor original: Honda, R. H. (Set/2022).
"""
import glob
import gzip
import os
import pickle
import re
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation


def geocentric2geodetic(geocentric_array):
    '''
    https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    https://docs.astropy.org/en/stable/api/astropy.coordinates.EarthLocation.html#astropy.coordinates.EarthLocation.from_geodetic
    
    Parameters:
    lonLongitude or float
    Earth East longitude. Can be anything that initialises an Angle object (if float, in degrees).

    latLatitude or float
    Earth latitude. Can be anything that initialises an Latitude object (if float, in degrees).

    heightQuantity [:ref: ‘length’] or float, optional
    Height above reference ellipsoid (if float, in meters; default: 0).

    ellipsoidstr, optional
    Name of the reference ellipsoid to use (default: ‘WGS84’). Available ellipsoids are: ‘WGS84’, ‘GRS80’, ‘WGS72’.
    '''
    
    sta_ellipsoidal  = EarthLocation.from_geocentric(*geocentric_array, unit='m')
    φ = sta_ellipsoidal.geodetic[0].value  # the ‘WGS84’ lat-value
    λ = sta_ellipsoidal.geodetic[1].value  # the ‘WGS84’ lon-value
    h = sta_ellipsoidal.geodetic[2].value  # the ‘WGS84’ height-value
    
    return [φ,λ,h]


def system_names(key):
    
    gnss_dict = {
    'G': 'GPS',
    'R': 'GLONASS',
    'S': 'SBAS payload',
    'E': 'Galileo',
    'C': 'BeiDou',
    'J': 'QZSS',
    'I': 'IRNSS'
    }
    
    return gnss_dict.get(key)

def read_rinex(fileName):
    with open(fileName, "r", encoding="utf-8", errors="replace") as rinFile:
        for line in rinFile:
            if line[60:].strip() == 'RINEX VERSION / TYPE':
                version = float(line[:9].rstrip())
                break
                
    
    if int(version) == 2:
        dic_rinex=read_v2(fileName)
    if int(version) == 3:
        dic_rinex=read_v3(fileName)
        
    return dic_rinex
    




def read_header(rinFile, fileName):
    
    #Station Name for the header:
    station_name = os.path.basename(fileName)[0:4]
    
    # # Open the RINEX file for reading
    # rinFile = open(fileName, "r")
    
    # Read the first line of the file
    currentLine = rinFile.readline()
    
    # Initialize lists to store GNSS information
    gnss = []       # GNSS system identifiers (e.g., G, R, S)
    gnss_nobs = []  # Number of observed types for each GNSS system
    obs_type = []   # List to store observed types
    obs_v2 = [] #List to store observed types rinex v2
    # interval = np.nan  # Variable to store the observation interval (initially set to NaN)
    
    # Loop until 'END OF HEADER' is encountered
    while currentLine[60:].rstrip() != 'END OF HEADER':
        
        # Extract RINEX version if the line contains 'RINEX VERSION / TYPE'
        if currentLine[60:].rstrip() == 'RINEX VERSION / TYPE':
            rinex_version = float(currentLine[:9].rstrip())
            
        # Extract approximate position if the line contains 'APPROX POSITION XYZ'
        if currentLine[60:].rstrip() == 'APPROX POSITION XYZ':
            sta_xyz = currentLine[:59].rstrip().split()
            sta_xyz = [float(coord) for coord in sta_xyz]
            
        # Extract GNSS system and number of observed types if the line contains 'SYS / # / OBS TYPES'
        if currentLine[60:].rstrip() == 'SYS / # / OBS TYPES':
            if currentLine[:1].rstrip().isalpha():
                gnss += currentLine[:1].rstrip()
                gnss_nobs += [int(currentLine[3:6])]
                            
            # Extract observed types from the line
            obs_type += currentLine[7:60].split()
        
        #This line is for rinex v2
        if currentLine[60:].rstrip() == '# / TYPES OF OBSERV':
            obs_v2 += currentLine[10:60].split()
        
        # # Extract observation interval if the line contains 'INTERVAL'
        # if currentLine[60:].rstrip() == 'INTERVAL':
        #     interval = float(currentLine[:60])
            
        # Read the next line
        currentLine = rinFile.readline()
    
    # Create a dictionary to store observed types for each GNSS system
    gnss_obs = {}
    
    # Populate the dictionary with observed types
    for i, cons in enumerate(gnss):
        if i == 0:
            gnss_obs[cons] = obs_type[0:gnss_nobs[0]]
        if i > 0:
            gnss_obs[cons] = obs_type[sum(gnss_nobs[:i]):sum(gnss_nobs[:i+1])]

    # # Close the RINEX file
    # rinFile.close()
    
    sta_xyz = np.array(sta_xyz)
    sta_ellip=geocentric2geodetic(sta_xyz) #[lat,lon,height]
    
    if int(rinex_version) == 2:
        observables = obs_v2
        
    if int(rinex_version) == 3:
        observables = gnss_obs
    
    
    header_inf = {
    'rinex_version': rinex_version,
    'station_name': station_name,
    'xyz_geocentric': sta_xyz/1000,
    'φλh_geodetic': np.array(sta_ellip),
    'gnss': gnss,
    'gnss_obs': observables,
    'gnss_nobs': np.array(gnss_nobs)
    }

    
    # Return the extracted information
    # return rinex_version, sta_xyz/1000, sta_ellip, gnss, gnss_obs, np.array(gnss_nobs)#, int(interval)
    return header_inf

def read_line2data(input_string, nobs, date_time):
    # Set the size of each column in the input string
    column_size = 16
    
    # Extract observation data from the input string (excluding the first 4 characters)
    string_obs = input_string[4:]
    
    # Extract the PRN (satellite identifier) from the first 3 characters of the input string
    prn_name = re.sub(r'[^a-zA-Z0-9 ]', '', input_string[:3])
    
    # Split the observation data into columns based on the specified column size
    columns = [string_obs[k:k+column_size] for k in range(0, column_size*nobs, column_size)]
    
    # Remove the last 3 characters from each entry in the columns
    columns = [column[:-3] for column in columns]
    
    # # Convert the columns to float, replacing empty entries with NaN
    # columns = [float(column) if column.strip() else np.nan for column in columns]
    
    # Append the PRN and date_time to the columns
    columns.append(prn_name)
    columns.append(date_time)
    
    # Return the columns as a NumPy array wrapped in another array
    return np.array([columns])


def dataframe_treatment(obsGNSS_list, gnss, gnss_obs):
    obsGNSS = np.vstack(obsGNSS_list)
    obsGNSSdf = pd.DataFrame(obsGNSS)
    columns_to_numeric = obsGNSSdf.columns[:-2]
    obsGNSSdf[columns_to_numeric] = obsGNSSdf[columns_to_numeric].apply(pd.to_numeric, errors='coerce')
    
    max_obs = max(len(obs) for obs in gnss_obs.values())
    gnss_obs_padded = {key: obs + ['empty'] * (max_obs - len(obs)) for key, obs in gnss_obs.items()}
    column_index_gnss = obsGNSSdf.columns[-2]
    
    df_GNSS_dic = {}
    for constelation in gnss:
        
        columns_new=gnss_obs_padded[constelation]
        columns_new.append('prn')
        columns_new.append('horario')
        
        constelation_filtered = obsGNSSdf[obsGNSSdf.iloc[:, column_index_gnss].str.startswith(constelation)].copy()
        constelation_filtered.columns = columns_new
        constelation_filtered = constelation_filtered.drop(columns=[col for col in constelation_filtered.columns if col == 'empty'])
        constelation_filtered = constelation_filtered.sort_values(by='horario').reset_index(drop=True)
       
        if len(constelation_filtered) != 0:
            df_GNSS_dic[constelation] = constelation_filtered
    
    return df_GNSS_dic

def read_v3(fileName):
    rinFile = open(fileName, "r")
    currentLine = rinFile.readline()
    
    header_info = read_header(fileName)
    gnss, gnss_obs, gnss_nobs = header_info['gnss'], header_info['gnss_obs'], header_info['gnss_nobs']
    
    # _, _, _, gnss, gnss_obs, gnss_nobs = read_header(fileName)

    while currentLine[60:].rstrip() != 'END OF HEADER':
        currentLine = rinFile.readline()
    
    # Initialize variables
    obsGNSS_list = []  # Use a list to store rows before converting to NumPy array

    # Iterate through the remaining lines
    for currentLine in rinFile:
        if currentLine[:1].rstrip() == '>':
            ti_int = [int(float(ts)) for ts in currentLine[2:29].rstrip().split()]
            date_time = datetime(*ti_int)
            # print(date_time)
        elif currentLine[:1].rstrip().isalpha():
            columns = read_line2data(currentLine, gnss_nobs.max(), date_time)
            obsGNSS_list.append(columns)
    
        
    df_GNSS_dic = dataframe_treatment(obsGNSS_list, gnss, gnss_obs)
    
    df_GNSS_dic['header'] = header_info

    return df_GNSS_dic


def rinex_v3folder(rinex_folder, save_folder, save_format='pkl'):

    files = glob.glob(f'{rinex_folder}/*.rnx')
    sorted_files = sorted(files)
    
    for rinex_file in sorted_files:
        
        dic_GNSS=read_v3(rinex_file)
        
        f_name= rinex_file.split('\\')[-1]
        f_name_b=f_name.split('.')[0]
        
        if save_format == 'gz':
            with gzip.open(f"{save_folder}/{f_name_b}.gz", 'wb') as file:
                pickle.dump(dic_GNSS, file)
               
        elif save_format == 'pkl':
            with open(f"{save_folder}/{f_name_b}.pkl", 'wb') as file:
                pickle.dump(dic_GNSS, file)
    
        elif save_format == 'csv':
            for key, df in dic_GNSS.items():
                df.to_csv(f'{save_folder}/{f_name_b}_{key}.csv', index=False)


    

# Assuming merge_columns function is defined

def merge_columns(df, columns_to_merge, new_column_name='MergedColumn'):
    """
    Merge specified columns in a DataFrame element-wise and add a new column.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - columns_to_merge (list): List of column names to merge.
    - new_column_name (str): Name of the new column to be added.

    Returns:
    - pd.DataFrame: The DataFrame with the new merged column.
    """
    # Use functools.reduce and combine_first to merge columns element-wise
    merged_column = reduce(lambda x, y: x.combine_first(y), [df[col] for col in columns_to_merge])

    # Add the merged column to the original DataFrame using .loc
    df.loc[:, new_column_name] = merged_column

    return df

def merge_signals(df, new_column_name='MergedColumn'):
    # Separate columns with unique first two characters
    # unique_prefixes = df.columns.str[:2].unique()
    unique_prefixes = [prefix for prefix in df.columns.str[:2].unique() if prefix not in ['pr', 'ho']]

    # Apply merge_columns to each group
    for prefix in unique_prefixes:
        columns_to_merge = df.columns[df.columns.str.startswith(prefix)].tolist()
        merged_column = merge_columns(df[columns_to_merge].copy(), columns_to_merge, new_column_name=new_column_name)
        # df[prefix + 'A'] = merged_column[new_column_name]
        df.insert(0, prefix + 'a', merged_column[new_column_name])
        
        df=df.reindex(sorted(df.columns)[:-2] + ['prn', 'horario'], axis=1)

    return df
    
    
#%%
def read_v2(fileName):
    
    
    with open(fileName, "r", encoding="utf-8", errors="replace") as rinFile:
        header_info = read_header(rinFile, fileName)
       
    
        # for currentLine in rinFile:
        #     if currentLine[60:].strip() == 'END OF HEADER':
        #         currentLine = rinFile.readline()
        #         break
        
        currentLine = rinFile.readline()
        obss = header_info['gnss_obs']
        numProd = len(obss)
    
                
        arr=[] #-Aqui e aonde sera armazenado os valores dos produtos 
                
        cccc=0
        break_all=False
        while currentLine != '':
    #%%
            #Removendo um bloco de texto especifico de uns arquivos rinex
            if (currentLine[26:29]=='  3') or (currentLine[26:29]=='  4') or (currentLine[26:29]=='  5'):
                currentLine = rinFile.readline()
                cccc+=1
     
                while ((currentLine[26:29] != '  0') or (currentLine[60:67] == 'COMMENT')) and (currentLine != ''):
                    # print(cccc, currentLine)
                    currentLine = rinFile.readline()
                    cccc+=1
                    
                    
                    if currentLine == '':
                        break_all = True
            #caso os comentarios estiverem no ultimo bloco, eu preciso de outro comando pra parar o while
            if break_all:
                break
    #%%       
            try:
                numSats = int(currentLine[30:32])
            
            
            
                if currentLine[32:33].isalpha() == True and numSats == 0:
                    currentLine = rinFile.readline()
                    cccc+=1
                
                
                if currentLine[32:33].isalpha() == True and numSats <= 11 and numSats > 0:
                    year = int(currentLine[1:3])
                    month = int(currentLine[4:6])
                    day = int(currentLine[7:9])
                    hour = int(currentLine[10:12])
                    mm = int(currentLine[13:15])
                    sec = float(currentLine[16:26])
        
                    # numSats = int(currentLine[30:32])
                    epochPRN = []
        
                    # if numSats < 12:
                    for i in range(numSats):
                        epochPRN.append(currentLine[32+i*3:35+i*3])
                    currentLine = rinFile.readline()
                    cccc+=1
                        
                if currentLine[32:33].isalpha() == True and numSats > 11:
                    year = int(currentLine[1:3])
                    month = int(currentLine[4:6])
                    day = int(currentLine[7:9])
                    hour = int(currentLine[10:12])
                    mm = int(currentLine[13:15])
                    sec = float(currentLine[16:26])
        
                    # numSats = int(currentLine[30:32])
                    epochPRN = []
        
                    if numSats <= 12:
                        for i in range(numSats):
                            epochPRN.append(currentLine[32+i*3:35+i*3])
                        currentLine = rinFile.readline()
                        cccc+=1
                        
                    if numSats > 12:
                        for j in range(int(numSats/12)):
                            for i in range(12):
                                epochPRN.append(currentLine[32+i*3:35+i*3])
                            currentLine = rinFile.readline()
                            cccc+=1
                            
                        if numSats % 12 != 0:
                            for i in range(numSats % 12):
                                epochPRN.append(currentLine[32+i*3:35+i*3])
                            currentLine = rinFile.readline()
                            cccc+=1
                        
        
                if numSats != 0:
                    time=datetime(year+2000,month,day,hour,mm,int(sec))
        
                #-Fazendo a leitura dos produtos
                if numSats == 0:
                    currentLine = rinFile.readline()
                    cccc+=1
        
        
                if numSats != 0:
                    for m, prn in enumerate(epochPRN):
                #         for k in range(numSats):
                        row=[]
                        row.append(time)
                        row.append(prn)
                        if numProd <=5:
                            for i in range(numProd):
                                row.append(currentLine[i*16+1:i*16+14].lstrip())
                            arr.append(row)
                            currentLine = rinFile.readline()
                            cccc+=1
                            
                        if numProd > 5:
                            
                            for k in range(int(numProd / 5)):
                                for i in range(5):
                                    row.append(currentLine[i*16+1:i*16+14].lstrip())
                                currentLine = rinFile.readline()
                                cccc+=1
                                
                            if (numProd % 5) != 0:
                                for i in range(numProd % 5):
                                    row.append(currentLine[i*16+1:i*16+14].lstrip())
                                currentLine = rinFile.readline()
                                cccc+=1
                            arr.append(row)
            except:
                print('#{}# Linha {} File {}'.format(currentLine,cccc,fileName))
                # break

    cname = ['horario', 'prn'] + obss
    #-Salvando os arquivos em .csv
    arr_df=pd.DataFrame(arr, columns=cname)
    
    
    arr_df[obss] = arr_df[obss].apply(pd.to_numeric, errors='coerce')
    
    dic_rinex = df2dic(arr_df)
    dic_rinex['header'] = header_info
    
    return dic_rinex  
    
    
def df2dic(df):
    # Set to store unique letters
    prns = df['prn'].unique()
    
    gnsss = set()

    # Iterate over each string in the array
    for string in prns:
        # Extract letters from each string and add them to the set
        gnsss.update(filter(str.isalpha, string))

    gnsss= list(gnsss)
    
    dic_rinex = {}
    
    for gnss in gnsss:
        dic_rinex[gnss] = df[df['prn'].str.startswith(gnss)]
        
    return dic_rinex
    
    
    
def to_sec(pd_df):
    
    import numpy as np
    from datetime import datetime

    c_time_ref = datetime(1970,1,1)
    n_days=pd_df['horario']-c_time_ref
    
    pd_df['horario_s']=n_days.astype('timedelta64[s]').astype(np.float64)
    
    # c_names = 'horario horario_s prn L1 iL1 C1 iC1 L2 iL2 P2 iP2'.split()
    c_names=pd_df.columns.values
    c_names=np.insert(c_names, 1, 'horario_s')
    c_names=np.delete(c_names, -1)
    
    
    pd_df=pd_df[c_names]
    
    return pd_df
    
    # print("Unique letters:", unique_letters)
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Universal GNSS Ephemeris Reader
Supports GPS, Galileo, BeiDou, GLONASS (RINEX v2 and v3 navigation files)
"""

# WGS84 / GLONASS datum constants
# -------------------------------
a = 6378137.00  # semi-major axis [meters]
b = 6356752.314245  # semi-minor axis [meters]
e = np.sqrt(1 - (b ** 2) / (a ** 2))
r_e = 6371e3  # Earth mean radius in meters

mu = 398600.44e9  # Earth's gravitational constant (m^3/s^2)
ae = 6378136       # GLONASS reference equatorial radius (m)
J0sqd = 1082625.7e-9  # 2nd zonal harmonic for GLONASS
C20 = -1082.63e-6     # Earth gravity field coefficient
omega = 7.2921151467e-5  # Earth's rotation rate (rad/s)


# Use this mu and omega for Keplerian model
MU = mu
OMEGA_E = omega


# Parse a RINEX navigation file (v2 or v3) into ephemeris dictionaries
def parse_nav_file(file_path):
    """
    Parse RINEX .nav file (v2 or v3) into a list of ephemeris dictionaries.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for i, line in enumerate(lines):
        if 'END OF HEADER' in line:
            break
    data_lines = lines[i+1:]

    ephemerides = []
    block = []
    for line in data_lines:
        if re.match(r'^[A-Z]\d{2}|^\d{2}', line[:3]):
            if block:
                eph = parse_ephemeris_block(block)
                ephemerides.append(eph)
                block = []
        block.append(line)
    if block:
        eph = parse_ephemeris_block(block)
        ephemerides.append(eph)

    return ephemerides


# Convert a RINEX ephemeris text block into a structured dictionary
def parse_ephemeris_block(block):
    """Parse a single ephemeris block into a dictionary."""
    prn_line = block[0]
    system = prn_line[0]
    prn = prn_line[:3].strip()
    
    # Parse epoch
    fields = prn_line.split()
    epoch = datetime(int(fields[1]), int(fields[2]), int(fields[3]),
                     int(fields[4]), int(fields[5]), int(float(fields[6])))

    # Parse ephemeris values (8 lines total)
    vals = []
    for line in block:
        vals += [float(line[i:i+19].replace('D', 'E')) for i in range(3, 80, 19)]

    eph = {
        'prn': prn,
        'system': system,
        'epoch': epoch,
        'vals': vals
    }
    return eph


# Solve Kepler's equation: M = E - e*sin(E) for E given M and eccentricity e
def kepler_E(M, e, tol=1e-12):
    """Solve Kepler's equation for eccentric anomaly."""
    E = M
    for _ in range(10):
        E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        if abs(E - E_new) < tol:
            return E_new
        E = E_new
    return E


# Compute satellite ECEF position (x, y, z) using ephemeris data at GPS time
def compute_satellite_xyz(ephemeris_data, gps_time_sec):
    """
    Compute satellite ECEF position from ephemeris at GPS time t (seconds).
    Automatically dispatches between Keplerian (G, E, C, J) and GLONASS (R).
    """
    if eph['system'] == 'R':
        return compute_glonass_xyz(eph, t)
    else:
        params = ephemeris_data['vals']
        tk = gps_time_sec - params[11]
        if tk > 302400: tk -= 604800
        if tk < -302400: tk += 604800

        A = params[10]**2
        n0 = np.sqrt(MU / A**3)
        n = n0 + params[5]
        M = params[6] + n * tk
        E = kepler_E(M, params[8])

        nu = np.arctan2(np.sqrt(1 - params[8]**2) * np.sin(E), np.cos(E) - params[8])
        u = nu + params[17] + params[9] * np.sin(2*nu) + params[7] * np.cos(2*nu)

        r = A * (1 - params[8] * np.cos(E)) + params[4] * np.sin(2*u) + params[16] * np.cos(2*u)
        i = params[12] + params[19] * tk + params[14] * np.sin(2*u) + params[13] * np.cos(2*u)

        x_orb = r * np.cos(u)
        y_orb = r * np.sin(u)

        Omega = params[10] + (params[18] - OMEGA_E) * tk - OMEGA_E * params[11]

        x = x_orb * np.cos(Omega) - y_orb * np.cos(i) * np.sin(Omega)
        y = x_orb * np.sin(Omega) + y_orb * np.cos(i) * np.cos(Omega)
        z = y_orb * np.sin(i)

        return np.array([x, y, z])


# Compute GLONASS satellite position using third-order integration (per ICD-GLONASS)
def compute_glonass_xyz(ephemeris_data, gps_time_sec):
    """
    Compute GLONASS satellite position using third-order integration (ICD-GLONASS based).
    """
    v = eph['vals']
    x0, vx, ax = v[0]*1000, v[1]*1000, v[2]*1000
    y0, vy, ay = v[4]*1000, v[5]*1000, v[6]*1000
    z0, vz, az = v[8]*1000, v[9]*1000, v[10]*1000

    dt = gps_time_sec - ephemeris_data['epoch'].timestamp()
    if dt == 0:
        return np.array([x0, y0, z0])

    radius = np.sqrt(x0**2 + y0**2 + z0**2)

    aJ2 = 1.5 * J0sqd * mu * ae**2 / r**5
    aG = -mu / r**3

    fx = lambda x, z: aG * x - aJ2 * x * (1 - 5 * z**2 / r**2) + omega**2 * x + 2 * omega * vy + ax
    fy = lambda y, z: aG * y - aJ2 * y * (1 - 5 * z**2 / r**2) + omega**2 * y - 2 * omega * vx + ay
    fz = lambda z:   aG * z - aJ2 * z * (3 - 5 * z**2 / r**2) + az

    kx1 = fx(x0, z0) * dt
    ky1 = fy(y0, z0) * dt
    kz1 = fz(z0) * dt

    x1 = x0 + vx * dt/2 + kx1 * dt/8
    y1 = y0 + vy * dt/2 + ky1 * dt/8
    z1 = z0 + vz * dt/2 + kz1 * dt/8

    kx2 = fx(x1, z1) * dt
    ky2 = fy(y1, z1) * dt
    kz2 = fz(z1) * dt

    x2 = x0 + vx * dt/2 + kx2 * dt/8
    y2 = y0 + vy * dt/2 + ky2 * dt/8
    z2 = z0 + vz * dt/2 + kz2 * dt/8

    kx3 = fx(x2, z2) * dt
    ky3 = fy(y2, z2) * dt
    kz3 = fz(z2) * dt

    x = x0 + vx * dt + (kx1 + kx2 + kx3) * dt / 6
    y = y0 + vy * dt + (ky1 + ky2 + ky3) * dt / 6
    z = z0 + vz * dt + (kz1 + kz2 + kz3) * dt / 6

    return np.array([x, y, z])

    
    
# import pandas as pd
# from datetime import datetime

def read_v2_from_filelike(rinFile):
    """
    Parses a RINEX v2 observation file from a file-like object (StringIO or open file).
    Returns a dictionary with data and header.
    """
    # Ensure pointer at beginning
    rinFile.seek(0)
    lines = rinFile.readlines()
    
    # Read header using in-memory lines
    header_info = read_header_from_lines(lines)

    # Re-initialize the file pointer after header reading
    rinFile.seek(0)
    for currentLine in rinFile:
        if currentLine[60:].strip() == 'END OF HEADER':
            currentLine = rinFile.readline()
            break

    obss = header_info['gnss_obs']
    numProd = len(obss)

    arr = []
    cccc = 0
    break_all = False

    while currentLine != '':
        if (currentLine[26:29] in ['  3', '  4', '  5']):
            currentLine = rinFile.readline()
            cccc += 1
            while ((currentLine[26:29] != '  0') or (currentLine[60:67] == 'COMMENT')) and (currentLine != ''):
                currentLine = rinFile.readline()
                cccc += 1
                if currentLine == '':
                    break_all = True
        if break_all:
            break

        try:
            numSats = int(currentLine[30:32])
            if currentLine[32:33].isalpha() and numSats == 0:
                currentLine = rinFile.readline()
                cccc += 1

            if currentLine[32:33].isalpha() and 0 < numSats <= 11:
                year = int(currentLine[1:3])
                month = int(currentLine[4:6])
                day = int(currentLine[7:9])
                hour = int(currentLine[10:12])
                mm = int(currentLine[13:15])
                sec = float(currentLine[16:26])
                epochPRN = [currentLine[32+i*3:35+i*3] for i in range(numSats)]
                currentLine = rinFile.readline()
                cccc += 1

            if currentLine[32:33].isalpha() and numSats > 11:
                year = int(currentLine[1:3])
                month = int(currentLine[4:6])
                day = int(currentLine[7:9])
                hour = int(currentLine[10:12])
                mm = int(currentLine[13:15])
                sec = float(currentLine[16:26])
                epochPRN = []

                for j in range(numSats // 12):
                    for i in range(12):
                        epochPRN.append(currentLine[32+i*3:35+i*3])
                    currentLine = rinFile.readline()
                    cccc += 1

                if numSats % 12 != 0:
                    for i in range(numSats % 12):
                        epochPRN.append(currentLine[32+i*3:35+i*3])
                    currentLine = rinFile.readline()
                    cccc += 1

            if numSats != 0:
                time = datetime(year + 2000, month, day, hour, mm, int(sec))

            if numSats == 0:
                currentLine = rinFile.readline()
                cccc += 1

            if numSats != 0:
                for prn in epochPRN:
                    row = [time, prn]
                    if numProd <= 5:
                        for i in range(numProd):
                            row.append(currentLine[i*16+1:i*16+14].lstrip())
                        arr.append(row)
                        currentLine = rinFile.readline()
                        cccc += 1
                    else:
                        for k in range(numProd // 5):
                            for i in range(5):
                                row.append(currentLine[i*16+1:i*16+14].lstrip())
                            currentLine = rinFile.readline()
                            cccc += 1
                        if numProd % 5 != 0:
                            for i in range(numProd % 5):
                                row.append(currentLine[i*16+1:i*16+14].lstrip())
                            currentLine = rinFile.readline()
                            cccc += 1
                        arr.append(row)

        except Exception as e:
            print(f'# {currentLine} # Linha {cccc} | Error: {e}')

    cname = ['horario', 'prn'] + obss
    arr_df = pd.DataFrame(arr, columns=cname)
    arr_df[obss] = arr_df[obss].apply(pd.to_numeric, errors='coerce')

    dic_rinex = df2dic(arr_df)
    dic_rinex['header'] = header_info

    return dic_rinex

    
# import numpy as np

def read_header_from_lines(lines):
    """
    Parse RINEX header from a list of lines already in memory.
    Returns a dictionary with header information.
    """
    # Extract station name from the first line (positions 0–4)
    station_name = lines[0][:4].strip()

    gnss = []
    gnss_nobs = []
    obs_type = []
    obs_v2 = []
    interval = np.nan
    sta_xyz = None

    for currentLine in lines:
        label = currentLine[60:].rstrip()

        if label == 'RINEX VERSION / TYPE':
            rinex_version = float(currentLine[:9].rstrip())

        elif label == 'APPROX POSITION XYZ':
            sta_xyz = currentLine[:60].split()
            sta_xyz = [float(coord) for coord in sta_xyz]

        elif label == 'SYS / # / OBS TYPES':
            if currentLine[0].isalpha():
                gnss += currentLine[0]
                gnss_nobs += [int(currentLine[3:6])]
            obs_type += currentLine[7:60].split()

        elif label == '# / TYPES OF OBSERV':
            obs_v2 += currentLine[10:60].split()

        elif label == 'END OF HEADER':
            break

    if sta_xyz is None:
        raise ValueError("APPROX POSITION XYZ not found in header.")

    gnss_obs = {}
    for i, cons in enumerate(gnss):
        start = sum(gnss_nobs[:i])
        end = start + gnss_nobs[i]
        gnss_obs[cons] = obs_type[start:end]

    sta_xyz = np.array(sta_xyz)
    sta_ellip = geocentric2geodetic(sta_xyz)  # [lat, lon, height]

    if int(rinex_version) == 2:
        observables = obs_v2
    elif int(rinex_version) == 3:
        observables = gnss_obs
    else:
        raise ValueError(f"Unsupported RINEX version: {rinex_version}")

    header_inf = {
        'rinex_version': rinex_version,
        'station_name': station_name,
        'xyz_geocentric': sta_xyz / 1000,
        'φλh_geodetic': np.array(sta_ellip),
        'gnss': gnss,
        'gnss_obs': observables,
        'gnss_nobs': np.array(gnss_nobs)
    }

    return header_inf

    
    
    
    
    
    
            
