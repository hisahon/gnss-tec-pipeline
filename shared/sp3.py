# -*- coding: utf-8 -*-
"""Leitura de efemérides SP3, interpolação de órbitas e cálculo de IPP.

Autor original: Honda, R. H. (Set/2022).
"""
import gzip
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from scipy.interpolate import interp1d, PchipInterpolator

def read_sp3(file_path):

    fileName=file_path
    
    if file_path.split('.')[-1] == 'gz':
        rinFile = gzip.open(file_path, 'rt')
    else:
        rinFile = open(fileName, "r")
    
        
    currentLine = rinFile.readline()
        
    #------------------------------------------------------
    #Lendo o cabeçalho dos arquivos sp3
    #------------------------------------------------------
    while currentLine[0:1].rstrip() != '*':
        currentLine = rinFile.readline()
    
    #------------------------------------------------------
    #Lendo o corpo dos arquivos sp3
    #------------------------------------------------------ 
    row=[]    
    # while currentLine[0:4].rstrip() != 'EOF':
    while 'EOF' not in currentLine:
        if currentLine[0:1].rstrip() == '*':
            #extraindo o tempo dos arquivos
            #--------------------------------------------
            t=currentLine[1:32].split()
            t=[round(float(x)) for x in t]
            t=datetime(t[0],t[1],t[2],t[3],t[4],t[5])
    #         t=datetime.time(t[3],t[4],t[5])
            #--------------------------------------------
            
            #Chamando a proxima linha
            currentLine = rinFile.readline()
        else:
            cow=[t,currentLine[1:4],
                currentLine[5:18],
                currentLine[19:32],
                currentLine[33:46],
                currentLine[47:60]]
                # currentLine[62:63],
                # currentLine[65:66],
                # currentLine[68:69],
                # currentLine[70:73]]
    
            # Adicionando uma coluna de tempo na ultima posição
            row.append(cow)
            
           
            #Chamando a proxima linha
            currentLine = rinFile.readline()
    
    df_positions=pd.DataFrame(row)
    c_names = 'horario prn x_km y_km z_km clock_musec'.split()
    
    df_positions=df_positions.set_axis(c_names, axis='columns')
    # pd.to_datetime(df_positions['horario'])
    
    
    
    
    # #extraindo o a data da leitura
    # ano=df_positions['horario'].dt.year
    # mes=df_positions['horario'].dt.month
    # dia=df_positions['horario'].dt.day
    # #out_data = str(ano[0])+'-'+str(mes[0])+'-'+str(dia[0])
    # out_data = datetime.date(ano[0],mes[0],dia[0])
    
    #Extraindo a informação dos pnrs contidos no arquivo
    prn_names=df_positions['prn'].unique()
    # prn_names
    
    cols = df_positions.columns.drop(['horario','prn'])
    # cols
    
    #convertendo as informações em numeros
    df_positions[cols] = df_positions[cols].apply(pd.to_numeric, errors='coerce')
    
    return times(df_positions), prn_names

def times(df):
    #reference time
    c_time_ref = datetime(1970,1,1)
    n_days=df['horario']-c_time_ref
    
    #Time in hours
    df['time_h'] = df['horario'].dt.hour + df['horario'].dt.minute/60 + df['horario'].dt.second/60/60
    #Time in julian-like
    df['time_j']=n_days.dt.total_seconds().astype(np.float64)
    #Time in seconds (epoch)
    df['time_e'] = (df['horario'] - df['horario'].dt.normalize()) // pd.Timedelta(seconds=1)
    
    return df


def data_interpol(df_orbit, prn_name):
    
    condition=(df_orbit[['x_km', 'y_km', 'z_km']] < 1e5).all(axis=1)
    df_orbit = df_orbit[condition]

    prn_data=df_orbit[df_orbit['prn']==prn_name].copy()
    
 
    prn_data = prn_data.drop_duplicates(subset=['time_j'])
        
    if not prn_data.empty:
    
        tt = prn_data['time_j']
        xx = prn_data['x_km']
        yy = prn_data['y_km']
        zz = prn_data['z_km']
       
        
        int_x = interp1d(tt,xx, kind='cubic')
        int_y = interp1d(tt,yy, kind='cubic')
        int_z = interp1d(tt,zz, kind='cubic')
        
        interpolation_output = [int_x, int_y, int_z]
        keep_this_prn = True
        
    else:
        
        interpolation_output = 'There isn''t {} in sp3 file'.format(prn_name)
        keep_this_prn = prn_name
    
    return interpolation_output, keep_this_prn
#Put this routine in the right place, doesn't make sense stay in a place only for rinex 3
def geocentric2geodetic(geocentric_array, unit):
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
    
    sta_ellipsoidal  = EarthLocation.from_geocentric(*geocentric_array, unit=unit)
    φ = sta_ellipsoidal.geodetic[0].value  # the ‘WGS84’ lon-value
    λ = sta_ellipsoidal.geodetic[1].value  # the ‘WGS84’ lat-value
    h = sta_ellipsoidal.geodetic[2].value  # the ‘WGS84’ height-value
    
    return [λ,φ,h]

# def wgs84_to_cartesian(longitude, latitude, altitude):
#     '''
#     https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
#     Semi-major axis (a): Approximately 6,378,137 meters.
#     Semi-minor axis (b): Approximately 6,356,752.314 meters

#     Convert WGS84 geodetic coordinates (longitude, latitude, altitude) to Cartesian coordinates (X, Y, Z).

#     Parameters:
#     - longitude (float): Longitude in degrees.
#     - latitude (float): Latitude in degrees.
#     - altitude (float): Altitude in kilometers.

#     Returns:
#     - X (float): Cartesian coordinate along the X-axis.
#     - Y (float): Cartesian coordinate along the Y-axis.
#     - Z (float): Cartesian coordinate along the Z-axis.
#     '''

#     # Convert altitude to meters
#     alt_km = altitude * 1000

#     # Constants for the WGS84 ellipsoid
#     a = 6378137.0  # Semi-major axis
#     b = 6356752.314245  # Semi-minor axis
#     e = math.sqrt((a**2 - b**2) / (a**2))  # Eccentricity

#     # Convert latitude and longitude to radians
#     lat_rad = math.radians(latitude)
#     lon_rad = math.radians(longitude)

#     # Calculate the radius of curvature in the prime vertical (N)
#     N = a / math.sqrt(1 - e**2 * math.sin(lat_rad) ** 2)

#     # Convert geodetic coordinates to Cartesian coordinates
#     X = (N + alt_km) * math.cos(lat_rad) * math.cos(lon_rad)
#     Y = (N + alt_km) * math.cos(lat_rad) * math.sin(lon_rad)
#     Z = (N * (1 - e**2) + alt_km) * math.sin(lat_rad)

#     return X/1000, Y/1000, Z/1000


def wgs84_to_XYZ(df, shellH):
    '''
    Convert WGS84 geodetic coordinates (longitude, latitude, altitude) to Cartesian coordinates (X, Y, Z).

    Parameters:
    - df (pd.DataFrame): Pandas DataFrame containing 'longitude', 'latitude', and 'altitude' columns.

    Returns:
    - df_out (pd.DataFrame): Pandas DataFrame containing 'X_km', 'Y_km', and 'Z_km' columns.
    '''

    # Convert altitude to meters
    alt_km = shellH * 1000

    # Constants for the WGS84 ellipsoid
    a = 6378137.0  # Semi-major axis in meters
    b = 6356752.314245  # Semi-minor axis in meters
    e = math.sqrt((a**2 - b**2) / (a**2))  # Eccentricity

    # Convert latitude and longitude to radians using np.radians()
    lat_rad = np.radians(df['latitude'])
    lon_rad = np.radians(df['longitude'])

    # Calculate the radius of curvature in the prime vertical (N)
    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad) ** 2)

    # # Convert geodetic coordinates to Cartesian coordinates
    # X = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    # Y = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    # Z = (N * (1 - e**2) + alt_km) * np.sin(lat_rad)
    
    # Convert geodetic coordinates to Cartesian coordinates
    df['X_km'] = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad) / 1000
    df['Y_km'] = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad) / 1000
    df['Z_km'] = ((N * (1 - e**2) + alt_km) * np.sin(lat_rad)) / 1000


    return df

def distance_from_a_point(df, point):
    
    x0=point[0]
    y0=point[1]
    z0=point[2]
    
    x = df['X_km']
    y = df['Y_km']
    z = df['Z_km']
    
    df['dr_km'] = np.sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
    
    return df
    


def coord_cartesian(inter_function, t_input):
    
    #Transforming into a matriz
    c_cart=zip(inter_function[0](t_input),
              inter_function[1](t_input),
              inter_function[2](t_input))
    
    #Interpolated cartesian coordinates
    xyz=pd.DataFrame(c_cart, columns = ['x_km','y_km','z_km'])
    
    return xyz



# def ipp_coordinates(Df_RINEX, df_orbit, sta_xyz, shell_h):
# # =============================================================================
# #  Calculo dos valores de TEC e vTEC
# # =============================================================================
#     # del_prn=[]
#     iall = Df_RINEX.index
#     # re=6371.   #raio da terra médio .
    
#     prns=Df_RINEX['prn'].unique()
#     prns.sort()
    
#     for prn in prns:
#         #Tem que ser por prn, cada prn tem uma trajetoria diferente
#         idx_prn = iall[Df_RINEX['prn']==prn].copy()
#         Inter_prn, keep_this_prn=data_interpol(df_orbit, prn)
        
#         try:
#             #Achando as Coortenadas do satelite
#             # T_input=Df_RINEX.loc[idx_prn,'horario_s'].copy()
#             T_input=Df_RINEX.loc[idx_prn,'time_j'].copy()
#             # prn_coord=sp3.coord_spherical(Inter_prn,T_input)
#             prn_xyz=coord_cartesian(Inter_prn,T_input)
#             # prn_coord=coord_elliptical(Inter_prn,T_input)
            
#             #Calculando o Ionospheric Piercing Point
#             lat, lon, ele = ipp_calculator(prn_xyz, sta_xyz, shell_h)
            

#             # Essa parte poderia estar dentro do ipp        
#             # =================================================================
#             Df_Geo=pd.DataFrame()
# #        
#             Df_Geo[''] = idx_prn
#             Df_Geo['lat'], Df_Geo['lon'], Df_Geo['ele'] = lat, lon, ele
#             Df_Geo.set_index('',inplace=True)
            
#             # #If its required to calculate the VTEC, uncomment this line
#             # Df_Geo['S_factor']=np.cos(np.arcsin(np.cos(np.radians(elev)) * re/(re+shell_h) ))
#             # Df_RINEX.loc[idx_prn,'S_factor']=Df_Geo.loc[idx_prn,'S_factor']

            
#             # Df_RINEX.loc[idx_prn,'rho_km']=Df_Geo.loc[idx_prn,'rho_km']
#             Df_RINEX.loc[idx_prn,f'lt{int(shell_h)}']=Df_Geo.loc[idx_prn,'lat']
#             Df_RINEX.loc[idx_prn,f'ln{int(shell_h)}']=Df_Geo.loc[idx_prn,'lon']
#             Df_RINEX.loc[idx_prn,f'el{int(shell_h)}']=Df_Geo.loc[idx_prn,'ele']
#             # =================================================================
#         except:
#             # bef = len(Df_RINEX)
#             iall = iall.drop(idx_prn)
#             Df_RINEX = Df_RINEX.drop(idx_prn)
#             # aft = len(Df_RINEX)
            
#             # print(f"The problem with PRN {prn} IPP resulted in the deletion of {bef-aft} entries.")

        

    
#         #Se o prn nao tiver dados, deleta ele de uma vez    
        
            
#             # del_prn.append(prn)    
    
#         # #Calculando os valores de vTEC
#         # # S_factor=np.cos(np.arcsin(np.cos(np.radians(elev))*re/(re+hm)))
#         # Df_RINEX.loc[idx_prn,'vTEC']=Df_RINEX.loc[idx_prn,'TEC']*Df_Geo.loc[idx_prn,'S_factor']
#         # # Df_RINEX.loc[idx_prn,'vTEC']=np.multiply(Df_RINEX['TEC'],Df_Geo['S_factor'])
        
        
#         # Df_RINEX.loc[idx_prn,'s_factor']=Df_Geo.loc[idx_prn,'S_factor']
    
#     return Df_RINEX


def satellite_coordinates(df_prn, df_orb_prn, prn, ref_date):

    orb_t = df_orb_prn['sec'].to_numpy()
    orb_x = df_orb_prn['x_km'].to_numpy()   
    orb_y = df_orb_prn['y_km'].to_numpy()   
    orb_z = df_orb_prn['z_km'].to_numpy()   
    

    prn_t = (df_prn["horario"] - ref_date).dt.total_seconds().astype(int).to_numpy()
    # prn_t = df_prn['sec'].to_numpy()
        
    fx = PchipInterpolator(orb_t, orb_x)
    fy = PchipInterpolator(orb_t, orb_y)
    fz = PchipInterpolator(orb_t, orb_z)
    
    x_i = fx(prn_t)
    y_i = fy(prn_t)
    z_i = fz(prn_t)
    
    df_prn['x_km'] = x_i
    df_prn['y_km'] = y_i
    df_prn['z_km'] = z_i
    
    return df_prn




def satellite_xyz(df_rinex, df_orbit):
    """
    Add satellite ECEF coordinates (x_km, y_km, z_km) to the GNSS observation DataFrame
    using interpolated SP3 orbit data.

    Parameters:
        df_rinex (pd.DataFrame): GNSS observation DataFrame with 'prn' and 'time_j' columns.
        df_orbit (pd.DataFrame): SP3 orbit DataFrame used for interpolation.

    Returns:
        pd.DataFrame: Updated DataFrame with satellite coordinates.
    """
    prns = sorted(df_rinex['prn'].unique())
    coord_cols = ['x_km', 'y_km', 'z_km']
    df_rinex[coord_cols] = None  # Initialize columns
    dropped = []

    for prn in prns:
        idxs = df_rinex['prn'] == prn
        try:
            Inter_prn, keep_info = data_interpol(df_orbit, prn)
            if isinstance(Inter_prn, str):
                # print(f"[SKIP] {prn}: {keep_info}")
                dropped.append(prn)
                df_rinex = df_rinex[~idxs]
                continue

            time_j = df_rinex.loc[idxs, 'time_j']
            prn_xyz = coord_cartesian(Inter_prn, time_j)

            df_rinex.loc[idxs, coord_cols] = prn_xyz.values

        except Exception as e:
            print(f"[ERROR] {prn}: {e}")
            dropped.append(prn)
            df_rinex = df_rinex[~idxs]

    # if dropped:
    #     print(f"[INFO] Dropped PRNs: {dropped} (total: {len(dropped)})")

    return df_rinex, dropped


def concat_daybyday(folder_path):
    
    
    if not os.path.exists(folder_path):
        sp3_day = pd.DataFrame()
    else:
        files = os.listdir(folder_path)
        files = [file for file in files if file.endswith('.gz')]
        for i, file_name in enumerate(files):
            
        # Construct the full file path

            file_path = os.path.join(folder_path, file_name)
            
            if i==0:
                sp3_day, prns = read_sp3(file_path)
            
            else:
                sp3_new, _ = read_sp3(file_path)
        
        
                for prn in prns:
                    sp3_new = sp3_new[sp3_new['prn'] != prn]
                    
                    
                sp3_day = pd.concat([sp3_day, sp3_new], ignore_index=True)
                
                prns=sp3_day['prn'].unique()
            
            print(f'Processed: {file_name}')
    
    return sp3_day



def concat_all(folder_base, datein):
    
    # one_day = timedelta(days=1)
    
    # dateout=datein - timedelta(days=1)
    # year, doy = dateout.year, dateout.strftime("%j")
    # folder_day = folder_base + f'{year}/{doy}/'
    # df_orbitday0 = concat_daybyday(folder_day)
    
    # dateout=datein
    # year, doy = dateout.year, dateout.strftime("%j")
    # folder_day = folder_base + f'{year}/{doy}/'
    # df_orbitday1 = concat_daybyday(folder_day)
    
    # sp3_all = pd.concat([df_orbitday0, df_orbitday1], ignore_index=True)
    
    # dateout=datein + timedelta(days=1)
    # year, doy = dateout.year, dateout.strftime("%j")
    # folder_day = folder_base + f'{year}/{doy}/'
    # df_orbitday2 = concat_daybyday(folder_day)
    
    # sp3_all = pd.concat([sp3_all, df_orbitday2], ignore_index=True)
    
    dfs = []

    # Loop over the range of days (from -1 day to +1 day)
    for i in range(-1, 3):
        dateout = datein + timedelta(days=i)
        year, doy = dateout.year, dateout.strftime("%j")
        folder_day = f'{folder_base}/{year}/{doy}/'
        dfs.append(concat_daybyday(folder_day))  # Append the dataframe to the list
    
    # Concatenate all the dataframes in the list
    sp3_all = pd.concat(dfs, ignore_index=True)
    
    # Sort DataFrame by 'horario_s' column in ascending order
    
    sp3_all = times(sp3_all)
    
    sp3_all.sort_values(by='time_j', inplace=True)
    
    # Reset the index
    sp3_all.reset_index(drop=True, inplace=True)
    
    # dateout=datein
    # year, doy = dateout.year, dateout.strftime("%j")
    # folder_day = f'{folder_base}/{year}/{doy}'
    
    # file_output = f'{folder_day}/{year}{doy}_complete.pkl'
    
    # print('----------------------------------------')
    # print(f'Saving file: {year}{doy}_complete.pkl')
    # print(f'Diretory: {folder_day}')
    
    # if not os.path.exists(file_output):
    #     # with open(file_output, 'wb') as file:
    #     #     pickle.dump(sp3_all, file)
    #     sp3_all.to_parquet(file_output)
    
    return sp3_all

def concat_all_v2(paths, datein):
   
    dfs = []

    # Loop over the range of days (from -1 day to +1 day)
    for i in range(-1, 2):
        dateout = datein + timedelta(days=i)
        year, doy = dateout.year, dateout.strftime("%j")
        # folder_day = f'{cv.fgnss1s}/raw/sp3/{year}/{doy}/'
        folder_day = paths["sp3_folder"].parent.parent / str(year) / str(doy)
        dfs.append(concat_daybyday(folder_day))  # Append the dataframe to the list
    
    # Concatenate all the dataframes in the list
    sp3_all = pd.concat(dfs, ignore_index=True)
    
    # Sort DataFrame by 'horario_s' column in ascending order
    
    sp3_all = times(sp3_all)
    
    sp3_all.sort_values(by='time_j', inplace=True)
    
    # Reset the index
    sp3_all.reset_index(drop=True, inplace=True)
    
    
    return sp3_all