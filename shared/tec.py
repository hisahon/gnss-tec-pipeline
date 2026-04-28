# -*- coding: utf-8 -*-
"""Cálculo de TEC a partir de RINEX (GPS/Galileo/BeiDou/QZSS/SBAS + GLONASS).

Autor original: Honda, R. H. (Fev/2023).
"""
import re
from datetime import timedelta
from functools import reduce

import numpy as np
import pandas as pd


# def extract_indexes(Df_RINEX):
#     # Extracting unique PRN names present in the file
#     prns = Df_RINEX['prn'].unique() 
#     prns.sort()
    
#     # Extracting unique constellations (first letter of PRN) present in the file
#     constellations = pd.Series(prns).str[:1].unique()
    
#     # # Dropping the columns 'horario', 'horario_s', 'prn' and converting remaining columns to numeric
#     # cols = Df_RINEX.columns.drop(['horario','horario_s','prn'])
#     # Df_RINEX[cols] = Df_RINEX[cols].apply(pd.to_numeric, errors='coerce')
    
#     # Extracting the index of the entire RINEX file
#     idx_all = Df_RINEX.index
    
#     # Returning the extracted indexes, constellations and PRNs
#     return idx_all, constellations, prns


def frequencies_touse(Df_RINEX, obs):

    # pseudo_range = prange
    # phase carier = pcarrier
    if obs == 'prange':
        obs_letter = 'C'
    elif obs =='pcarrier':
        obs_letter = 'L'
        
    
    # Drop unnecessary columns and keep only those starting with 'L' (phase carrier)
    kind = Df_RINEX.columns
    kind = kind[kind.str[:1] == obs_letter]
    
    # Create a slice for the 'G' satellites
    # Df_piece = Df_RINEX[Df_RINEX['prn'].str[:1] == gnss]
    # Df_piece = Df_RINEX

    # Count the number of non-null values for each frequency and store them in a list
    ndata = []
    for freq in kind:
        ndata.append([freq, Df_RINEX[freq].notnull().sum()])

    # Create a DataFrame from the list of frequency counts
    ndata = pd.DataFrame(ndata, columns=['Label', 'Value'])
    
    # Find the rows with at least two values for each kind of data
    valid_kinds = ndata['Label'].str[2].value_counts().index[ndata['Label'].str[2].value_counts() >= 2]
    
    # The "a" value will just appear in the data if, and only iff i make a treatment,
    #in the raw data that's combine the same kind of signals.
    
    if len(valid_kinds) > 0:
        if 'a' in valid_kinds:
            valid_kinds = ['a']
    
        # Find the two rows with the highest 'Value' for each valid kind of data
        top2 = ndata[ndata['Label'].str[2].isin(valid_kinds)].groupby(ndata['Label'].str[2])['Value'].nlargest(2).index.get_level_values(1)

    
        # # Filter the DataFrame based on the specified conditions
        # result_ndata = ndata.loc[top2]
        
    # i dont know if this routine works yet, but surely it will become another function
    # =============================================================================
    #     # # Sort the DataFrame by the 'Value' column in descending order and reset the index
    #     # ndata = ndata.sort_values('Value', ascending=False).reset_index(drop=True)
    #     
    #     # #This part of the routine excludes entries with duplicated channel
    #     # ndata['NumericalPart'] = ndata['Label'].str.extract('(\d+)')
    #     # first_occurrence_indices = ndata['NumericalPart'].duplicated(keep='first')
    #     # ndata = ndata[~first_occurrence_indices]
    #     # ndata=ndata.reset_index(drop=True)
    # =============================================================================
    
        # Get the labels of the first two frequencies with the highest count
        ch0 = ndata.loc[top2[0], 'Label']
        ch1 = ndata.loc[top2[1], 'Label']
    
        # Count the number of frequencies with non-null values
        ch_usable = (ndata['Value'] > 0).sum()
        
    else:
        ch0, ch1, ch_usable = None, None, None

    return [ch0, ch1, ch_usable]



def frequencies2use_rinex2(df, obs):

    # pseudo_range = prange
    # phase carier = pcarrier
    if obs == 'prange':
        obs_letter = 'C'
    elif obs =='pcarrier':
        obs_letter = 'L'
        
    
    # Drop unnecessary columns and keep only those starting with 'L' (phase carrier)
    kind = df.columns
    kind = kind[kind.str[:1] == obs_letter]
    
    # Create a slice for the 'G' satellites
    # Df_piece = Df_RINEX[Df_RINEX['prn'].str[:1] == gnss]
    # Df_piece = Df_RINEX

    # Count the number of non-null values for each frequency and store them in a list
    ndata = []
    for freq in kind:
        ndata.append([freq, df[freq].notnull().sum()])

    # Create a DataFrame from the list of frequency counts
    ndata = pd.DataFrame(ndata, columns=['Label', 'Value'])
    ndata.sort_values(by='Value')
    ndata = ndata[ndata['Value'] != 0]
    ndata = ndata.reset_index(drop=True)
    
    
    if len(ndata) >=2:

        # Get the labels of the first two frequencies with the highest count
        ch0 = ndata['Label'][0]
        ch1 = ndata['Label'][1]
    
        # Count the number of frequencies with non-null values
        ch_usable = (ndata['Value'] > 0).sum()
        
    else:
        ch0, ch1, ch_usable = None, None, None

    return [ch0, ch1, ch_usable]


def gnss_channels():
    from io import StringIO
    # https://files.igs.org/pub/data/format/rinex304.pdf
    data = '''C;1;1557.42
    C;2;1561.098
    C;5;1176.45
    C;6;1268.52
    C;7;1207.14
    C;8;1191.795
    E;1;1575.42
    E;5;1176.45
    E;6;1278.75
    E;7;1207.14
    E;8;1191.795
    G;1;1575.42
    G;2;1227.6
    G;5;1176.45
    I;5;1176.45
    I;9;2492.028
    J;1;1575.42
    J;2;1227.6
    J;5;1176.45
    J;6;1278.75
    S;1;1575.42
    S;5;1176.45
    '''
    
    # Read the CSV data with custom column names and semicolon separator
    df = pd.read_csv(StringIO(data), sep=';', header=None, names=['gnss', 'channel', 'frequency'])
    df['gnss'] = df['gnss'].str.strip()
    df['frequency']=df['frequency']*1.e6
    
    df.set_index(['gnss', 'channel'], inplace=True)

    return df



def chose_frequencies(gnss,channel):
    '''
    This function chooses the frequency values based on the selected constellation and channel.
    It uses a pre-defined dictionary to map each constellation and channel to its corresponding frequency value.
    
    Inputs:
    - constelation (str): the selected constellation (e.g. 'E', 'C', 'G', or 'S')
    - channel (list of str): the selected channels (e.g. ['L1', 'L5'] for GPS or ['L1', 'L7'] for Galileo)
    
    Outputs:
    - l1 (float): the frequency value for channel 1
    - l2 (float): the frequency value for channel 2
    '''

    df_freq=gnss_channels()
    
    # Looking up the frequency values for the selected constellation and channels
    # If there is this frequency in the database:
    
    warning='THIS_FREQUENCY_DOESNT_EXISTS_TO_THIS_PNR'
    
    ch0 = int(re.search(r'\d+', channel[0]).group())
    ch1 = int(re.search(r'\d+', channel[1]).group())
    
    if df_freq.index.isin([(gnss,ch0)]).any():
        l1=df_freq.loc[(gnss,ch0),'frequency']
    else:
        l1 = warning
   
    if df_freq.index.isin([(gnss,ch1)]).any():
        l2=df_freq.loc[(gnss,ch1),'frequency']
    else:
        l2 = warning
    
    
    warning_status = (l1 == warning) or (l2== warning)
    if warning_status:
        print(f'SOME OF THIS FREQUENCY DESNT BELONG TO THIS CONSTELATIOn [{gnss}]')
   
    return l1,l2,warning_status


def tec_bytype(f1,f2,L1,L2,obs):
    
    if (f1*f1 - f2*f2) ==0:
        return np.nan
    
    tf = f1*f1 * f2*f2/ (f1*f1 - f2*f2)
    # the constand 40.3 is m³/s²
    tf = tf/40.308/1e16
    
    #speed of light in vacuum
    c = 299792458
    # #speed of light in air
    # c = c / 1.0003
    
    if obs == 'prange':
        tec = tf*(L1 - L2)
    
    elif obs == 'pcarrier':
        tec = tf*(c/f1*L1 - c/f2*L2)
        
    # if obs == 'one_channel':
    #     tec = 0.5/40.3 * f1*f1 (C1 - c*L1/f1) / 1e16
        
    
        
    return tec
        
    

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

    # Add the merged column to the original DataFrame
    df[new_column_name] = merged_column

    return df

# # Example usage:
# df = pd.DataFrame({
#     'Column1': [1, np.nan, 3, 4, np.nan],
#     'Column2': [10, np.nan, 30, 40, np.nan],
#     'Column3': [100, 200, 300, np.nan, np.nan],
#     'Column4': [5, 4, 3, np.nan, 15],
#     'Column5': [50, np.nan, 30, 20, np.nan]
# })

# columns_to_merge = ['Column4', 'Column1', 'Column2', 'Column3', 'Column4']
# new_column_name = 'MergedColumn'

# result_df = merge_columns(df, columns_to_merge, new_column_name=new_column_name)
# print(result_df)



# =============================================================================
# Calculado TEC do Glonas
# =============================================================================
def calc_tecR(df_rinex, obs):
    '''
    Input:
    
        df_rinex: Pandas DataFrame containing RINEX data
        Df_prn_names: Pandas Series containing PRN names
        idx_all: Pandas Index containing all the indexes in the RINEX file
    
    Output:
    
        df_rinex: Pandas DataFrame with added TEC values for GLONASS PRNs
    
    Description:
    The function calculates TEC values for GLONASS PRNs in the given RINEX data. The function extracts only the GLONASS PRN values from the given PRN names. Then, it separates each PRN into its respective names. For each GLONASS PRN, it calculates the TEC using the formula, which takes into account the channel number, frequency, and speed of light. Finally, the TEC values are added to the respective indexes in the input DataFrame and returned.

    '''
    # # pseudo_range = psr
    # # phase carier = phc
    # obs = 'psr'
    # ch=frequencies_touse(df_rinex, obs)
    
    prns = df_rinex['prn'].unique(); prns.sort()
    iall=df_rinex.index
    
    channel = [1, -4, 5, 6, 1, -4, 5, 6, -2, -7,
               0, -1, -2, -7, 0, -1, 4, -3, 3, 2,
               4, -3, 3, 2, np.nan, -6,np.nan]
    

    
    
    for prn in prns:
        df_prn = df_rinex[df_rinex['prn']==prn]
        
        ch=frequencies_touse(df_prn, obs)
        # print(frequencies_touse(df_rinex[df_rinex['prn']==prn], obs))
        
        num = int(prn[1:])-1
        iprn = iall[df_rinex['prn']==prn].copy() #Extraindo os indexes apenas do prn de interesse
       
        if not any(value is None for value in ch):    
            #Calculo do TEC
            f1 = 1602e6 + channel[num] * 562.5e3
            f2 = 1246e6 + channel[num] * 437.5e3
            
            
            if ch[0][1:2] == '1':
                L1 = df_rinex.loc[iprn,ch[0]].copy()
                L2 = df_rinex.loc[iprn,ch[1]].copy()
            elif ch[1][1:2] == '1':
                L1 = df_rinex.loc[iprn,ch[1]].copy()
                L2 = df_rinex.loc[iprn,ch[0]].copy()
                
                
            df_rinex.loc[iprn,f'tec_{obs}'] = tec_bytype(f1,f2,L1,L2,obs)
        
        # else:
        #     df_rinex = df_rinex.drop(iprn)
        #     iall=df_rinex.index
            

    return df_rinex

def tec_rinex2_glonass(df_rinex, obs):
    '''
    Input:
    
        df_rinex: Pandas DataFrame containing RINEX data
        Df_prn_names: Pandas Series containing PRN names
        idx_all: Pandas Index containing all the indexes in the RINEX file
    
    Output:
    
        df_rinex: Pandas DataFrame with added TEC values for GLONASS PRNs
    
    Description:
    The function calculates TEC values for GLONASS PRNs in the given RINEX data. The function extracts only the GLONASS PRN values from the given PRN names. Then, it separates each PRN into its respective names. For each GLONASS PRN, it calculates the TEC using the formula, which takes into account the channel number, frequency, and speed of light. Finally, the TEC values are added to the respective indexes in the input DataFrame and returned.

    '''
    # # pseudo_range = psr
    # # phase carier = phc
    # obs = 'psr'
    # ch=frequencies_touse(df_rinex, obs)
    
    prns = df_rinex['prn'].unique(); prns.sort()
    iall=df_rinex.index
    
    channel = [1, -4, 5, 6, 1, -4, 5, 6, -2, -7,
               0, -1, -2, -7, 0, -1, 4, -3, 3, 2,
               4, -3, 3, 2, np.nan, -6,np.nan]
    

    
    
    for prn in prns:
        
        df_prn = df_rinex[df_rinex['prn']==prn]
        iprn = iall[df_rinex['prn']==prn].copy() #Extraindo os indexes apenas do prn de interesse
       
        ch=frequencies2use_rinex2(df_prn, obs)
        # print(frequencies_touse(df_rinex[df_rinex['prn']==prn], obs))
                
        if not any(value is None for value in ch):    
            num = int(prn[1:])-1
            #Calculo do TEC
            f1 = 1602e6 + channel[num] * 562.5e3
            f2 = 1246e6 + channel[num] * 437.5e3
            
            
            if ch[0][1:2] == '1':
                L1 = df_rinex.loc[iprn,ch[0]].copy()
                L2 = df_rinex.loc[iprn,ch[1]].copy()
            elif ch[1][1:2] == '1':
                L1 = df_rinex.loc[iprn,ch[1]].copy()
                L2 = df_rinex.loc[iprn,ch[0]].copy()
                
                
            df_rinex.loc[iprn, f'tec_{obs}'] = tec_bytype(f1,f2,L1,L2,obs)
        
        else:
            df_rinex.loc[iprn, f'tec_{obs}'] = np.nan
        #     df_rinex = df_rinex.drop(iprn)
        #     iall=df_rinex.index
            

    return df_rinex

# =============================================================================
# Calculado TEC das demais constelações
# =============================================================================
def calc_TEC(df_rinex,gnss,obs):
    '''
    Calculates the Total Electron Content (TEC) of a GPS signal using data from a RINEX file.
    
    Parameters:
    df_rinex (pandas.DataFrame): DataFrame with RINEX data.
    L1 (float): Frequency of GPS L1 signal in Hz.
    L2 (float): Frequency of GPS L2 signal in Hz.
    
    Returns: df_rinex
        
    '''
    prns = df_rinex['prn'].unique(); prns.sort()
    iall=df_rinex.index
    
    for prn in prns:
        
        df_prn = df_rinex[df_rinex['prn']==prn].copy()
        iprn = iall[df_rinex['prn']==prn].copy()
        
        ch=frequencies_touse(df_prn, obs)
        
        
        iprn = iall[df_rinex['prn']==prn].copy()
        
        if not any(value is None for value in ch):  
            # # ch=frequencies_touse(Df_RINEX, obs)
            # channels = frequencies_touse(df_rinex, obs)
    
            f1, f2, delete_this_constelation = chose_frequencies(gnss, ch)
            
            #Locating and calculating the TEC values where they correspond to the indices
            ch1, ch2 = ch[0], ch[1]
            
            L1 = df_prn[ch1].copy()
            L2 = df_prn[ch2].copy()
            
            # df_rinex[f'tec_{obs}']=tec_bytype(f1,f2,L1,L2,obs)
            
            df_rinex.loc[iprn, f'tec_{obs}'] = tec_bytype(f1,f2,L1,L2,obs) 
            
            # df_rinex[f'tec_{obs}'] = df_rinex[f'tec_{obs}'] - df_rinex[f'tec_{obs}'].min()
            # df_rinex['TEC'] = df_semi - df_semi.min()
        else:
            df_rinex.loc[iprn, f'tec_{obs}'] = np.nan
        #     df_rinex = df_rinex.drop(iprn)
        #     iall=df_rinex.index
            
            
    return df_rinex

def tec_rinex2_general(df_rinex,gnss,obs):
    '''
    Calculates the Total Electron Content (TEC) of a GPS signal using data from a RINEX file.
    
    Parameters:
    df_rinex (pandas.DataFrame): DataFrame with RINEX data.
    L1 (float): Frequency of GPS L1 signal in Hz.
    L2 (float): Frequency of GPS L2 signal in Hz.
    
    Returns: df_rinex
        
    '''
    prns = df_rinex['prn'].unique(); prns.sort()
    iall=df_rinex.index
    
    for prn in prns:
        
        
        df_prn = df_rinex[df_rinex['prn']==prn].copy()
        iprn = iall[df_rinex['prn']==prn].copy()
        
        ch=frequencies2use_rinex2(df_prn, obs)
        
        if not any(value is None for value in ch):  
            # # ch=frequencies_touse(Df_RINEX, obs)
            # channels = frequencies_touse(df_rinex, obs)
    
            f1, f2, delete_this_constelation = chose_frequencies(gnss, ch)
            
            #Locating and calculating the TEC values where they correspond to the indices
            ch1, ch2 = ch[0], ch[1]
            
            L1 = df_prn[ch1].copy()
            L2 = df_prn[ch2].copy()
            
            # df_rinex[f'tec_{obs}']=tec_bytype(f1,f2,L1,L2,obs)
            
            df_rinex.loc[iprn, f'tec_{obs}'] = tec_bytype(f1,f2,L1,L2,obs) 
            
            # df_rinex[f'tec_{obs}'] = df_rinex[f'tec_{obs}'] - df_rinex[f'tec_{obs}'].min()
            # df_rinex['TEC'] = df_semi - df_semi.min()
        else:
            df_rinex.loc[iprn, f'tec_{obs}'] = np.nan
        #     df_rinex = df_rinex.drop(iprn)
        #     iall=df_rinex.index
            
            
    return df_rinex


def tec(df_rinex, gnss, obs):
    # for gnss in constelations:
    #If the constellation is not GLONASS, it calculates the TEC for that constellation
    delete_gnss = False
    if gnss != 'R':
        df_rinex = calc_TEC(df_rinex, gnss, obs)
    
    #If the constellation is GLONASS, it calculates the TEC for that constellation
    elif gnss == 'R':
        df_rinex = calc_tecR(df_rinex, obs)
    
    #Returns the updated dataframe
    return df_rinex

def tec_rinex2(df_rinex, gnss, obs):
    # for gnss in constelations:
    #If the constellation is not GLONASS, it calculates the TEC for that constellation
    delete_gnss = False
    if gnss != 'R':
        df_rinex = tec_rinex2_general(df_rinex, gnss, obs)
    
    #If the constellation is GLONASS, it calculates the TEC for that constellation
    elif gnss == 'R':
        df_rinex = tec_rinex2_glonass(df_rinex, obs)
    
    #Returns the updated dataframe
    return df_rinex


def del_constelations(Df_RINEX,sta_prns,idx_all,con='E'):
# =============================================================================
#  Excluindo algumas constelações
# =============================================================================
    mask_del=Df_RINEX['prn'].str[:1]!=con
    Df_RINEX=Df_RINEX[mask_del]
    idx_all = idx_all[mask_del]
    sta_prns=sta_prns[~pd.Series(sta_prns).str.contains(con).values] #o sinal de ~ inverte os valores booleanos
    
    return Df_RINEX, idx_all, sta_prns


def moving_average(df_gnss, mean_time):
    prns = df_gnss['prn'].unique()
    
    for prn in prns:
        df_prn = df_gnss[df_gnss['prn']==prn].copy()
        iprn = df_prn.index.values
        
        df_tec = df_prn['tec_pcarrier']
        
        res = df_prn['time_e'].diff().mode().iloc[0]
        window_size = int(mean_time * 60 / res)
        
        rolling_avg = df_tec.rolling(window = window_size + 1, center=True).mean()
        
        df_gnss.loc[iprn, f'dtec{int(mean_time)}'] = df_tec - rolling_avg
    
    return df_gnss






# def del_prn(Df_RINEX,del_prn):
# # =============================================================================
# #  Excluindo algumas constelações
# # =============================================================================
#     for prn in del_prn:
#         mask_del=Df_RINEX['prn'].str[:3]!=prn
#         Df_RINEX=Df_RINEX[mask_del]
    
#     sta_prns=Df_RINEX['prn'].unique()
    
#     return Df_RINEX, sta_prns


# def TEC_input(file,t_min,t_max,elev):
    
#     Df_TEC = pd.read_csv(file)
#     # cols = Df_TEC.columns.drop(['horario','prn'])
#     # Df_TEC[cols] = Df_TEC[cols].apply(pd.to_numeric, errors='coerce')
#     Df_TEC['horario']=pd.to_datetime(Df_TEC['horario'])
#     Df_TEC = Df_TEC[(Df_TEC['horario'] >= t_min) & (Df_TEC['horario'] <= t_max)]
    
#     Df_TEC = Df_TEC[Df_TEC['elevation'] >= elev]
    
    
#     Prn_names = Df_TEC['prn'].unique() #Extraindo os nomes dos prns presentes no arquivo
#     Prn_names.sort()
    
#     return Df_TEC, Prn_names
    

# def doy_to_month_day(year, doy):
#     # create a datetime object for January 1st of the given year
#     date = datetime(year, 1, 1)
#     # add the number of days to get to the desired day of the year
#     date += timedelta(days=doy - 1)
#     # extract the month and day information
#     month = date.month
#     day = date.day
#     return month, day

# def intervals_minute(year, doy, t_min, t_max, time_frame, ttype):
    
#     month,day=doy_to_month_day(year, doy)
#     # create datetime objects for the start and end times
#     # t_min = datetime(year, month, day, h_min, m_min, 0)
#     # t_max = datetime(year, month, day, h_max, m_max, 0)
    
#     # create a timedelta object representing 5 minutes
    
#     if ttype == 'min':
#         frame = timedelta(minutes=time_frame)
#     if ttype == 'sec':
#         frame = timedelta(seconds=time_frame)
    
#     # create an empty list to hold the intervals
#     intervals = []
    
#     # loop through the five-minute intervals between t_min and t_max
#     interval = t_min
#     while interval <= t_max:
#         intervals.append(interval)
#         interval += frame
    
#     return month, day, intervals

def intervals_smoth(year, t_min, t_max, time_frame,time_constant, ttype):
    
    # month,day=doy_to_month_day(year, doy)
    # create datetime objects for the start and end times
    # t_min = datetime(year, month, day, h_min, m_min, 0)
    # t_max = datetime(year, month, day, h_max, m_max, 0)
    
    # create a timedelta object representing 5 minutes
    
    if ttype == 'min':
        frame = timedelta(minutes=time_frame)
        cframe = timedelta(minutes=time_constant)
    if ttype == 'sec':
        frame = timedelta(seconds=time_frame)
        cframe = timedelta(seconds=time_constant)
    
    # create an empty list to hold the intervals
    intervals = []
    
    # loop through the five-minute intervals between t_min and t_max
    interval_low = t_min
    interval_sup = t_min + cframe + frame
    while interval_sup <= t_max:
        intervals.append([interval_low,interval_sup])
        interval_low += frame
        interval_sup += frame
    
    return intervals






