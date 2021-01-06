#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:53:18 2020

@author: ghiggi
"""
import vaex 
import pandas as pd
import xarray as xr 

# - Define corrupted channels 
# chs = ['16','15','14hi','13hi','14lo','13lo','12','11','10','9',  # lot missing / saturated 
#         '30', # 27,         # noisy
#         '8',                # partially corrupted at the borders
#         '36','24','21','6'] # stripes

def NDI_fun(x,y):
    return (x-y)/(x+y)
#-----------------------------------------------------------------------------.
############################################
## Custom xarray Dataset preprocessing #####
############################################
def original_MODIS_bands_ds(ds):
    chs = ['1', '2', '3', '4', '5','7', 
           '17','18', '19', '20',
           '22', '23', '25', '26',
           '27', '28', '29', '31', '32', '33', '34', '35']
    return ds[chs]

def normalize_MODIS_bands_ds(ds):
    l_ds = []
    VIS_chs = ["1", "2", "3", "4", "5", "17", "18", "19"]
    SWIR_chs = ['6','7']
    IR_chs = ['22','23','27','32','33','34','35']
    #-------------------------------------------------------------------------.
    # VIS normalization
    ds_stack = ds[VIS_chs].to_stacked_array(new_dim="channels", 
                                            sample_dims=['x','y','time'],
                                            name="StackedChannels").stack()
    VIS_mean = ds_stack.mean('channels').compute()
    for ch in VIS_chs:
        tmp_da = ds[ch]/VIS_mean
        tmp_da.name = ch
        l_ds.append(tmp_da)
    #-------------------------------------------------------------------------.
    # SWIR normalization
    ds_stack = ds[SWIR_chs].to_stacked_array(new_dim="channels", 
                                             sample_dims=['x','y','time'],
                                             name="StackedChannels").stack()
    SWIR_mean = ds_stack.mean('channels').compute()
    for ch in SWIR_chs:
        tmp_da = ds[ch]/SWIR_mean
        tmp_da.name = ch
        l_ds.append(tmp_da)
    #-------------------------------------------------------------------------.    
    ## IR Normalization
    ds_stack = ds[IR_chs].to_stacked_array(new_dim="channels", 
                                             sample_dims=['x','y','time'],
                                             name="StackedChannels").stack()
    IR_mean = ds_stack.mean('channels').compute()
    for ch in IR_chs:
        tmp_da = ds[ch]/IR_mean
        tmp_da.name = ch
        l_ds.append(tmp_da) 
    #-------------------------------------------------------------------------.    
    ## Merge  
    ds_normalized = xr.merge(l_ds)
    ds_normalized = ds_normalized.compute()
    return ds_normalized

def NDI_MODIS_bands_ds(ds):
    l_ds = []
    VIS_chs = ["1", "2", "3", "4", "5", "17", "18", "19"]
    SWIR_chs = ['6','7', '26']
    IR_chs = ['22','23','27','32','33','34','35']
    solar_chs = VIS_chs + SWIR_chs
    ##------------------------------------------------------------------------.
    # VIS NDI
    for i, ch in enumerate(solar_chs):
        for j in range(i+1, len(solar_chs)):
            tmp_da = NDI_fun(ds[ch], ds[solar_chs[j]]) 
            tmp_da.name = ch  + "_" + solar_chs[j]
            l_ds.append(tmp_da)
    ##------------------------------------------------------------------------.
    # IR NDI
    for i, ch in enumerate(IR_chs):
        for j in range(i+1, len(IR_chs)):
            tmp_da = NDI_fun(ds[ch], ds[IR_chs[j]]) 
            tmp_da.name = ch  + "_" + IR_chs[j]
            l_ds.append(tmp_da)
    ##------------------------------------------------------------------------.
    ## Merge  
    ds_NSI = xr.merge(l_ds)
    ds_NSI = ds_NSI.compute()
    return ds_NSI

def preprocess_MODIS_ds(ds, preprocessing_option):
    if (preprocessing_option == "Original_Bands"):
        return original_MODIS_bands_ds(ds)
    elif (preprocessing_option == "Normalized_Bands"):
        return normalize_MODIS_bands_ds(ds)
    elif (preprocessing_option == "Normalized_Difference_Indices"):
        return NDI_MODIS_bands_ds(ds)
    else: 
        raise ValueError("Such preprocessing option not yet implemented!")
#-----------------------------------------------------------------------------. 
##############################
## Dataframe preprocessing ###
##############################
def thin_df(df_vaex, channels_selected, rounding_decimals=2):
    # - Round values & drop duplicates
    df_pandas = df_vaex.to_pandas_df()
    df_pandas[channels_selected] = df_pandas[channels_selected].round(decimals=rounding_decimals)
    # df_train[channels_selected] = df_train[channels_selected].round(decimals=rounding_decimals) # vaex > v4
    # - Drop duplicates 
    df_pandas = df_pandas.drop_duplicates(subset=channels_selected)   
    ##------------------------------------------------------------------------.
    # - Convert back to vaex 
    df_train = vaex.from_pandas(df_pandas)
    return df_train

##-----------------------------------------------------------------------------.
### Filter the data
# TODO ??? Histogram for each band of 0-rounded counts
# TODO ??? Binning by interval of 2 ? Custom for each channel?  
# TODO ??? Unique, Count occurence  

#-----------------------------------------------------------------------------.
############################################
### Custom preprocessing options for df  ###
############################################
    
def original_MODIS_bands_df(df_vaex, ID_vars):
    # - Define channels not corrupted and without many NaN 
    chs = ['1', '2', '3', '4', '5','7', 
           '17','18', '19', '20',
           '22', '23', '25', '26',
           '27', '28', '29', '31', '32', '33', '34', '35']
    chs = ["ch_" + c for c in chs]  
    columns_subset = ID_vars + chs
    ##------------------------------------------------------------------------.
    # - Remove columns with corrupted channels 
    df_vaex = df_vaex[columns_subset]
    df_vaex.column_names
    ##------------------------------------------------------------------------.
    # - Remove rows with NaN
    df_vaex = df_vaex.dropnan()
    df_vaex = df_vaex.dropna()
    return df_vaex
    
def normalize_MODIS_bands_df(df_vaex, ID_vars):
    VIS_chs = ["1", "2", "3", "4", "5", "17", "18", "19"]
    SWIR_chs = ['6','7']
    IR_chs = ['22','23','27','32','33','34','35']
    VIS_chs = ["ch_" + c for c in VIS_chs] 
    SWIR_chs = ["ch_" + c for c in SWIR_chs] 
    IR_chs = ["ch_" + c for c in IR_chs] 
    chs = VIS_chs + SWIR_chs + IR_chs
    columns_subset = ID_vars + chs
    ##------------------------------------------------------------------------.
    # - Remove columns with corrupted channels 
    df_vaex = df_vaex[columns_subset]
    df_vaex.column_names
    ##------------------------------------------------------------------------.
    # - Remove rows with NaN
    df_vaex = df_vaex.dropnan()
    df_vaex = df_vaex.dropna()
    df_pandas = df_vaex.to_pandas_df()
    ##------------------------------------------------------------------------.
    # - Normalize bands 
    df_pandas[VIS_chs] = df_pandas[VIS_chs].div(df_pandas[VIS_chs].mean(axis=1), axis=0)   
    df_pandas[SWIR_chs] = df_pandas[SWIR_chs].div(df_pandas[SWIR_chs].mean(axis=1), axis=0) 
    df_pandas[IR_chs] = df_pandas[IR_chs].div(df_pandas[IR_chs].mean(axis=1), axis=0)
    return vaex.from_pandas(df_pandas)

    
def NDI_MODIS_bands_df(df_vaex, ID_vars):
    VIS_chs = ["1", "2", "3", "4", "5", "17", "18", "19"]
    SWIR_chs = ['6','7', '26']
    IR_chs = ['22','23','27','32','33','34','35']
    VIS_chs = ["ch_" + c for c in VIS_chs] 
    SWIR_chs = ["ch_" + c for c in SWIR_chs] 
    IR_chs = ["ch_" + c for c in IR_chs] 
    solar_chs = VIS_chs + SWIR_chs
    chs = VIS_chs + SWIR_chs + IR_chs
    columns_subset = ID_vars + chs
    ##------------------------------------------------------------------------.
    # - Remove columns with corrupted channels 
    df_vaex = df_vaex[columns_subset]
    df_vaex.column_names
    ##------------------------------------------------------------------------.
    # - Remove rows with NaN
    df_vaex = df_vaex.dropnan()
    df_vaex = df_vaex.dropna()
    df_pandas = df_vaex.to_pandas_df()
    ##------------------------------------------------------------------------.
    # - Compute Normalized Difference Index 
    df_NDI = df_pandas[ID_vars]
    # - Solar channels 
    for i, ch in enumerate(solar_chs):
        for j in range(i+1, len(solar_chs)):
            tmp_name = ch  + "_" + solar_chs[j]
            df_NDI[tmp_name] = NDI_fun(df_pandas[ch], df_pandas[solar_chs[j]])
    # - IR channels
    for i, ch in enumerate(IR_chs):
        for j in range(i+1, len(IR_chs)):
            tmp_name = ch  + "_" + IR_chs[j]
            df_NDI[tmp_name] = NDI_fun(df_pandas[ch], df_pandas[IR_chs[j]])
    ##------------------------------------------------------------------------.       
    return vaex.from_pandas(df_NDI)
    
    
    
def preprocess_MODIS_df(df_vaex, ID_vars, preprocessing_option):
    if (preprocessing_option == "Original_Bands"):
        return original_MODIS_bands_df(df_vaex, ID_vars)
    elif (preprocessing_option == "Normalized_Bands"):
        return normalize_MODIS_bands_df(df_vaex, ID_vars)
    elif (preprocessing_option == "Normalized_Difference_Indices"):
        return NDI_MODIS_bands_df(df_vaex, ID_vars)
    else: 
        raise ValueError("Such preprocessing option not yet implemented!")



    
  
 