#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:37:08 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import glob
import vaex 

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from latentpy.preprocessing import preprocess_MODIS_df
from latentpy.preprocessing import thin_df
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 

proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data" 
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"

# - Define tabular granule archive from which to read
df_type = "Remapped_Reshaped" # Reshaped 
product = 'MYD021KM' # 'MOD021KM'
# - Define training archive 
archive_name = "archive_PE_bbox"  
# preprocessing_name = "Normalized_Bands"  #  # "Original_Bands"
preprocessing_name = "Normalized_Difference_Indices"
#-----------------------------------------------------------------------------.
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
tabular_data_dir = os.path.join(proj_data_dir, archive_name, df_type, product)
training_data_dir = os.path.join(proj_data_dir, archive_name, "training_data", df_type, preprocessing_name)
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)  
if not os.path.exists(model_dir):
    os.makedirs(model_dir)      
#-----------------------------------------------------------------------------.
### Define ID variables   
if df_type == "Remapped_Reshaped":
    ID_vars = ['idx_x','idx_y','overpass_time','x','y']
else: # Reshaped
    ID_vars = ['idx_x','idx_y','overpass_time','latitude','longitude']
    
#-----------------------------------------------------------------------------.
### Open the the multispectral tabular dataset 
# - Find hdf5 files
list_hdf5 = glob.glob(os.path.join(tabular_data_dir,"*.hdf5"))
# - Open it into vaex
df_full = vaex.open_many(list_hdf5)
# - Preprocess dataframe 
df_full = preprocess_MODIS_df(df_full, ID_vars=ID_vars, preprocessing_option=preprocessing_name)
# - Subset daytime data
df_full = df_full[df_full['overpass_time'].dt.hour > 7 ] 
df_full = df_full[df_full['overpass_time'].dt.hour < 17 ]
# - Save database
df_full.export_hdf5(os.path.join(training_data_dir,"full_training_data.hdf5"))
##----------------------------------------------------------------------------.
# - Retrieve channels names
column_names = df_full.column_names
channels_selected = [column_name for column_name in column_names if column_name not in ID_vars]
##----------------------------------------------------------------------------.
# - Thin the training dataset (round and drop duplicates)
df_train = thin_df(df_full, channels_selected, rounding_decimals = 2)
df_train.export_hdf5(os.path.join(training_data_dir,"unique_training_data.hdf5"))
#-----------------------------------------------------------------------------.
# - Standardize the data to [0-1]
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(df_train[channels_selected])
joblib.dump(minmax_scaler, os.path.join(model_dir,'scaler_minmax.sav'))

#-----------------------------------------------------------------------------.
### Normalize the data to [0-1]
# - Training data
X_train_std = minmax_scaler.transform(df_train[channels_selected])
df_train_std_pandas = pd.DataFrame(data = X_train_std,
                                   columns = channels_selected)
df_train_std_pandas[ID_vars] = df_train[ID_vars].to_pandas_df()
df_train_std = vaex.from_pandas(df_train_std_pandas)
df_train_std.export_hdf5(os.path.join(training_data_dir,"unique_training_data_std.hdf5"))
# - Full database 
X_full_std = minmax_scaler.transform(df_full[channels_selected])
df_full_std_pandas = pd.DataFrame(data = X_full_std,
                                   columns = channels_selected)
df_full_pandas = df_full.to_pandas_df()
df_full_std_pandas[ID_vars] = df_full[ID_vars].to_pandas_df()
df_full_std = vaex.from_pandas(df_full_std_pandas)
df_full_std.export_hdf5(os.path.join(training_data_dir,"full_training_data_std.hdf5"))
#-----------------------------------------------------------------------------.
 

