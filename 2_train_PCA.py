#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:03:44 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")

import vaex 
import joblib
from sklearn.decomposition import PCA
from latentpy.dimred import latentRGB_Scaler 
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data" 
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"

# - Define tabular granule archive from which to read
df_type = "Remapped_Reshaped" # Reshaped 

# - Define training archive 
archive_name = "archive_PE_bbox"  
# preprocessing_name = "Normalized_Bands"   # "Original_Subset"
preprocessing_name = "Normalized_Difference_Indices"
#-----------------------------------------------------------------------------.
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
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
### Load standardized training data 
df_train_std = vaex.open(os.path.join(training_data_dir,"unique_training_data_std.hdf5"))
# - Retrieve channels names
column_names = df_train_std.column_names
channels_selected = [column_name for column_name in column_names if column_name not in ID_vars]
# - Subset channels
df_train_std = df_train_std[channels_selected] 
# - Convert to numpy
df_train_std = df_train_std.to_pandas_df().to_numpy() 

#-----------------------------------------------------------------------------.
### Define PCA 2D projection 
pca2D = PCA(n_components=2)
pca2D.fit(df_train_std) # "trained PCA

### Define PCA 3D projection 
pca3D = PCA(n_components=3)
pca3D.fit(df_train_std) # "trained PCA
#-----------------------------------------------------------------------------
### Save PCA models 
pca2D_filepath = os.path.join(model_dir,'pca2D.sav')
pca3D_filepath = os.path.join(model_dir,'pca3D.sav')
joblib.dump(pca2D, pca2D_filepath)
joblib.dump(pca3D, pca3D_filepath)
#-----------------------------------------------------------------------------.
### Define PCA 3D RGB scaler  
pca3D_latent = pca3D.transform(df_train_std) 
pca3D_RGB_scaler = latentRGB_Scaler(pca3D_latent)
# - Save the scaler
joblib.dump(pca3D_RGB_scaler, os.path.join(model_dir,'scaler_pca3D_RGB.sav'))
#-----------------------------------------------------------------------------.
print("PCA training completed")
