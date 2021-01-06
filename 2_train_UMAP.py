#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:19:38 2020

@author: ghiggi
"""
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
from latentpy.dimred import train_UMAP

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
### Define UMAP hyperparameters 
PCA_preprocessing = True
PCA_n_components = 20 # 10, 8, 5

n_neighbors = 15  
min_dist = 0.1
# n_neighbors = 100
# min_dist = 0.2
metric = "euclidean"
n_epochs = 200

#-----------------------------------------------------------------------------.
### Train UMAP 2D
train_UMAP( X = df_train_std,
            model_dir = model_dir,
            n_components = 2,
            # UMAP settings 
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            metric = metric,
            n_epochs = n_epochs,
            PCA_preprocessing = PCA_preprocessing, 
            PCA_n_components = PCA_n_components)
                   
#-----------------------------------------------------------------------------.
### Train UMAP 3D
train_UMAP( X = df_train_std,
            model_dir = model_dir,
            n_components = 3,
            # UMAP settings 
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            metric = metric,
            n_epochs = n_epochs,
            PCA_preprocessing = PCA_preprocessing, 
            PCA_n_components = PCA_n_components)



 