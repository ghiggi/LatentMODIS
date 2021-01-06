#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 01:38:27 2020

@author: ghiggi
"""
 
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import glob
import xarray as xr 
import numpy as np
import dask 
import vaex 
import vaex.ml
import matplotlib.pyplot as plt
import joblib
 
from latentpy.dimred import load_UMAP
from latentpy.dimred import load_ParametricUMAP

from latentpy.plotting import plot_pca2D
from latentpy.plotting import plot_umap2D
from latentpy.plotting import plot_scatter2D
from latentpy.plotting import plot_scatter3D
from latentpy.plotting import plot_scatter2D_napari
from latentpy.plotting import plot_scatter3D_napari
 
from latentpy.utils_reshape import reshape_to_RGB_composite
from latentpy.plotting import plot_RGB_DataArray
from latentpy.plotting import plot_RGB_DataArrays
from latentpy.plotting import plot_latent_spectral_space
 
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
model_dir = '/ltenas3/LatentMODIS/models' 
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# model_dir = "/home/ghiggi/Projects/LatentMODIS/models"

# - Define tabular granule archive from which to read
df_type = "Remapped_Reshaped" # Reshaped 

# - Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Normalized_Difference_Indices"  #  # "Original_Subset"
 
#-----------------------------------------------------------------------------.
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
tabular_data_dir = os.path.join(proj_data_dir, archive_name, df_type)
training_data_dir = os.path.join(proj_data_dir, archive_name, "training_data", df_type, preprocessing_name)
      
#-----------------------------------------------------------------------------.
### Define ID variables   
if df_type == "Remapped_Reshaped":
    s_coords = ['x','y']        # x and y of the planar projection 
    t_coords = 'overpass_time'
    ID_vars = ['idx_x','idx_y','overpass_time','x','y']
else: # Reshaped
    s_coords = ['idx_x','idx_y'] # idx of image row and column 
    t_coords = 'overpass_time'
    ID_vars = ['idx_x','idx_y','overpass_time','latitude','longitude']
    
#-----------------------------------------------------------------------------. 
### Load lazily the dataset
# - full_* means that all pixels inside the bbox are present 
#   --> It is possible to reshape the df to a space-time cube
# - unique_* means that many pixels (values) have been discarded during preprocessing
#   --> It is not possible to reshape the df to a space-time cube
df_full = vaex.open(os.path.join(training_data_dir,"full_training_data.hdf5")) 
df_full_std = vaex.open(os.path.join(training_data_dir,"full_training_data_std.hdf5")) 
 
df_train = vaex.open(os.path.join(training_data_dir,"unique_training_data.hdf5")) 
df_train_std = vaex.open(os.path.join(training_data_dir,"unique_training_data_std.hdf5")) 

#-----------------------------------------------------------------------------. 
### Select the desired dataset
# df_std = df_train_std.to_pandas_df() 
df_std = df_full_std.to_pandas_df() 

# - Subset daytime data
df_std = df_std[df_std['overpass_time'].dt.hour > 7 ] 
df_std = df_std[df_std['overpass_time'].dt.hour < 17 ]

# - Retrieve channels names
column_names = df_std.columns
channels_selected = [column_name for column_name in column_names if column_name not in ID_vars]

#-----------------------------------------------------------------------------.
####################
### Load models ####
####################
### - Load trained ParametricUMAP models
PCA_preprocessing = True
PCA_n_components = 20
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"

NN_umap2D = load_ParametricUMAP(model_dir = model_dir,
                                n_components = 2,
                                n_neighbors = n_neighbors, 
                                min_dist = min_dist, 
                                metric = metric,
                                PCA_preprocessing = PCA_preprocessing, 
                                PCA_n_components = PCA_n_components)
NN_umap3D = load_ParametricUMAP(model_dir = model_dir,
                                n_components = 3,
                                n_neighbors = n_neighbors, 
                                min_dist = min_dist, 
                                metric = metric,
                                PCA_preprocessing = PCA_preprocessing, 
                                PCA_n_components = PCA_n_components)

# - Watch training loss function 
plt.plot(NN_umap2D._history['loss'])
plt.plot(NN_umap3D._history['loss'])

##-----------------------------------------------------------------------------.
### - Load trained UMAP models
PCA_preprocessing = True
PCA_n_components = 20
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"

umap2D = load_UMAP(model_dir = model_dir, 
                   n_components = 2,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   metric = metric,
                   PCA_preprocessing = PCA_preprocessing, 
                   PCA_n_components = PCA_n_components)
umap3D = load_UMAP(model_dir = model_dir, 
                   n_components = 3,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   metric = metric,
                   PCA_preprocessing = PCA_preprocessing, 
                   PCA_n_components = PCA_n_components)

##----------------------------------------------------------------------------.
### - Load trained PCA models
pca2D_filepath = os.path.join(model_dir,'pca2D.sav')
pca3D_filepath = os.path.join(model_dir,'pca3D.sav')
pca2D = joblib.load(pca2D_filepath)
pca3D = joblib.load(pca3D_filepath)
pca3D_RGB_scaler = joblib.load(os.path.join(model_dir,'scaler_pca3D_RGB.sav'))

#-----------------------------------------------------------------------------.                
############################################
### Project data into the latent spaces ####
############################################
# - Project to PCA latent space 
pca2D_latent = pca2D.transform(df_std[channels_selected]) 
pca3D_latent = pca3D.transform(df_std[channels_selected]) 
# - Define colors from 3D latent space    
pca3D_RGB_color = pca3D_RGB_scaler.transform(pca3D_latent)
##----------------------------------------------------------------------------.
# - Project to UMAP latent space        
umap2D_latent = umap2D.transform(df_std[channels_selected]) # this is slow
umap3D_latent = umap3D.transform(df_std[channels_selected]) # this is slow
# - Define colors from 3D latent space 
# umap3D_RGB_color = umap3D.transform_to_RGB(df_std[channels_selected]) 
umap3D_RGB_color = umap3D.RGB_scaler.transform(umap3D_latent)

## ---------------------------------------------------------------------------.
# - Project to ParametricUMAP latent space  
NN_umap2D_latent = NN_umap2D.transform(df_std[channels_selected]) 
NN_umap3D_latent = NN_umap3D.transform(df_std[channels_selected]) 
# - Define colors from 3D latent space
# umap3D_RGB_color = NN_umap3D.transform_to_RGB(df_std[channels_selected]) 
NN_umap3D_RGB_color = NN_umap3D.RGB_scaler.transform(NN_umap3D_latent)
 
#-----------------------------------------------------------------------------.
##################################
### Latent space scatterplots ####
##################################
### - Plot 2D latent space (with matplotlib)
plt.figure()
plot_pca2D(pca2D_latent, color=pca3D_RGB_color)

plt.figure()
plot_umap2D(umap2D_latent, color=umap3D_RGB_color) 

plt.figure()
plot_umap2D(NN_umap2D_latent, color=NN_umap3D_RGB_color)

# Switching color between dim reduction methods 
plt.figure()
plot_umap2D(umap2D_latent, color=pca3D_RGB_color) 

plt.figure()
plot_umap2D(pca2D_latent, color=umap3D_RGB_color) 

# - PCA 2D embedding with ParametricUMAP 3D embedding colors
plt.figure()
plot_umap2D(pca2D_latent, color=NN_umap3D_RGB_color) 
# - ParametricUMAP 2D embedding with PCA 3D embedding colors
plt.figure()
plot_umap2D(NN_umap2D_latent, color=pca3D_RGB_color)

## Colored by overpass time [TODO: Check that do not depend on overpass time !] 
plt.figure()
plot_umap2D(umap2D_latent, color=df_std['overpass_time']) 


##----------------------------------------------------------------------------.
### - Plot 2D latent space (in napari)        
# plot_scatter2D_napari(embedding = pca2D_latent,
#                       color = pca3D_RGB_color, 
#                       name = "pca2D_latent", 
#                       axis_labels = ["PCA1", "PCA2"])

# plot_scatter2D_napari(embedding = umap2D_latent,
#                       color = umap3D_RGB_color, 
#                       name = "umap2D_latent", 
#                       axis_labels = ["Z1", "Z2"])

# plot_scatter2D_napari(embedding = NN_umap2D_latent,
#                       color = NN_umap3D_RGB_color, 
#                       name = "NN_umap2D_latent", 
#                       axis_labels = ["Z1", "Z2"])
##----------------------------------------------------------------------------.
### - Plot 3D latent space (with matplotlib)
# - Color do not represent same stuff between UMAP and PCA !!! 
# - Pay attention to not disinterpret
plot_scatter3D(embedding = NN_umap3D_latent,
               color = NN_umap3D_RGB_color,
               title = "ParametricUMAP 3D embedding",
               azim = 40) # view angle

plot_scatter3D(embedding = pca3D_latent,
               color = pca3D_RGB_color,
               title = "PCA 3D embedding",
               azim = 10) # view angle

##-----------------------------------------------------------------------------.
### - Plot 3D latent space (in napari)        
# plot_scatter3D_napari(embedding = pca3D_latent,
#                       color = pca3D_RGB_color, 
#                       name = "pca3D_latent", 
#                       axis_labels = ["PCA1", "PCA2", "PCA3"])

# plot_scatter3D_napari(embedding = umap3D_latent,
#                       color = umap3D_RGB_color, 
#                       name = "umap3D_latent", 
#                       axis_labels = ["Z1", "Z2", "Z3"])

# plot_scatter3D_napari(embedding = NN_umap3D_latent,
#                       color = NN_umap3D_RGB_color, 
#                       name = "NN_umap3D_latent", 
#                       axis_labels = ["Z1", "Z2", "Z3"])
#-----------------------------------------------------------------------------.
######################################
### Display latent image composite ###
######################################
# - Compute latent image composite 
df = df_std
dims = s_coords + [t_coords]
da_PCA_RGB = reshape_to_RGB_composite(df = df,
                                      RGB_matrix = pca3D_RGB_color,
                                      dims = dims, 
                                      name="PCA_Composite")

da_UMAP_RGB = reshape_to_RGB_composite(df = df,
                                       RGB_matrix = umap3D_RGB_color,
                                       dims = dims, 
                                       name="UMAP_Composite")

da_NN_UMAP_RGB = reshape_to_RGB_composite(df = df,
                                       RGB_matrix = NN_umap3D_RGB_color,
                                       dims = dims, 
                                       name="NN_UMAP_Composite")

# - Plot single RGB images  
da_PCA_RGB.isel(overpass_time=0).plot.imshow(x='x', y='y')
da_UMAP_RGB.isel(overpass_time=0).plot.imshow(x='x', y='y')
da_NN_UMAP_RGB.isel(overpass_time=0).plot.imshow(x='x', y='y')

# - Plot multiple RGB images   
plot_RGB_DataArrays(da_PCA_RGB.isel(overpass_time=slice(0,16)),  title="PCA RGB", time_dim=t_coords)
plot_RGB_DataArrays(da_PCA_RGB.isel(overpass_time=slice(16,32)), title="PCA RGB", time_dim=t_coords)
plot_RGB_DataArrays(da_UMAP_RGB.isel(overpass_time=slice(0,16)), title="UMAP RGB", time_dim=t_coords)  
plot_RGB_DataArrays(da_UMAP_RGB.isel(overpass_time=slice(16,32)), title="UMAP RGB", time_dim=t_coords)
plot_RGB_DataArrays(da_NN_UMAP_RGB.isel(overpass_time=slice(0,16)), title="ParametricUMAP RGB", time_dim=t_coords) 
plot_RGB_DataArrays(da_NN_UMAP_RGB.isel(overpass_time=slice(16,32)), title="ParametricUMAP RGB", time_dim=t_coords)

##----------------------------------------------------------------------------.
### - Plot with napari 
# da_PCA_RGB_t = da_PCA_RGB.transpose('RGB','overpass_time','x','y') 
# # da_PCA_RGB_t = da_PCA_RGB_t.reset_index("RGB") # Remove MultiIndex
# with napari.gui_qt():
#     # create an empty viewer
#     viewer = napari.Viewer()   
#     viewer.add_image(da_PCA_RGB_t, name='PCA_Composite', rgb=True) 

#-----------------------------------------------------------------------------.
######################################
### Display latent space vs. image ###
######################################
for t in range(0, len(da_PCA_RGB['overpass_time'])):
    plot_latent_spectral_space(da_RGB = da_PCA_RGB, 
                               df = df_std,
                               df_latent = pca2D_latent,
                               df_latent_colors = pca3D_RGB_color,
                               t_coords = t_coords,
                               time_idx = t)
for t in range(0, len(da_NN_UMAP_RGB['overpass_time'])):
    plot_latent_spectral_space(da_RGB = da_NN_UMAP_RGB, 
                               df = df_std,
                               df_latent = NN_umap2D_latent,
                               df_latent_colors = NN_umap3D_RGB_color,
                               t_coords = t_coords,
                               time_idx = t)

plot_latent_spectral_space(da_RGB = da_UMAP_RGB, 
                           df = df_std,
                           df_latent = umap2D_latent,
                           df_latent_colors = umap3D_RGB_color,
                           t_coords = t_coords,
                           time_idx = 0)

#-----------------------------------------------------------------------------.


