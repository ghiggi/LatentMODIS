#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:23:43 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import xarray as xr 
import joblib

from latentpy.preprocessing import preprocess_MODIS_ds
from latentpy.dimred import load_UMAP
from latentpy.dimred import load_ParametricUMAP
from latentpy.utils_reshape import reshape_to_RGB_composite
from latentpy.plotting import plot_RGB_DataArray
from latentpy.plotting import plot_RGB_DataArrays

#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD'    
# proj_dir = "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

# - Define MODIS product 
product = 'MYD021KM' # 'MOD021KM'
 
# Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Normalized_Difference_Indices" # "Normalized_Bands"   # "Original_Subset"
#-----------------------------------------------------------------------------.
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
data_remapped_dir = os.path.join(proj_data_dir, archive_name, "Remapped", product)

#-----------------------------------------------------------------------------.
########################### 
### Load remapped data ####
########################### 
fpaths = [os.path.join(data_remapped_dir, p) for p in os.listdir(data_remapped_dir)]
# - Load all images
ds_list = [xr.open_zarr(fpath, chunks='auto',decode_coords = False) for fpath in fpaths]
ds_list = [ds.set_coords('time') for ds in ds_list]  
# - Create a spatio-temporal cube
ds = xr.concat(ds_list, dim = 'time')
# - Subset daytime 
ds = ds.isel(time=ds.time.dt.hour > 7)     
ds = ds.isel(time=ds.time.dt.hour <= 17)
#-----------------------------------------------------------------------------.
######################## 
### Load native data ###
######################## 
# geo_filepaths, l1b_filepaths = find_MODIS_L1B_paths(data_dir = satellite_data_dir,
#                                                     satellite = satellite,
#                                                     product = product,
#                                                     collection = collection)
# l1b_fpath = l1b_filepaths[0]
# geo_fpath = geo_filepaths[0]
# scn = Scene(reader='modis_l1b', filenames=[l1b_fpath, geo_fpath)
# # Define channels 
# channels = [str(i) for i in range(1, 39)] 
# channels[12] = '13lo'
# channels[13] = '14lo'
# channels[36] = '13hi'
# channels[37] = '14hi'
# # Read data at 1 km resolution 
# scn.load(channels, resolution = 1000)
# # Retrieve xarray dataset 
# ds = scn.to_xarray_dataset()
# ds = ds.drop('crs')

#-----------------------------------------------------------------------------.
#################################
### Preprocess MODIS Dataset ####
#################################
ds_preprocessed = preprocess_MODIS_ds(ds, preprocessing_option=preprocessing_name)
channels_selected = list(ds_preprocessed.data_vars.keys())
#-----------------------------------------------------------------------------.
####################
### Load models ####
####################
##----------------------------------------------------------------------------.
### - Load PCA 3D model
pca3D_filepath = os.path.join(model_dir,'pca3D.sav')
pca3D = joblib.load(pca3D_filepath)
pca3D_RGB_scaler = joblib.load(os.path.join(model_dir,'scaler_pca3D_RGB.sav'))

##----------------------------------------------------------------------------.
### - Load UMAP 3D model
PCA_preprocessing = True
PCA_n_components = 20
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"

umap3D = load_UMAP(model_dir = model_dir, 
                   n_components = 3,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   metric = metric,
                   PCA_preprocessing = PCA_preprocessing, 
                   PCA_n_components = PCA_n_components)

##----------------------------------------------------------------------------.
### - Load ParametricUMAP 3D model
PCA_preprocessing = True
PCA_n_components = 20
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"

NN_umap3D = load_ParametricUMAP(model_dir = model_dir,
                                n_components = 3,
                                n_neighbors = n_neighbors, 
                                min_dist = min_dist, 
                                metric = metric,
                                PCA_preprocessing = PCA_preprocessing, 
                                PCA_n_components = PCA_n_components)

##----------------------------------------------------------------------------.
### - Load MinMax Scaler 
minmax_scaler = joblib.load(os.path.join(model_dir,'scaler_minmax.sav'))

#-----------------------------------------------------------------------------. 
###################################
### Create latent RGB composite ###
###################################
# - Reshape to pandas Dataframe 
df_pandas = ds_preprocessed.to_dataframe()
##-----------------------------------------------------------------------.
dict_channels = {c: "ch_" + c for c in channels_selected}
df_pandas = df_pandas.rename(columns=dict_channels)
# - Subset the channels 
df_pandas = df_pandas[dict_channels.values()]
df_pandas = df_pandas.dropna(how='any')
# - Standardize variables 
df_pandas_std = minmax_scaler.transform(df_pandas)
##------------------------------------------------------------------------.
# - Project into 3D latent space  and assign color 
pca3D_latent = pca3D.transform(df_pandas_std) 
pca3D_RGB_color = pca3D_RGB_scaler.transform(pca3D_latent)

umap3D_RGB_color = umap3D.transform_to_RGB(df_pandas_std) # this is slow !
NN_umap3D_RGB_color = NN_umap3D.transform_to_RGB(df_pandas_std)    

# - Reshape to RGB DataArray
da_PCA = reshape_to_RGB_composite(df = df_pandas, 
                                  RGB_matrix = pca3D_RGB_color, 
                                  # dims = ds.dims.keys(), 
                                  name = "PCA_Composite")

da_UMAP = reshape_to_RGB_composite(df = df_pandas, 
                                   RGB_matrix = umap3D_RGB_color, 
                                   # dims = ds.dims.keys(), 
                                   name = "UMAP_Composite")

da_NN_UMAP = reshape_to_RGB_composite(df = df_pandas, 
                                   RGB_matrix = NN_umap3D_RGB_color, 
                                   # dims = ds.dims.keys(), 
                                   name = "NN_UMAP_Composite")

#-----------------------------------------------------------------------------.
######################################
### Display composite with cartopy ###
######################################
# - Plot single image for each timestep
for t in range(0, len(da_PCA.time)):
    plot_RGB_DataArray(da_PCA, timestep=t)

for t in range(0, len(da_UMAP.time)):
    plot_RGB_DataArray(da_UMAP, timestep=t)

for t in range(0, len(da_NN_UMAP.time)):
    plot_RGB_DataArray(da_NN_UMAP, timestep=t)
    
# - Plot multiple RGB images   
plot_RGB_DataArrays(da_PCA.isel(time=slice(0,16)),  title="PCA RGB")
plot_RGB_DataArrays(da_PCA.isel(time=slice(16,32)), title="PCA RGB")
plot_RGB_DataArrays(da_UMAP.isel(time=slice(0,16)), title="UMAP RGB")  
plot_RGB_DataArrays(da_UMAP.isel(time=slice(16,32)), title="UMAP RGB")
plot_RGB_DataArrays(da_NN_UMAP.isel(time=slice(0,16)), title="ParametricUMAP RGB") 
plot_RGB_DataArrays(da_NN_UMAP.isel(time=slice(16,32)), title="ParametricUMAP RGB") 
