#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:29:07 2020

@author: ghiggi
"""
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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from latentpy.dimred import load_UMAP
from latentpy.dimred import load_ParametricUMAP
from latentpy.preprocessing import preprocess_MODIS_ds 
from latentpy.plotting import plot_DataArrayPanels

##-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD'    
# proj_dir = "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

# - Define MODIS product 
satellite = 'Aqua'   # 'Terra'
product = 'MYD021KM' # 'MOD021KM'
collection = '61'
 
# Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Normalized_Difference_Indices" # "Normalized_Bands"   # "Original_Subset"

# Define output label directory
out_label_dir = "/ltenas3/LatentMODIS/labels"

##-----------------------------------------------------------------------------.
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
print(ds_list[-1].time)
# - Create a spatio-temporal cube
ds = xr.concat(ds_list, dim = 'time')
# - Subset daytime 
ds = ds.isel(time=ds.time.dt.hour > 7)     
ds = ds.isel(time=ds.time.dt.hour <= 17)
print(ds)

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
### - Load PCA models
pca2D_filepath = os.path.join(model_dir,'pca2D.sav')
pca2D = joblib.load(pca2D_filepath)

##----------------------------------------------------------------------------.
### - Load UMAP 2D model
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

##----------------------------------------------------------------------------.
### - Load ParametricUMAP 2D model
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

##----------------------------------------------------------------------------.
### - Load MinMax Scaler 
minmax_scaler = joblib.load(os.path.join(model_dir,'scaler_minmax.sav'))

##----------------------------------------------------------------------------.
### - Load PolyClassifiers   
pca2D_PolyClassifier = joblib.load(os.path.join(model_dir,
                                                'PCA_PolyClassifier_alfonso_20201223.sav'))
#umap2D_PolyClassifier = joblib.load(os.path.join(model_dir,'UMAP_PolyClassifier.sav'))
NN_umap2D_PolyClassifier = joblib.load(os.path.join(model_dir,
                                                    'ParametricUMAP_PolyClassifier_alfonso_20201231.sav'))

#-----------------------------------------------------------------------------. 
############################ 
### Apply classification ###
############################ 
# - Reshape to pandas Dataframe 
df_pandas = ds_preprocessed.to_dataframe()
# - Subset the channels 
df_pandas = df_pandas[channels_selected]
df_pandas = df_pandas.dropna(how='any')
# - Standardize variables 
df_pandas_std = minmax_scaler.transform(df_pandas)

##----------------------------------------------------------------------------.
# - Project into 2D latent space 
pca2D_latent = pca2D.transform(df_pandas_std) 
umap2D_latent = umap2D.transform(df_pandas_std) # this is slow !
NN_umap2D_latent = NN_umap2D.transform(df_pandas_std)

# - Predict class  
df_pandas["pca2D_class"] = pca2D_PolyClassifier.predict(pca2D_latent, option = 'labels')
#df_pandas["umap2D_class"] = umap2D_PolyClassifier.predict(umap2D_latent, option = 'labels')
df_pandas["NN_umap2D_class"] = NN_umap2D_PolyClassifier.predict(NN_umap2D_latent, option = 'labels')

##----------------------------------------------------------------------------.
# - Reshape to xarray Dataset
ds_classified = df_pandas.to_xarray() 
#----------------------------------------------------------------------------.
######################################
### Display classes with cartopy   ###
######################################  
da_NN_umap2d_class = ds_classified['NN_umap2D_class']
da_PCA_class = ds_classified['pca2D_class']

levels = [0, 0.9, 1.1]
plot_DataArrayPanels(da_NN_umap2d_class.isel(time=slice(0,16)), 
                     loop_over = "time",
                     title="Parametric UMAP Classification",
                     figsize=(16,14),
                     levels=levels)

plot_DataArrayPanels(da_NN_umap2d_class.isel(time=slice(16,32)), 
                     loop_over = "time",
                     title="Parametric UMAP Classification",
                     figsize=(16,5.5),
                     levels=levels)

plot_DataArrayPanels(da_PCA_class.isel(time=slice(0,16)), 
                     loop_over = "time",
                     title="PCA Classification",
                     figsize=(16,14),
                     levels=levels)

plot_DataArrayPanels(da_PCA_class.isel(time=slice(16,32)), 
                     loop_over = "time",
                     title="PCA Classification",
                     figsize=(16,5.5),
                     levels=levels)
##----------------------------------------------------------------------------.
##############################
### Save to file (netCDF)  ###
##############################
out_NN_umap2d_class_fpath = os.path.join(out_label_dir,
                                         'out_NN_umap2d_class_alfonso_20201231.nc')
da_NN_umap2d_class.to_netcdf(out_NN_umap2d_class_fpath)

out_PCA_class_fpath = os.path.join(out_label_dir,
                                   'out_PCA_class_alfonso_20201231.nc')
da_PCA_class.to_netcdf(out_PCA_class_fpath)

