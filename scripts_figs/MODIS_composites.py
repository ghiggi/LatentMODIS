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
from trollimage.xrimage import XRImage
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from matplotlib import colors
import matplotlib.gridspec as gridspec
import numpy as np
from latentpy.preprocessing import preprocess_MODIS_ds
from latentpy.dimred import load_ParametricUMAP
from latentpy.utils_reshape import reshape_to_RGB_composite
from latentpy.plotting import plot_RGB_DataArrays
 
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD' 
figs_dir = '/ltenas3/LatentMODIS/figs'   
# proj_dir = "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

# - Define MODIS product 
satellite = 'Aqua'   # 'Terra'
product = 'MYD021KM' # 'MOD021KM'
collection = '61'
 
# Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Normalized_Difference_Indices"  
#-----------------------------------------------------------------------------.
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
data_remapped_dir = os.path.join(proj_data_dir, archive_name, "Remapped", product)

#-----------------------------------------------------------------------------.
############################### 
### Load L1B remapped data ####
###############################
fpaths = [os.path.join(data_remapped_dir, p) for p in os.listdir(data_remapped_dir)]
# - Load all images
ds_list = [xr.open_zarr(fpath, chunks='auto',decode_coords = False) for fpath in fpaths]
ds_list = [ds.set_coords('time') for ds in ds_list]  
# - Create a spatio-temporal cube
ds = xr.concat(ds_list, dim = 'time')
# - Subset daytime 
ds = ds.isel(time=ds.time.dt.hour > 7)     
ds = ds.isel(time=ds.time.dt.hour <= 17)
# - Compute NDIs
ds_NDI = preprocess_MODIS_ds(ds, preprocessing_option=preprocessing_name)
channels_selected = list(ds_NDI.data_vars.keys())
# - Select NDSI
da_NDSI = ds_NDI['4_6']

############################  
### Load L2 Cloud Mask  ####
############################ 
# (0=Cloudy, 1=Uncertain, 2=Probably Clear, 3=Confident Clear)
data_remapped_MYD35_dir = os.path.join(proj_data_dir, archive_name, "Remapped", 'MYD35_L2')
fpaths = [os.path.join(data_remapped_MYD35_dir, p) for p in os.listdir(data_remapped_MYD35_dir)]
# - Load all images
ds_list = [xr.open_zarr(fpath, chunks='auto',decode_coords = False) for fpath in fpaths]
ds_list = [ds.set_coords('time') for ds in ds_list]  
# - Create a spatio-temporal cube
ds_MYD35 = xr.concat(ds_list, dim = 'time')
# - Subset daytime 
ds_MYD35 = ds_MYD35.isel(time=ds_MYD35.time.dt.hour > 7)     
ds_MYD35 = ds_MYD35.isel(time=ds_MYD35.time.dt.hour <= 17)
ds_MYD35['cloud_mask'].isel(time=slice(0,8)).plot.imshow(x='x',y='y', col="time", col_wrap=4, vmin=0, vmax=4)
# - Convert to float and set NA   
da_MYD35 = ds_MYD35['cloud_mask'].astype('float32').compute()
da_MYD35 = da_MYD35.where(da_MYD35.values != 255)
da_MYD35.isel(time=slice(0,8)).plot.imshow(x='x',y='y', col="time", col_wrap=4, vmin=0, vmax=4)

#-----------------------------------------------------------------------------.
####################################### 
### Load cloud classification data ####
#######################################
# - Retrive labels fpath
labels_dir =  os.path.join(proj_dir, "labels")

labels_PCA_fpath = os.path.join(labels_dir, "out_PCA_class_alfonso_20201231.nc")
labels_NN_umap_fpath = os.path.join(labels_dir, "out_NN_umap2d_class_alfonso_20201231.nc") 
# - Load cloud masks
ds_labels_PCA = xr.open_dataset(labels_PCA_fpath, chunks='auto')
ds_labels_NN_umap = xr.open_dataset(labels_NN_umap_fpath, chunks='auto')
 
ds_cloud_mask = xr.merge([ds_labels_PCA, ds_labels_NN_umap])
# - Subset daytime 
ds_cloud_mask = ds_cloud_mask.isel(time=ds_cloud_mask.time.dt.hour > 7)     
ds_cloud_mask = ds_cloud_mask.isel(time=ds_cloud_mask.time.dt.hour <= 17)

#-----------------------------------------------------------------------------.
############################## 
### Create RGB Composites ####
############################## 
### - Create True color composite
da_TrueColor = ds[['1','4','3']].to_stacked_array(new_dim="bands", 
                                                       sample_dims=['x','y','time'],
                                                       name="True Color")
da_TrueColor = da_TrueColor/100
im = XRImage(da_TrueColor)
im.stretch_hist_equalize()
da_TrueColor = im.data.compute() 

#-----------------------------------------------------------------------------.
### - Create 'Blue/SWIR1/SWIR2s composite
da_SnowCloud = ds[['3','6','7']].to_stacked_array(new_dim="bands", 
                                                  sample_dims=['x','y','time'],
                                                  name="True Color")
da_SnowCloud = da_SnowCloud/100
im = XRImage(da_SnowCloud)
im.stretch_hist_equalize()
da_SnowCloud = im.data.compute() 

#-----------------------------------------------------------------------------.
##########################################
### Create PCA and UMAP RGB Composites ###
##########################################
##----------------------------------------------------------------------------.
## - Load PCA 3D model
pca3D_filepath = os.path.join(model_dir,'pca3D.sav')
pca3D = joblib.load(pca3D_filepath)
pca3D_RGB_scaler = joblib.load(os.path.join(model_dir,'scaler_pca3D_RGB.sav'))

##----------------------------------------------------------------------------.
## - Load ParametricUMAP 3D model
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
## - Load MinMax Scaler 
minmax_scaler = joblib.load(os.path.join(model_dir,'scaler_minmax.sav'))

#-----------------------------------------------------------------------------. 
## - Create latent RGB composite  
# - Reshape to pandas Dataframe 
df_pandas = ds_NDI.to_dataframe()
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
NN_umap3D_RGB_color = NN_umap3D.transform_to_RGB(df_pandas_std)    

# - Reshape to RGB DataArray
da_PCA = reshape_to_RGB_composite(df = df_pandas, 
                                  RGB_matrix = pca3D_RGB_color, 
                                  # dims = ds.dims.keys(), 
                                  name = "PCA_Composite")

da_NN_UMAP = reshape_to_RGB_composite(df = df_pandas, 
                                   RGB_matrix = NN_umap3D_RGB_color, 
                                   # dims = ds.dims.keys(), 
                                   name = "NN_UMAP_Composite")

#-----------------------------------------------------------------------------.
############################ 
#### Select 4 timesteps ####
############################ 

# plot_RGB_DataArrays(da_PCA.isel(time=slice(0,16)),  title="PCA RGB")
time_dim = "time"
timesteps = [2,11,12,22]
time_str = da_PCA.time.values[timesteps]

# plot_RGB_DataArrays(ds_cloud_mask['pca2D_class'].sel(time=time_str), title="PCA Cloud Mask")
# plot_RGB_DataArrays(ds_cloud_mask['NN_umap2D_class'].sel(time=time_str), title="PCA Cloud Mask")
# plot_RGB_DataArrays(da_MYD35.sel(time=time_str),  title="MYD35 Cloud Mask")
# plot_RGB_DataArrays(da_NDSI.sel(time=time_str), title="NDSI")
# plot_RGB_DataArrays(da_TrueColor.sel(time=time_str), title="TrueColor")
# plot_RGB_DataArrays(da_SnowCloud.sel(time=time_str), title="Snow Cloud")
# plot_RGB_DataArrays(da_PCA.sel(time=time_str),  title="PCA RGB")
# plot_RGB_DataArrays(da_NN_UMAP.sel(time=time_str),  title="ParametricUMAP RGB")
#-----------------------------------------------------------------------------.
########################### 
### Create the figure   ###
###########################
# - Define map settings
PE_lat, PE_lon = -71.949944, 23.347079
crs_WGS84 = ccrs.PlateCarree()  
crs_proj = ccrs.SouthPolarStereo()
# - Define cloud mask colors for PolyClassifier
cmap_cloud_mask = colors.ListedColormap(['dodgerblue','slategray'])
cmap_cloud_mask_bounds=[0,1,2]
cmap_cloud_mask_norm = colors.BoundaryNorm(cmap_cloud_mask_bounds, cmap_cloud_mask.N)
# ds_cloud_mask['pca2D_class'].isel(time=1).plot.imshow(add_colorbar=False, cmap=cmap_cloud_mask, norm=cmap_cloud_mask_norm)

# - Define cloud mask colors for MYD35
cmap_MYD35 = colors.ListedColormap(['slategray', 'darkgrey','lightskyblue', 'dodgerblue'])
cmap_MYD35_bounds=[0,1,2,3,4]
cmap_MYD35_norm = colors.BoundaryNorm(cmap_MYD35_bounds, cmap_MYD35.N)
# da_MYD35.isel(time=1).plot.imshow(add_colorbar=False, cmap=cmap_MYD35, norm=cmap_MYD35_norm)

# - Define figure options
figsize = (12,18) # (width, height) [in inches]
fig = plt.figure(figsize = figsize, dpi = 600) 
# - Define  plot layout 
ncols = 4
nrows = 8
n_plots = ncols * nrows
l_GridSpec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
l_ax = [None for i in range(n_plots)]
##------------------------------------------------------------------------.
### - Plot True Color 
da = da_TrueColor
r_i = 0 
ylabel = "True Color"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    # - Select title
    tmp_title = np.datetime_as_string(da_tmp[time_dim].values, unit='s')   
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0) # hspace=0,
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i])
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
       
##-----------------------------------------------------------------------------.
### - Plot Snow Cloud 
da = da_SnowCloud
r_i = 1 
ylabel = "Blue / SWIR1 / SWIR2"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    # - Select title
    tmp_title = ""   
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0,hspace=0) 
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i])
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

##-----------------------------------------------------------------------------.
### - Plot PCA RGB
da = da_PCA
r_i = 2 
ylabel = "PCA RGB"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    # - Select title
    tmp_title = ""  
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0,hspace=0) 
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i])
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

##-----------------------------------------------------------------------------.
### - Plot ParametricUMAP RGB
da = da_NN_UMAP
r_i = 3
ylabel = "ParametricUMAP RGB"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    # - Select title
    tmp_title = ""   
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0, hspace=0) 
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i])
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False    
    
##-----------------------------------------------------------------------------.
### - Plot NDSI  
da = da_NDSI
r_i = 4
ylabel = "NDSI"
vmin = -0.8
vmax = 0.8
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    tmp_title = ""
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0, hspace=0)  
    # - Plot the image 
    if (j != ncols-1):
        da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i],
                            add_colorbar = False,
                            vmin=vmin, vmax=vmax)                  
    else:
        da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i],
                            add_colorbar = False,
                            #  cbar_kwargs={'label': ''},
                            vmin=vmin, vmax=vmax)
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False 
##-----------------------------------------------------------------------------.
### - Plot PCA Cloud Mask 
da = ds_cloud_mask['pca2D_class']
r_i = 5
ylabel = "Cloud mask - PCA"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    tmp_title = ""
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0, hspace=0)  
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i],
                       add_colorbar = False, cmap=cmap_cloud_mask, norm=cmap_cloud_mask_norm)                     
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

##-----------------------------------------------------------------------------.
### - Plot ParametricUMAP Cloud Mask 
da = ds_cloud_mask['NN_umap2D_class']
r_i = 6
ylabel = "Cloud mask - ParametricUMAP"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    tmp_title = ""
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0, hspace=0)  
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i],
                       add_colorbar = False, cmap=cmap_cloud_mask, norm=cmap_cloud_mask_norm)
                                            
                            
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

##----------------------------------------------------------------------------.
### - Plot MODIS Cloud Mask 
da = da_MYD35
r_i = 7
ylabel = "Cloud mask - MYD35"
for j in range(0, ncols):
    # - Compute ax index
    i = r_i*ncols + j
    # - Select timestep
    da_tmp = da.sel({time_dim: time_str[j]})
    tmp_title = ""
    # - Create subplot
    l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
    l_ax[i].get_gridspec().update(wspace=0, hspace=0)  
    # - Plot the image 
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i],
                       add_colorbar = False, cmap=cmap_MYD35, norm=cmap_MYD35_norm)
                                            
                            
    l_ax[i].set_title(tmp_title)
    if (j == 0):
        l_ax[i].text(-0.07, 0.50, ylabel, va='bottom', ha='center',
                     rotation='vertical', rotation_mode='anchor',
                     transform=l_ax[i].transAxes)
    # - Add coastlines
    l_ax[i].coastlines() 
    # - Add PE marker 
    l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84, s=10)
    # - Add gridlines 
    gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = False

fig.tight_layout()
fig.savefig(fname=os.path.join(figs_dir, "MODIS_composites.png"))
 
    


 