#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:27:48 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import xarray as xr 
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import napari
from trollimage.xrimage import XRImage

from latentpy.preprocessing import normalize_MODIS_bands_ds
from latentpy.preprocessing import NDI_MODIS_bands_ds

from latentpy.plotting import plot_hourly_variability_boxplot
from latentpy.plotting import plot_variability_boxplot
from latentpy.plotting import plot_bands
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD'
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

# - Define archive name 
product = 'MYD021KM' # 'MOD021KM'
archive_name = "archive_PE_bbox"

daytime_channels = ["1","2", "3", "4", "5", "6", "7", "8", "17", "18", "19", "26"]

IR_channels = ["20", "22", "24",  # '23', '25',  
               "27", "28", "29", "31", "32", "33", "34", "35"] 

#-----------------------------------------------------------------------------.
### Create space-time multispectral cube 
# - Define data folder 
data_remapped_dir = os.path.join(proj_data_dir, archive_name, "Remapped", product)
fpaths = [os.path.join(data_remapped_dir, p) for p in os.listdir(data_remapped_dir)]
# - Load all images
ds_list = [xr.open_zarr(fpath, chunks='auto',decode_coords = False) for fpath in fpaths]
ds_list = [ds.set_coords('time') for ds in ds_list]  
# - Create a spatio-temporal cube
ds = xr.concat(ds_list, dim = 'time')
# - Subset daytime 
ds = ds.isel(time=ds.time.dt.hour > 7)     
ds = ds.isel(time=ds.time.dt.hour <= 17)
# - Retrieve number of overpass
n_overpass = ds.dims['time']
#-----------------------------------------------------------------------------.
### EDA variability on original Reflectance/Brightness Temperature
# - Hourly Boxplots
for ch in daytime_channels:
    plot_hourly_variability_boxplot(da=ds[ch], ylab='Reflectance')

for ch in IR_channels:
    plot_hourly_variability_boxplot(da=ds[ch], ylab='Brightness Temperature')

# - Overpass Boxplots
for ch in daytime_channels:
    plot_variability_boxplot(ds[ch], ylab='Reflectance')

for ch in IR_channels:
    plot_variability_boxplot(ds[ch], ylab='Brightness Temperature')

# Maps 
for i in range(0,n_overpass):
    plot_bands(ds=ds, 
               bands = daytime_channels, 
               timestep = i,
               fname=None)
#-----------------------------------------------------------------------------.
### EDA variability on normalized radiance   
VIS_chs = ["1", "2", "3", "4", "5", "17", "18", "19", '6','7']
IR_chs = ['22','23','32','33','34','35']

ds_normalized = normalize_MODIS_bands_ds(ds)
## VIS
# - Hourly Boxplots
for ch in VIS_chs:
    plot_hourly_variability_boxplot(ds_normalized[ch])

# - Overpass Boxplots
for ch in VIS_chs:
    plot_variability_boxplot(ds_normalized[ch])
    
# - Maps 
for i in range(0,n_overpass):
    plot_bands(ds=ds_normalized, 
               bands = VIS_chs, 
               timestep = i,
               fname=None)

## IR
# - Hourly Boxplots
for ch in IR_chs:
    plot_hourly_variability_boxplot(ds_normalized[ch])

# - Overpass Boxplots
for ch in IR_chs:
    plot_variability_boxplot(ds_normalized[ch])
    
# - Maps 
for i in range(0,n_overpass):
    plot_bands(ds=ds_normalized, 
               bands = IR_chs, 
               timestep = i,
               fname=None)
    
#-----------------------------------------------------------------------------. 
### EDA Variability of Normalized Difference Snow Index (NDSI) 
# A pixel with NDSI > 0.0 is considered to have some snow present. 
# A pixel with NDSI <= 0.0 does not have snow present 
NDSI_da = (ds['4']-ds['6']) / (ds['4'] + ds['6'])
NDSI_da.name = 'NDSI'
NDSI_ds = NDSI_da.to_dataset()   

plot_hourly_variability_boxplot(NDSI_da)
plot_variability_boxplot(NDSI_da)

for i in range(0,n_overpass):
    plot_bands(ds=NDSI_ds, 
               bands = ['NDSI'], 
               timestep = i,
               fname=None)                  
 
#-----------------------------------------------------------------------------.
### EDA Variability of Normalized Difference Indeces (NDI) 
ds_NDI = NDI_MODIS_bands_ds(ds)
NDI_names = list(ds_NDI.data_vars.keys())
# - Hourly Boxplots
for ch in NDI_names[0:5]:
    plot_hourly_variability_boxplot(ds_NDI[ch])

# - Overpass Boxplots
for ch in NDI_names[0:5]:
    plot_variability_boxplot(ds_NDI[ch])
    
# - Maps 
for i in range(0,n_overpass):
    plot_bands(ds=ds_NDI, 
               bands = NDI_names[0:20], 
               timestep = i,
               title_prefix="NDI ",
               fname=None) 

for i in range(0,n_overpass):
    plot_bands(ds=ds_NDI, 
               bands = NDI_names[20:40], 
               timestep = i,
               title_prefix="NDI ",
               fname=None) 
    
for i in range(0,n_overpass):
    plot_bands(ds=ds_NDI, 
               bands = NDI_names[40:60], 
               timestep = i,
               title_prefix="NDI ",
               fname=None) 