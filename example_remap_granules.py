#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:00:18 2020

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/LatentMODIS")
import xarray as xr 
import pandas as pd
import numpy as np
import dask 
import vaex 
import satpy 
import cartopy.crs as ccrs
from satpy import Scene
from pyresample import create_area_def

from latentpy.utils_crs import adapt_bbox_for_resolution
from latentpy.utils_crs import get_projected_bbox
from latentpy.laads_utils import find_MODIS_L1B_paths
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
satellite_data_dir = '/home/ghiggi/Data/'
# - Define MODIS product 
satellite = 'Aqua'   # 'Terra'
product = 'MYD021KM' # 'MOD021KM'
collection = '61'
# - Define area of interest 
archive_name = "archive_PE_bbox"

#-----------------------------------------------------------------------------.
### Define gridded area into which to reproject 
bbox = (22.21, -72.52, 28.37, -71.57) # PE bounding bbox (LON_MIN,LAT_MIN, LON_MAX, LAT_MAX)
bbox = (-100, -74, -50, -70)   
##-----------------------------------------------------------------------------.
# - Define the CRS of the original data
crs_WGS84 = ccrs.PlateCarree(central_longitude=0) # reference CRS of Swath data
# - Define appropriate CRS for the study area 
crs_proj = ccrs.SouthPolarStereo() # 'Antarctic EASE grid'
description='Antarctic EASE grid'
proj4_string = crs_proj.proj4_init # Extract PROJ4 string from cartopy CRS
# - Project bbox into the new CRS
bbox_proj = get_projected_bbox(bbox, crs_ref = crs_WGS84, crs_proj = crs_proj)
# - Specify the resolution of the grid
bbox_proj = adapt_bbox_for_resolution(bbox_proj, resolution=1000)
# - Define pyresample AreaDefinition 
areaDef = create_area_def(area_id = 'ease_sh',
                          description = description,
                          projection = proj4_string, 
                          units = 'meters',
                          area_extent = bbox_proj, 
                          resolution = '1000')

### Retrieve filepaths of MODIS L1B data
geo_filepaths, l1b_filepaths = find_MODIS_L1B_paths(data_dir = satellite_data_dir,
                                                    satellite = satellite,
                                                    product = product,
                                                    collection = collection)
#-----------------------------------------------------------------------------.
# Load data
scn = Scene(reader='modis_l1b', filenames=[l1b_filepaths[0], geo_filepaths[0]])
scn.available_dataset_names()
# Define channels 
channels = [str(i) for i in range(1, 39)] 
channels[12] = '13lo'
channels[13] = '14lo'
channels[36] = '13hi'
channels[37] = '14hi'
scn.load(channels, resolution = 1000)
#-----------------------------------------------------------------------------.
### Crop and remap the data 
remapped_scn = scn.resample(areaDef,
                            resampler='bilinear',
                            radius_of_influence=5000)
### Create xarray dataset 
ds = remapped_scn.to_xarray_dataset()
# ds = scn.to_xarray_dataset()
#-----------------------------------------------------------------------------.
PE_lat, PE_lon = -71.949944, 23.347079

### Plot remapped data
import matplotlib.pyplot as plt
plt.figure()
ax = plt.subplot(projection=crs_proj)
im = ax.pcolormesh(ds['x'].values,
                   ds['y'].values, 
                   ds['1'].values,
                   transform=crs_proj,
                   linewidth=0)
ax.coastlines()  

## - Add PE marker 
ax.scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
ax.coastlines()  
## - Add gridlines 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--') 
gl.xlabels_bottom = False
gl.xlabels_left = False
gl.xlabels_right = False
gl.ylabels_right = True
plt.show() 

# Using xarray plotting method
plt.figure()
ax = plt.subplot(projection=crs_proj)
ds['1'].plot.pcolormesh('x','y',# '1',
                        transform=crs_proj,
                        linewidth=0)
ax.coastlines()   
plt.show() 

