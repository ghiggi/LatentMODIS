#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:27:35 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import xarray as xr 
import pandas as pd
import numpy as np
import dask 
import vaex 
import satpy 
import datetime
import cartopy.crs as ccrs
from satpy import Scene
from pyresample import create_area_def

from latentpy.utils_crs import adapt_bbox_for_resolution
from latentpy.utils_crs import get_projected_bbox
from latentpy.laads_utils import find_MODIS_L1B_paths
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS" # "/home/ghiggi/Projects/LatentMODIS"
proj_data_dir = "/ltenas3/LatentMODIS/data" #"/home/ghiggi/Projects/LatentMODIS/data"
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD' #'/home/ghiggi/Data/' 
# - Define MODIS product 
satellite = 'Aqua'   # 'Terra'
product = 'MYD021KM' # 'MOD021KM'
collection = '61'
# - Define area of interest 
archive_name = "archive_PE_bbox"
# - Define logs 
VERBOSE = True
PRINT_EVERY = 1
#-----------------------------------------------------------------------------.
### Define directories and logs files 
# - Define directory to save vaex archives
tabular_data_dir = os.path.join(proj_data_dir, archive_name, "Remapped_Reshaped", product)
if not os.path.exists(tabular_data_dir):
    os.makedirs(tabular_data_dir)
##-----------------------------------------------------------------------------.
# Define file to keep track of processed files
logs_processed_l1b_fname = "".join([product,"_remapped_reshaped",".txt"])
logs_processed_l1b_dir_path = os.path.join(proj_data_dir, archive_name, "logs")
if not os.path.exists(logs_processed_l1b_dir_path):
    os.makedirs(logs_processed_l1b_dir_path)
logs_processed_l1b_fpath = os.path.join(logs_processed_l1b_dir_path, logs_processed_l1b_fname)
if not os.path.exists(logs_processed_l1b_fpath):
    with open(logs_processed_l1b_fpath, 'w') as f:
        f.write('\n')
        
#-----------------------------------------------------------------------------.
### Define gridded area into which to reproject 
# - Define study area bounding box 
# bbox = (LON_MIN,LAT_MIN, LON_MAX, LAT_MAX)
# bbox = (xmin, ymin, xmax, ymax) 
if archive_name == "archive_PE_bbox":
    bbox = (22.21, -72.52, 28.37, -71.57) # PE bounding bbox 

if archive_name == "archive_Antarctica":
    bbox = (-180, -90, 180, -70)  
    bbox = (-179, -90, 179, -70) 
##-----------------------------------------------------------------------------.
# - Define the CRS of the original data
crs_WGS84 = ccrs.PlateCarree(central_longitude=0) # reference CRS of Swath data
# - Define appropriate CRS for the study area 
crs_proj = ccrs.SouthPolarStereo() # 'Antarctic EASE grid definition?'
description='South Pole Stereographic'
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

#-----------------------------------------------------------------------------.
### Retrieve filepaths of MODIS L1B data
geo_filepaths, l1b_filepaths = find_MODIS_L1B_paths(data_dir = satellite_data_dir,
                                                    satellite = satellite,
                                                    product = product,
                                                    collection = collection)
if VERBOSE:
    print('Found %d files' % len(geo_filepaths))

#-----------------------------------------------------------------------------.
### Process all granules 
for i_f in range(len(l1b_filepaths)):
    #-------------------------------------------------------------------------.  
    # Print progress 
    if VERBOSE and (not i_f % PRINT_EVERY):
        print('%d / %d' % (i_f, len(l1b_filepaths)))
        print("Processing", os.path.basename(l1b_filepaths[i_f]))
        t_i = datetime.datetime.now()
    #-------------------------------------------------------------------------.  
    # Checking if the file has already been processed
    with open(logs_processed_l1b_fpath, 'r') as f:
        processed_L1B_list = f.readlines()
    if "".join([l1b_filepaths[i_f],'\n']) in processed_L1B_list:
        # print('This granule was already processed')
        continue
    else:
        pass
    #-------------------------------------------------------------------------.
    # Load data
    scn = Scene(reader='modis_l1b', filenames=[l1b_filepaths[i_f], geo_filepaths[i_f]])
    # scn.available_dataset_names()
    # Define channels 
    channels = [str(i) for i in range(1, 39)] 
    channels[12] = '13lo'
    channels[13] = '14lo'
    channels[36] = '13hi'
    channels[37] = '14hi'
    # Read data at 1 km resolution 
    scn.load(channels, resolution = 1000)
    # Remap the data 
    remapped_scn = scn.resample(areaDef,
                                resampler='nearest',
                                radius_of_influence=5000)
    #-------------------------------------------------------------------------.
    # Retrieve xarray dataset 
    ds = remapped_scn.to_xarray_dataset()
    # Drop crs 
    ds = ds.drop('crs')
    # Add row/col idx 
    ds = ds.assign_coords(idx_x=("x", np.arange(ds.x.shape[0])))
    ds = ds.assign_coords(idx_y=("y", np.arange(ds.y.shape[0])))
    #-------------------------------------------------------------------------.
    # Convert to pandas  
    df_pandas = ds.to_dataframe()
    # - Remove crs column
    # df_pandas = df_pandas.drop('crs', 1)   
    # - Remove rows with all NaN 
    df_pandas = df_pandas.dropna(how='all', subset=channels)
    # If no rows are remaining after discarding rows with all NaN
    if (df_pandas.empty is True):
        print(os.path.basename(l1b_filepaths[i_f]), "does not have non NaN data inside the bbox")
    else:
        # - Remove MultiIndex (for conversion to vaex)
        # df_pandas.index 
        df_pandas.reset_index(inplace=True)  # add x y as columns 
        # df_pandas.index
        # - Add overpass time 
        df_pandas['overpass_time'] = ds.attrs['start_time']  # 5 min granules ... 
        # - Rename channels (because vaex bug with '<number>' column names ) 
        dict_channels = {c: "ch_" + c for c in channels}
        df_pandas = df_pandas.rename(columns=dict_channels)
        #-------------------------------------------------------------------------.
        # If any point in the selected area:
        if (df_pandas.empty is True):
            print(os.path.basename(l1b_filepaths[i_f]), " does not have data inside the bbox")
        else:
            # Convert to vaex
            df_vaex = vaex.from_pandas(df_pandas, copy_index=False)       
            # Save the dataframe as hdf5
            out_fname = product + "_Remapped_Reshaped_" + ds.start_time.strftime('%Y%m%d_%H%M%S') + '.hdf5'
            out_fpath = os.path.join(tabular_data_dir, out_fname)
            df_vaex.export_hdf5(out_fpath)
    #-------------------------------------------------------------------------.
    # Add processed granules to the logs         
    with open(logs_processed_l1b_fpath, 'a') as f:
        f.write("".join([l1b_filepaths[i_f],'\n']))
    #-------------------------------------------------------------------------.
    # Display execution time 
    if VERBOSE: 
        t_f = datetime.datetime.now()
        dt = t_f - t_i 
        print("Processed in ", dt.seconds, "seconds")
#-----------------------------------------------------------------------------.
