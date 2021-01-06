#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:46:42 2020

@author: ghiggi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:27:35 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import zarr
import datetime
import xarray as xr 
import numpy as np
import cartopy.crs as ccrs
import satpy 
from satpy import Scene
from pyresample import create_area_def

from latentpy.utils_crs import adapt_bbox_for_resolution
from latentpy.utils_crs import get_projected_bbox
from latentpy.laads_utils import find_MODIS_L1B_paths
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data" 
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD'  
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

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
# - Define directory where to save remapped data
data_remapped_dir = os.path.join(proj_data_dir, archive_name, "Remapped", product)
if not os.path.exists(data_remapped_dir):
    os.makedirs(data_remapped_dir)
##-----------------------------------------------------------------------------.
# Define file to keep track of processed files
logs_processed_l1b_fname = "".join([product,"_remapped",".txt"])
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
# bbox = (W S E N)
if archive_name == "archive_PE_bbox":
    bbox = (22.21, -72.52, 28.37, -71.57) # PE bounding bbox 

if archive_name == "archive_Antarctica":
    bbox = (-180, -90, 180, -70)  # buggy
    bbox = (-179, -90, 179, -70) 
    bbox = (-100, -74, -50, -70)     

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
### Remap all granules 
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
    # Remove crs coordinate 
    ds = ds.drop('crs')
    # Add time coordinate 
    ds = ds.assign_coords(time=(ds.start_time))
    #-------------------------------------------------------------------------.
    # Check if there are some values inside 
    idx_all_na = [np.alltrue(np.isnan(ds[var].values)) for var in ds.data_vars.keys()]
    empty_ds = np.alltrue(idx_all_na).tolist()
    #-------------------------------------------------------------------------.
    # If no data inside the bounding box
    if (empty_ds is True):
        print(os.path.basename(l1b_filepaths[i_f]), "does not have non NaN data inside the bbox")
    else:
        # Define zarr name 
        zarr_fname = product + "_Remapped_" + ds.start_time.strftime('%Y%m%d_%H%M%S') + '.zarr'
        zarr_fpath = os.path.join(data_remapped_dir, zarr_fname)
        # Discard datetime attributes 
        for var in ds.data_vars.keys():
            # da[var].attrs = {}
            _ = ds[var].attrs.pop('start_time', None)
            _ = ds[var].attrs.pop('end_time', None)
            _ = ds[var].attrs.pop('area', None)
            _ = ds[var].attrs.pop('_satpy_id', None)
        _ = ds.attrs.pop('start_time', None)
        _ = ds.attrs.pop('end_time', None)
        _ = ds.attrs.pop('area', None)
        _ = ds.attrs.pop('_satpy_id', None)
        # Save the dataset as zarr 
        # - If a DataArray is a dask array, it is written with those chunks. 
        zarr_store = zarr.DirectoryStore(zarr_fpath)
        ds.to_zarr(store=zarr_store, mode='w',
                   synchronizer=None, group=None, 
                   encoding = None, compute=True)               
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
