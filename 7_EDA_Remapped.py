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
archive_name = "archive_PE_bbox"
product = "MYD021KM"
# - Define channels 
channels_solar = ['1','2','3','4','5','7','17','18','19','26']
channels_thermal = ['20','21','22','23','25','27','28','29','31','32','33','34']
channels_selected = channels_solar + channels_thermal

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

#-----------------------------------------------------------------------------.
######################
### Plot one band ####
######################
# - Select bands
# 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 26
# 20, 22, 24, 27, 28, 29, 31, 32, 33, 34, 35
band = '35'

# - Select timestep
timestep = 1

# - Figure options
PE_lat, PE_lon = -71.949944, 23.347079
crs_WGS84 = ccrs.PlateCarree()  
crs_proj = ccrs.SouthPolarStereo()

# - Define vmin and vmax over time 
vmin_tmp = ds[band].min().compute().values.tolist()
vmax_tmp = ds[band].max().compute().values.tolist()
# - Select timestep
ds_tmp = ds.isel(time=timestep)

# - Create Figure
plt.figure(figsize=(10,8))
ax = plt.subplot(projection=crs_proj)
ds_tmp[band].plot(transform=crs_proj, vmin=vmin_tmp, vmax=vmax_tmp)
# ax.set_title(band)
# - Add coastlines
ax.coastlines() 
## - Add PE marker 
ax.scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
## - Add gridlines 
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--') 
gl.xlabels_bottom = False
gl.xlabels_left = False
gl.xlabels_right = False
gl.ylabels_right = True
plt.show() 

#-----------------------------------------------------------------------------.
########################
#### Plot many bands ###
########################
# - Display solar channels
plot_bands(ds=ds, 
           bands = channels_solar, 
           timestep = 0,
           fname=None)

plot_bands(ds=ds, 
           bands = channels_solar, 
           timestep = 1,
           fname=None)

plot_bands(ds=ds, 
           bands = channels_solar, 
           timestep = 2,
           fname=None)

# - Display thermal channels
plot_bands(ds=ds, 
           bands = channels_thermal, 
           timestep = 0,
           fname=None)

plot_bands(ds=ds, 
           bands = channels_thermal, 
           timestep = 1,
           fname=None)

plot_bands(ds=ds, 
           bands = channels_thermal, 
           timestep = 2,
           fname=None)
#-----------------------------------------------------------------------------.
### Display data in napari
# - Load data into memory 
ds = ds.compute()
# - Define channels 
channels_selected = ['1', '2', '3', '4', '5','7', 
                    '17','18', '19', '20',
                    '22', '23', '25', '26',
                    '27', '28', '29', '31', '32', '33', '34', '35']
ds_channels_stack = ds[channels_selected].to_stacked_array(new_dim="channels", 
                                                           sample_dims=['x','y','time'],
                                                           name="StackedChannels")
ds_channels_stack = ds_channels_stack.transpose('channels','time','y','x') 

ds_channels_stack.coords['channels'] # TOD change name index 

### - Display all original channels (separately)(-->MultiPanel)
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    for ch in channels_selected:
        viewer.add_image(ds[ch], name=ch, rgb=False)

### - Display all original channels (4D tensor) 
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    viewer.add_image(ds_channels_stack, name='Stacked Channels', rgb=False) 

#----------------------------------------------------------------------------. 
### TODO:
# - NAPARI BUG if all values in an array are nan 
# - Enhance channels with histogram equalization
# ---> Histogram equalization 
#      - Collapse all dimension (collapse time)
#      - Iterate over dimension : dim = time 

# l_da_L = list()
# for cmp in ds.data_vars.keys():
#     im = XRImage(ds[cmp])
#     im.stretch_hist_equalize()
#     # da, mode = im.finalize(fill_value=None)
#     # da = da.compute()
#     da = im.data.compute() 
#     l_da_L.append(da)
# ds_equalized = xr.merge(l_da_L) 
# ds_equalized = ds_equalized.compute()

#----------------------------------------------------------------------------. 

 
 