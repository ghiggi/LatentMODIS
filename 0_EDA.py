#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:34:29 2020

@author: ghiggi
"""
import os 
os.chdir("/home/ghiggi/Projects/LatentMODIS")
import napari 
import satpy 
import xarray as xr
from satpy.writers import get_enhanced_image
from satpy import Scene
from trollimage.xrimage import XRImage
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from latentpy.laads_utils import find_MODIS_L1B_paths
##-----------------------------------------------------------------------------.
# Define directory where satellite data are stored on disk 
data_dir = '/home/ghiggi/Data/'
##-----------------------------------------------------------------------------.
# Retrieve filepaths of MODIS L1B data
geo_filepaths, l1b_filepaths = find_MODIS_L1B_paths(data_dir = data_dir,
                                                    satellite = 'Aqua',
                                                    product = "MYD021KM",
                                                    collection = '61')
##-----------------------------------------------------------------------------.
### Load data
scn = Scene(reader='modis_l1b', filenames=[l1b_filepaths[0], geo_filepaths[0]])
scn.available_dataset_names()
scn.available_composite_names()
##----------------------------------------------------------------------------.
### Retrieve channels 
# Define channels 
channels = [str(i) for i in range(1, 39)] 
channels[12] = '13lo'
channels[13] = '14lo'
channels[36] = '13hi'
channels[37] = '14hi'
# Load channels 
scn.load(channels, resolution = 1000)
# Convert to xarray Dataset
ds_channels = scn.to_xarray_dataset(channels) 
# Drop crs as coordinate
ds_channels = ds_channels.drop_vars('crs')

### Enhance channels with histogram equalization
l_da_L = list()
for cmp in ds_channels.data_vars.keys():
    im = XRImage(ds_channels[cmp])
    im.stretch_hist_equalize()
    # da, mode = im.finalize(fill_value=None)
    # da = da.compute()
    da = im.data.compute() 
    l_da_L.append(da)
 
ds_channels_equalized = xr.merge(l_da_L) 
# ds_channels_equalized = ds_channels_equalized.transpose('y','x','bands')  

##----------------------------------------------------------------------------.
### Retrieve RGB composite 
# Define composites
composites = scn.available_composite_names()
# Load composites 
scn.load(composites, resolution = 1000)
# Retrieve RGB from composites
# ds_RGB = xr.merge([get_enhanced_image(scn[cmp]).data for cmp in composites])
l_da_RGB = list()
for cmp in composites:
    da = get_enhanced_image(scn[cmp]).data 
    da.name = da.attrs['name']
    l_da_RGB.append(da)
ds_composites = xr.merge(l_da_RGB) 
# Drop crs as coordinate
ds_composites = ds_composites.drop_vars('crs')
# Select just RGB (luminance and transparency removed...)
ds_composites = ds_composites.sel(bands=["R","G","B"])
# Move RGB dimension as last 
ds_composites = ds_composites.transpose('y','x','bands')  
# Compute the composites 
ds_composites = ds_composites.compute()
# Remove composites with only NaN
ds_composites = ds_composites.drop_vars(['ir108_3d','ir_cloud_day'])
# Set NaN to alpha transparency 1 
l_da_RGBA = list()
for cmp in ds_composites.data_vars.keys():
    im = XRImage(ds_composites[cmp])
    da, mode = im.finalize(fill_value=None)
    da = da.compute()
    l_da_RGBA.append(da)
ds_composites = xr.merge(l_da_RGBA)    
 
##----------------------------------------------------------------------------.
# Create stacked array with all channels along a dimension
ds_channels_stack = ds_channels.to_stacked_array(new_dim="channels", sample_dims=['x','y'], name="StackedChannels")
ds_channels_stack = ds_channels_stack.transpose('channels','y','x') 
ds_channels_equalized_stack = ds_channels_equalized.to_stacked_array(new_dim="channels", sample_dims=['x','y'], name="StackedChannels")
ds_channels_equalized_stack = ds_channels_equalized_stack.transpose('channels','y','x') 

##----------------------------------------------------------------------------.              
# Create a stack of RGB composites 
ds_composites_stack = ds_composites.to_stacked_array(new_dim="composite", sample_dims=['x','y','bands'], name="StackedRGBComposites")
ds_composites_stack = ds_composites_stack.transpose('composite','y','x','bands')  

##----------------------------------------------------------------------------. 
### Preload data to visualize (satpy is lazy)
ds_channels = ds_channels.compute()
ds_channels_equalized = ds_channels_equalized.compute()
ds_channels_stack = ds_channels_stack.compute()
ds_channels_equalized_stack = ds_channels_equalized_stack.compute()
ds_composites = ds_composites.compute()
ds_composites_stack = ds_composites_stack.compute()
##----------------------------------------------------------------------------. 
### Display all original channels (separately)(-->MultiPanel)
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    for ch in ds_channels.data_vars.keys():
        viewer.add_image(ds_channels[ch], name=ch, rgb=False)

### Display all histogram-equalized channels (separately)(-->MultiPanel)       
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    for ch in ds_channels_equalized.data_vars.keys():
        viewer.add_image(ds_channels_equalized[ch], name=ch, contrast_limits = (0,1), rgb=False)        

### Display channels stacked along 3D dimension
# --> Slides over channels 
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    viewer.add_image(ds_channels_stack, name='Stacked Channels', rgb=False) 

### Display histogram-equalized channels stacked along 3D dimension
# --> Slides over channels 
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    viewer.add_image(ds_channels_equalized_stack, name='Stacked Channels', contrast_limits = (0,1), rgb=False) 
    
#-----------------------------------------------------------------------------.
### Display all RGB composites (separately)(-->Multipanel)
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    for cmp in ds_composites.data_vars.keys():
        print(cmp)
        viewer.add_image(ds_composites[cmp], name=cmp, rgb=True, visible = True)

### Display all RGB composites stacked along 3D dimension     
with napari.gui_qt():
    # create an empty viewer
    viewer = napari.Viewer(ndisplay=2)   
    viewer.add_image(ds_composites_stack, name='Stacked_RGB_Composites', rgb=True, visible = True)
   
 