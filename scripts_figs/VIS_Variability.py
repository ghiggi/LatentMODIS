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
import seaborn as sns
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
satellite_data_dir = '/ltenas3/alfonso/image_proc/MYD'
figs_dir = '/ltenas3/LatentMODIS/figs'
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# satellite_data_dir = '/home/ghiggi/Data/' 

# - Define archive name 
product = 'MYD021KM'
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
# - Select band
var = "1"
# - MODIS band 1 to dataframe   
df = ds[var].to_dataframe().reset_index()
# - Subset of 4 timesteps of MODIS band 1
da = ds[var].isel(time=slice(5,9))
#-----------------------------------------------------------------------------.
########################### 
### Create the figure   ###
###########################
# - Define map settings
PE_lat, PE_lon = -71.949944, 23.347079
crs_WGS84 = ccrs.PlateCarree()  
crs_proj = ccrs.SouthPolarStereo()
# - Define figure options 
fig = plt.figure(figsize=(12,5), dpi=600, constrained_layout=True)
# - Specify figure layout
gs = fig.add_gridspec(2, 4)
ax1 = fig.add_subplot(gs[:, 0:2])
ax2 = fig.add_subplot(gs[0, 2], projection=crs_proj)  
ax3 = fig.add_subplot(gs[0, 3], projection=crs_proj)  
ax4 = fig.add_subplot(gs[1, 2], projection=crs_proj)  
ax5 = fig.add_subplot(gs[1, 3], projection=crs_proj)  
l_ax = [ax2, ax3, ax4, ax5]
# - Add violinplot
ax = sns.violinplot(x=df.time.dt.hour, y=var, data=df, # palette="light:g",
                    orient="v", ax=ax1)
ax.set_xlabel('Hour') 
ax.set_ylabel('Reflectance [%]') 
ax.set_title('Diurnal variability of MODIS Band 1') 
# - Add maps
for i in range(len(l_ax)):
    # - Select timestep
    da_tmp = da.isel({'time': i})
    # - Select title
    tmp_title = np.datetime_as_string(da_tmp['time'].values, unit='s')   
    # - Create subplot
    l_ax[i].get_gridspec().update(wspace=0) # hspace=0,
    # - Plot the image 
    im = da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i], 
                       add_colorbar = False, vmin = 20, vmax = 70)
    l_ax[i].set_title(tmp_title)
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
# - Add colorbar
cbar = fig.colorbar(im, ax=l_ax, extend='both',
                     shrink=1, aspect=50)
cbar.ax.set_ylabel('Reflectance [%]')
# - Save figure
fig.savefig(fname=os.path.join(figs_dir, "VIS_Variability.png"))
 


 
 
 
