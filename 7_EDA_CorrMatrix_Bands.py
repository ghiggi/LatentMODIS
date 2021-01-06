#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:52:10 2020

@author: ghiggi
"""
import os
os.chdir("/home/ghiggi/Projects/LatentMODIS")
import glob
import vaex 
import seaborn as sns
import matplotlib.pyplot as plt

## MODIS Reflected Solar Bands (Daytime bands)
# - MODIS bands 1-19 and 26 are known as reflected solar bands.
# - Sense solar radiation (photons) reflected and scattered from the atmosphere 
# and surface at wavelengths from 0.41 to 2.2 microns

## MODIS Thermal Emissive Bands
# - MODIS bands 20-25 and 27-36
# - The primary source of these photons is emission by
#   the atmosphere, clouds, land surface, and water surface.

#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir = "/home/ghiggi/Projects/LatentMODIS"
proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# - Define training archive 
archive_name = "archive_PE_bbox"
# - Data type
df_type = "Remapped_Reshaped"
# df_type = "Reshaped"
# - Define channels 
channels_selected = ['1', '2', '3', '4', '5','7', 
                    '17','18', '19', '20',
                    '22', '23', '25', '26',
                    '27', '28', '29', '31', '32', '33', '34', '35']
channels_selected = ["ch_" + c for c in channels_selected]  

#-----------------------------------------------------------------------------.
### Define data and model paths   
training_data_dir = os.path.join(proj_data_dir, archive_name, "training_data", df_type)

#-----------------------------------------------------------------------------.
### Load standardized training data 
df = vaex.open(os.path.join(training_data_dir,"unique_training_data_std.hdf5"))
df = df[channels_selected].to_pandas_df()

# Compute correlation matrix 
corrMatrix = df.corr()
print (corrMatrix)

# Display correlation matrix
f, ax = plt.subplots(figsize =(9, 8)) 
ax = sns.heatmap(
    corrMatrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    # linewidths = 0.1, # for white border of each cell 
    square=True,
    annot=False)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
plt.show()

# Hiearchical clustering of correlation matrix  
cg = sns.clustermap(corrMatrix, 
                    cmap = sns.diverging_palette(20, 220, n=200),
                    linewidths = 0.1); 
ax = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
plt.show()


# 