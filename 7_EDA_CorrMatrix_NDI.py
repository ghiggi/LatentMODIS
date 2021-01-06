#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:46:27 2020

@author: ghiggi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:52:10 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import glob
import vaex 
import seaborn as sns
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
model_dir = '/ltenas3/LatentMODIS/models' 
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# model_dir = "/home/ghiggi/Projects/LatentMODIS/models"

# - Define tabular granule archive from which to read
df_type = "Remapped_Reshaped" # Reshaped 

# - Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Normalized_Difference_Indices"  #  # "Original_Subset"
 
#-----------------------------------------------------------------------------.
### Define data path
training_data_dir = os.path.join(proj_data_dir, archive_name, "training_data", df_type, preprocessing_name)
      
#-----------------------------------------------------------------------------.
### Define ID variables   
if df_type == "Remapped_Reshaped":
    s_coords = ['x','y']        # x and y of the planar projection 
    t_coords = 'overpass_time'
    ID_vars = ['idx_x','idx_y','overpass_time','x','y']
else: # Reshaped
    s_coords = ['idx_x','idx_y'] # idx of image row and column 
    t_coords = 'overpass_time'
    ID_vars = ['idx_x','idx_y','overpass_time','latitude','longitude']
    
#-----------------------------------------------------------------------------. 
### Load lazily the dataset
# - full_* means that all pixels inside the bbox are present 
#   --> It is possible to reshape the df to a space-time cube
# - unique_* means that many pixels (values) have been discarded during preprocessing
#   --> It is not possible to reshape the df to a space-time cube
df = vaex.open(os.path.join(training_data_dir,"unique_training_data.hdf5"))  
df = df.to_pandas_df()
# - Retrieve channels names
NDI_selected = [column_name for column_name in  df.columns if column_name not in ID_vars]
df = df[NDI_selected]
 
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