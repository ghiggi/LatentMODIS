#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:52:10 2020

@author: ghiggi
"""
import os
os.chdir("/ltenas3/LatentMODIS")
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
import vaex 
import re
import seaborn as sns
import numpy as np
#-----------------------------------------------------------------------------.
### Main Settings
# - Define project directory 
proj_dir =  "/ltenas3/LatentMODIS"  
proj_data_dir = "/ltenas3/LatentMODIS/data"  
model_dir = '/ltenas3/LatentMODIS/models' 
figs_dir = '/ltenas3/LatentMODIS/figs'
# proj_dir =  "/home/ghiggi/Projects/LatentMODIS"
# proj_data_dir = "/home/ghiggi/Projects/LatentMODIS/data"
# model_dir = "/home/ghiggi/Projects/LatentMODIS/models"

# - Define tabular granule archive from which to read
df_type = "Remapped_Reshaped" # Reshaped 

# - Define training archive 
archive_name = "archive_PE_bbox"  
preprocessing_name = "Original_Bands"   
 
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
channels_selected = [column_name for column_name in  df.columns if column_name not in ID_vars]
df = df[channels_selected]
#-----------------------------------------------------------------------------.
# Compute correlation matrix 
corrMatrix = df.corr()

#-----------------------------------------------------------------------------. 
### Display clustered correlation matrix (all) 
fontsize = 10

sns.set(rc={'figure.figsize':(10,10),
            'figure.dpi': 300})
cg = sns.clustermap(corrMatrix, 
                    cbar_pos=(0, .25, .02, .4), # left, bottom, width, height
                    cbar_kws={'label': 'Correlation'},
                    dendrogram_ratio=(0.1,0.1),
                    cmap = sns.diverging_palette(20, 220, n=200),
                    linewidths = 0.1)
# - Remove dendrograms
cg.ax_row_dendrogram.remove()
cg.ax_col_dendrogram.remove()
# cg.ax_row_dendrogram.set_visible(False)
# cg.ax_row_dendrogram.set_xlim([0,0.001])
# cg.ax_col_dendrogram.set_visible(False)
# cg.ax_col_dendrogram.set_ylim([0,0.001])

# - Colorbar options 
# cg.cax.yaxis.set_label_position('left') # .set_ticks_position
# cg.cax.yaxis.set_label('Correlation')
# cg.cax.set_ylabel('Correlation')
cg.ax_cbar.set_ylabel('Correlation')

# - Display all ticks  
clustered_corrMatrix = cg.data2d
xlabels = clustered_corrMatrix.columns
ylabels = clustered_corrMatrix.index

# - Define channel labels
xlabels = [re.sub("ch_","",s) for s in xlabels]
ylabels = [re.sub("ch_","",s) for s in ylabels]

cg.ax_heatmap.set_xticks(np.arange(clustered_corrMatrix.shape[1])+0.5)
cg.ax_heatmap.set_yticks(np.arange(clustered_corrMatrix.shape[0])+0.5)
cg.ax_heatmap.tick_params(axis=u'both', which=u'both',length=0)
cg.ax_heatmap.set_xticklabels(xlabels, size=fontsize, rotation=45)
cg.ax_heatmap.set_yticklabels(ylabels, size=fontsize)

cg.ax_heatmap.set_title('Correlation between MODIS channels')
# - Add supertitle
# cg.fig.suptitle('Correlation between NDI')
 
# - Save figure
cg.fig.savefig(fname=os.path.join(figs_dir, "CorrMatrix_Channels.png"))

# 