#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:13:01 2020

@author: ghiggi & ferrone
"""
import os
# os.chdir("/home/ghiggi/Projects/LatentMODIS")
os.chdir("/ltenas3/LatentMODIS")
import vaex
import joblib
import matplotlib.pyplot as plt

from latentpy.poly_classifier import PolyClassifier 
from latentpy.utils_reshape import reshape_to_RGB_composite
from latentpy.plotting import plot_RGB_DataArrays
from latentpy.plotting import plot_2D_category
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
### Define data and model paths   
model_dir =  os.path.join(proj_dir, "models", archive_name, preprocessing_name)
tabular_data_dir = os.path.join(proj_data_dir, archive_name, df_type)
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
### Open the training dataset
# - full_* means that all pixels inside the bbox are present 
#   --> It is possible to reshape the df to a space-time cube
# - unique_* means that many pixels (values) have been discarded during preprocessing
#   --> It is not possible to reshape the df to a space-time cube
df_full = vaex.open(os.path.join(training_data_dir,"full_training_data.hdf5")) 
df_full_std = vaex.open(os.path.join(training_data_dir,"full_training_data_std.hdf5")) 

#-----------------------------------------------------------------------------. 
### Select the desired dataset
df_std = df_full_std.to_pandas_df() 

# - Subset daytime data
df_std = df_std[df_std['overpass_time'].dt.hour > 7 ] 
df_std = df_std[df_std['overpass_time'].dt.hour < 17 ]

# - Retrieve channels names
column_names = df_std.columns
channels_selected = [column_name for column_name in column_names if column_name not in ID_vars]

#-----------------------------------------------------------------------------.
####################
### Load models ####
####################
##----------------------------------------------------------------------------.
### - Load trained PCA models
pca2D_filepath = os.path.join(model_dir,'pca2D.sav')
pca3D_filepath = os.path.join(model_dir,'pca3D.sav')
pca2D = joblib.load(pca2D_filepath)
pca3D = joblib.load(pca3D_filepath)
##----------------------------------------------------------------------------.
### - Load RGB Scalers
pca3D_RGB_scaler = joblib.load(os.path.join(model_dir,'scaler_pca3D_RGB.sav'))

#-----------------------------------------------------------------------------.
################################ 
### Project to latent space ####
################################
pca2D_latent = pca2D.transform(df_std[channels_selected]) 
pca3D_latent = pca3D.transform(df_std[channels_selected]) 
pca3D_RGB_color = pca3D_RGB_scaler.transform(pca3D_latent)

#-----------------------------------------------------------------------------.
############################
### Plot RGB composites ####
############################
da_PCA_RGB = reshape_to_RGB_composite(df = df_std,
                                      RGB_matrix = pca3D_RGB_color,
                                      dims = s_coords + [t_coords], 
                                      name="PCA_Composite")
plot_RGB_DataArrays(da_PCA_RGB.isel(overpass_time=slice(0,16)),  title="PCA RGB", time_dim=t_coords)
plot_RGB_DataArrays(da_PCA_RGB.isel(overpass_time=slice(16,32)), title="PCA RGB", time_dim=t_coords)

#----------------------------------------------------------------------------.
####################################### 
### Define PolyClassifier for PCA  ####
#######################################
# - Draw polygons 
pc = PolyClassifier(X = pca2D_latent,
                    color = pca3D_RGB_color,
                    s = 0.01)

# - Display latent-space PolygonClassifiers 
pc.gpd.plot('labels')

# - Define class name and assign name to labels 
dict_labels = {'1': 'Clouds'}#,
#               '2': 'Snow'}
pc.assign_labels_names(dict_labels) 
print(pc.labels)
print(pc.labels_names)
print(pc.dict_labels)

##----------------------------------------------------------------------------.
# - Plot classified labels latent-space 
labels = pc.predict(X=pca2D_latent, option = 'labels')
plt.scatter(pca2D_latent[:,0], pca2D_latent[:,1], c=labels, s=0.1)
plt.show()

# - Plot classified label_names
category_colors_dict = {'Clouds':'cornflowerblue', \
                        'Unclassified': 'gray' \
}
#                         'Snow': 'green', \
#                         'LowClouds': 'cornflowerblue', \
#                         'HighClouds': 'cornflowerblue', \
#                         'Mountains': 'black', \
#                         'Glacier': 'mediumpurple'}
labels_names = pc.predict(X=pca2D_latent, option = 'labels_names')
plot_2D_category(pca2D_latent, 
                 labels = labels_names,
                 category_colors_dict = category_colors_dict)
ax = plt.gca()
ax.grid(ls=':', c='tab:gray', alpha=0.5)

##----------------------------------------------------------------------------.
# - Save PolyClassifier  
pc_fpath = os.path.join(model_dir,'PCA_PolyClassifier_alfonso_20201223.sav')
joblib.dump(pc, pc_fpath)

#-----------------------------------------------------------------------------.
### Load PolyClassifier  
# pc = joblib.load(os.path.join(model_dir,'PCA_PolyClassifier.sav'))

#-----------------------------------------------------------------------------.
### Modify polygons if necessary 
# pc.modify(X = pca2D_latent,
#           colors = pca3D_RGB_color,        # points colors 
#           cmap = cc.cm.glasbey_light) # polygons colors
#
# pc.gpd.plot(column='labels')
# pc.gpd.plot(column='labels_names')
# print(pc.labels)
# print(pc.labels_names)
# print(pc.dict_labels)
#
# ##----------------------------------------------------------------------------.
# - Save PolyClassifier  
# pc_fpath = os.path.join(model_dir,'PCA_PolyClassifier.sav')
# joblib.dump(pc, pc_fpath)