#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:43:34 2020

@author: ghiggi
"""

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
from latentpy.dimred  import load_UMAP
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
PCA_preprocessing = True
PCA_n_components = 20
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"

umap2D = load_UMAP(model_dir = model_dir, 
                   n_components = 2,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   metric = metric,
                   PCA_preprocessing = PCA_preprocessing, 
                   PCA_n_components = PCA_n_components)
umap3D = load_UMAP(model_dir = model_dir, 
                   n_components = 3,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   metric = metric,
                   PCA_preprocessing = PCA_preprocessing, 
                   PCA_n_components = PCA_n_components)
#-----------------------------------------------------------------------------.                
################################ 
### Project to latent space ####
################################ 
# - Project to UMAP latent space        
umap2D_latent = umap2D.transform(df_std[channels_selected])
umap3D_latent = umap3D.transform(df_std[channels_selected])
# - Define colors from 3D latent space 
# umap3D_RGB_color = umap3D.transform_to_RGB(df_std[channels_selected]) 
umap3D_RGB_color = umap3D.RGB_scaler.transform(umap3D_latent)
 
#-----------------------------------------------------------------------------.
############################
### Plot RGB composites ####
############################
da_UMAP_RGB = reshape_to_RGB_composite(df = df_std,
                                        RGB_matrix = umap3D_RGB_color,
                                        dims = s_coords + [t_coords], 
                                        name="UMAP_Composite")

plot_RGB_DataArrays(da_UMAP_RGB.isel(overpass_time=slice(0,16)), title="UMAP RGB", time_dim=t_coords)  
plot_RGB_DataArrays(da_UMAP_RGB.isel(overpass_time=slice(16,32)), title="UMAP RGB", time_dim=t_coords)

#----------------------------------------------------------------------------.
####################################### 
### Define PolyClassifier for UMAP ####
#######################################
# - Draw polygons 
pc = PolyClassifier(X=umap2D_latent,
                    color=umap3D_RGB_color,
                    s = 0.01)

# - Display latent-space PolygonClassifiers 
pc.gpd.plot('labels')

# - Define class name and assign name to labels 
dict_labels = {'1': 'Clouds',
               '2': 'Snow'}
pc.assign_labels_names(dict_labels) 
print(pc.labels)
print(pc.labels_names)
print(pc.dict_labels)

##----------------------------------------------------------------------------.
# - Plot classified labels latent-space 
labels = pc.predict(X=umap2D_latent, option = 'labels')
plt.scatter(umap2D_latent[:,0], umap2D_latent[:,1], c=labels, s=0.1)
plt.show()

# - Plot classified label_names
category_colors_dict = {'Clouds':'cornflowerblue', \
                        'Snow': 'green', \
                        'Unclassified': 'gray', \
}
#                         'LowClouds': 'cornflowerblue', \
#                         'HighClouds': 'cornflowerblue', \
#                         'Mountains': 'black', \
#                         'Glacier': 'mediumpurple'}
labels_names = pc.predict(X=umap2D_latent, option = 'labels_names')
plot_2D_category(umap2D_latent, 
                 labels = labels_names,
                 category_colors_dict = category_colors_dict)

##----------------------------------------------------------------------------.
# - Save PolyClassifier  
pc_fpath = os.path.join(model_dir,'UMAP_PolyClassifier.sav')
joblib.dump(pc, pc_fpath)

#----------------------------------------------------------------------------.
### Load PolyClassifier  
# pc = joblib.load(os.path.join(model_dir,'UMAP_PolyClassifier.sav'))
#
#----------------------------------------------------------------------------.
### - Modify polygons if necessary 
# pc.modify(X=umap2D_latent,
#           colors = umap3D_RGB_color,        # points colors 
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
# pc_fpath = os.path.join(model_dir,'UMAP_PolyClassifier.sav')
# joblib.dump(pc, pc_fpath)