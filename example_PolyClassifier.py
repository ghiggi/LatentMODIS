#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:13:01 2020

@author: ghiggi & ferrone
"""
import os
os.chdir("/home/ghiggi/Projects/LatentMODIS")
# os.chdir("/ltenas3/LatentMODIS")
import joblib
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
from latentpy.poly_classifier import PolyClassifier 

##----------------------------------------------------------------------------. 
### Generating points for scatterplot
# Points normalized between 1 and 100 otherwise visualization on Napari
#  defaults to weird values...
np.random.seed(666)
NUM_POINTS = 10000
latent2D = np.random.rand(NUM_POINTS, 2)*100
latent_RGB = np.random.rand(NUM_POINTS, 3)

plt.plot(latent2D[:,0],latent2D[:,1])

##----------------------------------------------------------------------------.
### Draw polygons 
pc = PolyClassifier(X=latent2D,
                    color=latent_RGB,
                    s = 0.5)

# Display labels of PolygonClassifiers 
pc.gpd.plot('labels')
##----------------------------------------------------------------------------.
### With labels 
# - Predict labels (Classify points)
labels = pc.predict(X=latent2D, option = 'labels')
print(labels)
np.unique(labels)

# - Plot in napari X and PolyClassifier 
pc.plot(X=latent2D,
        colors = latent_RGB,        # points colors 
        cmap = cc.cm.glasbey_light) # polygons colors 

# - Modify polygons 
# pc.modify(X=latent2D,
#           colors = latent_RGB,        # points colors 
#           cmap = cc.cm.glasbey_light) # polygons colors 

# - Define class name and assign name to labels 
dict_labels = {'1': 'Class 1',
               '2': 'Cloud',
               '3': 'Wrong'}
pc.assign_labels_names(dict_labels) 

print(pc.labels)
print(pc.labels_names)
print(pc.dict_labels)

pc.gpd.plot(column='labels')
pc.gpd.plot(column='labels_names')

###----------------------------------------------------------------------------. 
### With labels names 
# - Predict label names (Classify points)
labels = pc.predict(X=latent2D, option = 'labels_names')
labels   

# Plot in napari X and PolyClassifier
pc.plot(X=latent2D,
        colors = latent_RGB,        # points colors 
        cmap = cc.cm.glasbey_light) # polygons colors 

##-----------------------------------------------------------------------------.
### Modify polygons  
pc.modify(X=latent2D,
          colors = latent_RGB,        # points colors 
          cmap = cc.cm.glasbey_light) # polygons colors

pc.gpd.plot(column='labels')
pc.gpd.plot(column='labels_names')

print(pc.labels)
print(pc.labels_names)
print(pc.dict_labels)

##-----------------------------------------------------------------------------.
### Plot predictions 
labels = pc.predict(X=latent2D, option = 'labels')
plt.scatter(latent2D[:,0], latent2D[:,1], c=labels, s=0.1)
plt.show()

##-----------------------------------------------------------------------------.
## Save PolyClassifier  
pc_filepath = os.path.join(model_dir, 'PCA_PolyClassifier.sav')
joblib.dump(pc, pc_filepath)

## Load PolyClassifier
pc1 = joblib.load(pc_filepath)

##-----------------------------------------------------------------------------.
### TODO: Geometry operations 
# - Check polygon do not intersect ! 
# - Modify polygon automatically so that do not intersect !
# - Check polygon not outside bbox 

# - Polygon topology check: P.buffer(0) 
# - check p.isvalid 

### TODO: Improvements:
# - Increase all polygons to fill all latent space bbox 
 
# - Create custom cmap for each category    
## Colorbar 
# - ColorDictKeys
# - Discrete 
# - Below one 
# - Colorbar dict 
# '''
# # In case you don't want to use a discrete colormap, but instead you
# # want to discretize a continuous one (like 'jet'), use the following code:
# cNorm  = colors.Normalize(vmin=0, vmax=len(polygon_shapely_list)+1)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

# # And pass:
# # np.array(scalarMap.to_rgba(labels))
# # as the face_color value for viewer.add_points
# '''
##-----------------------------------------------------------------------------.
 
 
 
 
 






 