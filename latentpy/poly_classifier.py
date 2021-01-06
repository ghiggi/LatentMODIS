#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:39:10 2020

@author: ghiggi
"""
import numpy as np
import napari
import colorcet as cc
from itertools import chain
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
from shapely.ops import cascaded_union # unary_union
from geopandas import GeoDataFrame
import geopandas
import pandas as pd 

##----------------------------------------------------------------------------.    
### Polygon Format Conversions 
## TODO: Update with PolyConversion()

def match(a, b):
    """For each element of a, if existing in b it returns the index position in b.""" 
    if not isinstance(a , list):
        a = [a] 
    if not isinstance(b, list):
        b = [b]     
    return [ b.index(x) if x in b else None for x in a ]
##----------------------------------------------------------------------------.
def geopandas_to_PolyArrayList(gpd):
    if not isinstance(gpd, (geopandas.geodataframe.GeoDataFrame, geopandas.geoseries.GeoSeries)):
        raise TypeError("Please provide a GeoDataFrame or a GeoSeries")
    if isinstance(gpd, geopandas.geodataframe.GeoDataFrame):
        gpd = gpd.geometry 
    # Convert to PolyArrayList      
    return [np.array(p.exterior.coords.xy).T for p in gpd.to_list()]

##----------------------------------------------------------------------------.    
### Shapely Tools 
def get_shapely_bbox(bbox):
    return box(bbox[0], bbox[1], bbox[2], bbox[3])

def get_shapely_extent(extent):
    return box(extent[0], extent[2], extent[1], extent[3])

#----------------------------------------------------------------------------.  
### PolyClassifier
def _create_gpd(self, PolyArrayList, labels=None, labels_names=None):                                          
    ##--------------------------------------------------------------------.
    # Define labels if not provided 
    if labels is None: 
        labels = np.arange(1,len(PolyArrayList)+1)
    else: 
        if isinstance(labels, list):
            raise TypeError("Provide 'labels' as numpy array")     
    if labels_names is not None: 
        if isinstance(labels_names, list):
            raise TypeError("Provide 'labels_names' as numpy array")
        if len(labels) != len(labels_names): 
            raise ValueError("Length of 'labels' and labels_names' must coincide")
    ##--------------------------------------------------------------------.        
    # Add the 0 label for the unclassified area
    labels = labels.tolist()
    labels.insert(0, 0) # for unclassified area
    labels = np.array(labels)
    if labels_names is not None: 
        labels_names = labels_names.tolist()
        labels_names.insert(0, 'Unclassified')  
        labels_names = np.array(labels_names)     
    ##------------------------------------------------------------------------.
    # - Convert PolyArrayList to PolyShapelyList
    Poly_ShapelyList = [Polygon(poly) for poly in PolyArrayList]
    ##------------------------------------------------------------------------.
    # - Create a polygon rectangle delimiting 2D latent space 
    LatentSpace_ShapelyPolygon = get_shapely_bbox(self.bbox)
    ##------------------------------------------------------------------------.
    # - Create a polygon for the unlabelled data points 
    DrawnPoly_Union = cascaded_union(Poly_ShapelyList)
    Unlabelled_ShapelyPolygon = LatentSpace_ShapelyPolygon.difference(DrawnPoly_Union)
    ##--------------------------------------------------------------------.
    # - Combine polygon for unlabelled point and drawn polygons into a ShapelyList
    PolyClassifier_ShapelyList = [Unlabelled_ShapelyPolygon] + Poly_ShapelyList
    ##------------------------------------------------------------------------.
    # Define pandas DataFrame 
    if labels_names is not None: 
        df = pd.DataFrame({'labels': labels, 'labels_names': labels_names})
    else: 
        df = pd.DataFrame({'labels': labels})
    ##------------------------------------------------------------------------.
    ## Create geopandas PolyClassifier
    # - Create geopandas object    
    self.gpd = GeoDataFrame(df,
                            crs=None,
                            geometry=PolyClassifier_ShapelyList)
    # - Add labels info 
    self.labels = labels[labels != 0]
    if labels_names is None: 
        self.labels_names = None
    else:
        self.labels_names = labels_names[labels_names != 'Unclassified']
    ##------------------------------------------------------------------------.
            
class PolyClassifier:           
    def __init__(self, X, color='black', s=0.5, verbose=True):
        # Color can also be RGB array 
        ##--------------------------------------------------------------------.
        # Check X has 2 columns 
        if (X.shape[1] != 2): 
            raise ValueError("X must have 2 columns") 
        ##--------------------------------------------------------------------.
        # Check color argument
        if ((len(color) != 1) and (len(X) != len(color))):
            raise ValueError('Provided (RGB) array must have same number of rows as X')
        ##--------------------------------------------------------------------.
        # Explain commands in napari 
        if verbose is True:
            print("Draw polygons around points; close the window when finished.")
        ##--------------------------------------------------------------------.
        # Open Napari interface
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_points(X, size = s, face_color = color)
            polygons_layer = viewer.add_shapes(None, shape_type='polygon')
        ##--------------------------------------------------------------------.     
        if verbose is True:
            print("You have drawn", len(polygons_layer.data), "polygons")
        ##--------------------------------------------------------------------.
        ##--------------------------------------------------------------------.
        # - Define 2D space bounding box 
        x_minmax = [np.min(X[:,0]), np.max(X[:,0])]
        y_minmax = [np.min(X[:,1]), np.max(X[:,1])]
        self.bbox = [x_minmax[0], y_minmax[0], x_minmax[1], y_minmax[1]]
        self.xmin = self.bbox[0]
        self.xmax = self.bbox[2]
        self.ymin = self.bbox[1]
        self.ymax = self.bbox[3]
        ##--------------------------------------------------------------------.
        ## Process the drawn polygons
        # - Get a PolyArrayList describing the drawn polygon
        PolyArrayList = polygons_layer.data
        # - Check topology 
        # TODO 
        ##--------------------------------------------------------------------.
        # - Construct the PolyClassifier 
        _create_gpd(self, PolyArrayList = PolyArrayList)
        ##--------------------------------------------------------------------.
        # - Define labels options
        self.PolyArrayList = PolyArrayList     
        self.option = 'labels' 
        
    ##------------------------------------------------------------------------.
    def assign_labels_names(self, dict_labels):
        # Check dict is not empty 
        if len(dict_labels) == 0: 
            return 
        # Convert integer labels to str 
        labels_str = [str(i) for i in self.labels]
        # Be sure that the dict key are strings 
        dict_labels ={str(k): v for k,v in dict_labels.items()}
        # Select keys that are also in labels 
        dict_keys = [k for k in dict_labels.keys() if k in labels_str]
        if len(dict_keys) == 0:
            return 
        # Create labels_names 
        labels_names = [dict_labels[s] if s in dict_keys else "[Unassigned] " + s for s in labels_str] 
        ##--------------------------------------------------------------------.
        # Create dict_labels 
        dict_labels = dict(zip(labels_str, labels_names))
        dict_labels['0'] = 'Unclassified'
        ##--------------------------------------------------------------------.
        # Get labels_names for gpd
        gpd_labels_names = [dict_labels[str(k)] for k in self.gpd['labels'].tolist()]
        ##--------------------------------------------------------------------.        
        # Update object 
        self.dict_labels = dict_labels 
        self.labels_names = np.array(labels_names)
        self.gpd['labels_names'] = gpd_labels_names 
        self.option = 'labels_names'
                 
    ##------------------------------------------------------------------------.
    def predict(self, X, option='labels'):
        #---------------------------------------------------------------------.
        ## Checks 
        # Check X has 2 columns 
        if (X.shape[1] != 2): 
            raise ValueError("X must have 2 columns") 
        # Check option is either labels or labels_names   
        if option not in ['labels','labels_names']:
            raise ValueError("Option argument must be 'labels' or 'labels_names'")
        if (option == 'labels_names') and ('labels_names' not in self.gpd.columns.to_list()):
            print("Labels names not available. Predict integer labels")
            option = 'labels'
        #---------------------------------------------------------------------.
        # Force all data points in X inside the training bbox
        X = X.copy()
        X[X[:,0] < self.xmin, 0] = self.xmin 
        X[X[:,0] > self.xmax, 0] = self.xmax 
        X[X[:,1] < self.ymin, 1] = self.ymin
        X[X[:,1] > self.ymax, 1] = self.ymax 
        #---------------------------------------------------------------------.
        # - Convert data points in 2D space to ShapelyPoint 
        Point_ShapelyList = [Point(p) for p in X]
        # - Labels points based on polygons
        labels = np.zeros(len(X), dtype=int)
        for i, poly in enumerate(self.gpd.geometry): 
            idx_points_inside_poly = [poly.contains(p) for p in Point_ShapelyList]
            # Labels start at 1 (label 0: unclassified)
            labels[idx_points_inside_poly] = self.gpd['labels'][i] 
        # - Convert to labels_names if required
        if option == 'labels_names':
            labels = np.array([self.dict_labels[str(k)] for k in labels])
        return labels
          
    def plot(self, X, colors=None, 
             cmap = cc.cm.glasbey_light, s=0.5, modify=False):
        #---------------------------------------------------------------------.
        # Check X has 2 columns 
        if (X.shape[1] != 2): 
            raise ValueError("X must have 2 columns") 
        #---------------------------------------------------------------------.
        # Check labels_names or labels are available in self.gpd  
        if self.option not in self.gpd.columns.to_list():
            raise ValueError(self.option,"is not a column name of self.gpd!")
        #---------------------------------------------------------------------.
        # Retrieve labels
        labels = self.predict(X = X, option = 'labels')
        #-------------------------------------------------------------------------.
        # Remove polygon with label == 0 (unclassified) 
        # --> Napari does not plot polygons with holes
        poly_gpd = self.gpd 
        poly_gpd = poly_gpd[poly_gpd['labels'] != 0] 
        # Retrieve and parse labels for polygons
        poly_labels = poly_gpd['labels'].to_list()
        # - Parse Polygons label name if integer  
        if (self.option == 'labels'):
            poly_labels_names = ['Label %d' % lab for lab in poly_labels]
        else: 
            poly_labels_names = [self.dict_labels[str(k)] for k in poly_labels] 
        poly_labels_names = [s + " [Polygon]" for s in poly_labels_names]
        #-------------------------------------------------------------------------.    
        # Retrieve and parse labels for points 
        points_labels = np.unique(labels)
        # - Parse Points label name if integer  
        if (self.option == 'labels'):
            idx_0 = points_labels.tolist() == 0
            points_labels_names = ['Label %d' % lab for lab in points_labels.tolist()]
            points_labels_names[idx_0] = 'Unclassified' 
        else: 
            points_labels_names = [self.dict_labels[str(k)] for k in points_labels]
        points_labels_names = [s + "[Points]" for s in points_labels_names]
        #-------------------------------------------------------------------------.
        # Assign colors to polygons  
        # TODO: cmap with class dictionary 
        # - If integer labels --> CategoricalCMAP 
        # - If strings --> DictionaryCMAP 
        poly_labels_colors = cmap(poly_labels) 
        #-------------------------------------------------------------------------.
        # Assign colors to points if not specified  
        if colors is None:
            if cmap is not None:
                colors = cmap(points_labels) 
            else: 
                colors = np.ones((len(points_labels),4)) # white 
        else:
            # If colors is a string, repeat n times 
            if isinstance(colors, str): 
                colors = [colors for i in range(len(points_labels))]
            # If is a list or RGB array 
            else: 
                if (len(colors) != len(labels)):
                    raise ValueError('colors argument must have same size as labels')
        #-------------------------------------------------------------------------.    
        # Convert geopandas polygons to PolyArrayList 
        PolyArrayList = geopandas_to_PolyArrayList(poly_gpd) 
        #-------------------------------------------------------------------------.    
        # Store polygons (to record modifications) 
        n_polygons = len(PolyArrayList)
        poly_list = [0 for i in range(n_polygons)]
        
        with napari.gui_qt():
            viewer = napari.Viewer()
            # Plot every polygon separately 
            for i in range(n_polygons):
                poly_list[i] = viewer.add_shapes(data = [PolyArrayList[i]],  # [] to void BUG with color
                                                shape_type = 'polygon',
                                                edge_color = 'white',
                                                face_color = poly_labels_colors[i],
                                                name = poly_labels_names[i])
            # Plot data points for each label separetely 
            for j in range(len(points_labels)):
                idx_current_label = labels == points_labels[j]
                # If there is at least a point with the current label
                if np.sum(idx_current_label) > 0:
                    viewer.add_points(data = X[idx_current_label, :], 
                                      size = s,
                                      face_color = colors[j],
                                      name = points_labels_names[j])
        ##--------------------------------------------------------------------.
        if (modify is True):
            # Update PolyClassifier 
            PolyArrayList = [poly_layer.data for poly_layer in poly_list]
            self.PolyArrayList = PolyArrayList     
        
    def modify(self, X, colors=None, cmap = cc.cm.glasbey_light, s=0.5):
        #---------------------------------------------------------------------.
        # Modify the polygons 
        print("1. On the left-menu, select the polygon/class to modify")
        print("2. Then modify/add/remove the polygons for such class")
        print("3. Exit Napari when you are done.")
        self.plot(X=X, colors=colors, cmap=cmap, s=s, modify=True)
        #---------------------------------------------------------------------.
        # - Check polygon topology 
        # --> check_topology(self.PolyArrayList) 
        #     --> If error --> Redrawn 
        # --> Check if sublist in PolyArrayList (to adapt self.labels)
        #---------------------------------------------------------------------.   
        # PolyArrayList = pc.PolyArrayList 
        # labels = pc.labels
        # labels = labels.tolist()
        # labels_names = pc.labels_names
        PolyArrayList = self.PolyArrayList 
        labels = self.labels
        labels_names = self.labels_names
        labels = labels.tolist()
        labels_names = labels_names.tolist()
        ##-----------------------------------------------------------------------------.
        # Check if polygons have been removed and update labels 
        idx_removed_polygons = [len(elem) == 0 for i, elem in enumerate(PolyArrayList)]
        if (all(idx_removed_polygons)):
            raise ValueError("All polygons have been deleted. Please restart.")
        if (any(idx_removed_polygons)):
            # Update labels and PolyArrayList 
            PolyArrayList = [p for (p, idx_remove) in zip(PolyArrayList, idx_removed_polygons) if not idx_remove]
            labels = [l for (l, idx_remove) in zip(labels, idx_removed_polygons) if not idx_remove]
            if self.option == 'labels_names':
                labels_names = [l for (l, idx_remove) in zip(labels_names, idx_removed_polygons) if not idx_remove]
        #-----------------------------------------------------------------------------.
        # Check if polygons have been added for a given class 
        idx_added_polygons = [len(elem) > 1 for i, elem in enumerate(PolyArrayList)]
        n_added_polygons = [len(elem) for i, elem in enumerate(PolyArrayList)]
        if any(idx_added_polygons):
            # Relabel 
            n_insertion=0
            for i, n_poly in enumerate(n_added_polygons):
                if (n_poly >= 2): 
                    for j in range(n_poly - 1):  
                        labels.insert(n_insertion + i, labels[n_insertion + i]) 
                        if self.option == 'labels_names':
                            labels_names.insert(n_insertion + i, labels_names[n_insertion + i]) 
                        n_insertion = n_insertion + 1
        # Unnest the list of Arrays
        PolyArrayList = list(chain.from_iterable(PolyArrayList))
        #-----------------------------------------------------------------------------.
        # - Define the new geopandas PolyClassifier  
        _create_gpd(self, 
                    PolyArrayList = PolyArrayList,
                    labels = np.array(labels),
                    labels_names = np.array(labels_names))
#-----------------------------------------------------------------------------.