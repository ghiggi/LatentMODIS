#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:50:35 2020

@author: ghiggi
"""
import os
import datetime
import tensorflow as tf 
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP as Original_ParametricUMAP_Loader

#-----------------------------------------------------------------------------.
##########################
## Define RGB scaler  ####
##########################

class latentRGB_Scaler:           
    def __init__(self, embedding):
        # Checks has 3 columns 
        if (embedding.shape[1] != 3):
            raise ValueError("Provide array/dataframe with 3 columns")
        # Normalize to 0-1 
        scaler = MinMaxScaler()
        scaler = scaler.fit(embedding)
        self.scaler = scaler
        
    def transform(self, X): 
        """Return RGB color between 0 and 1."""
        if (X.shape[1] != 3):
            raise ValueError("Provide array/dataframe with 3 columns")
        ##--------------------------------------------------------------------. 
        # Scale the data        
        RGB = self.scaler.transform(X)
        ##--------------------------------------------------------------------. 
        # If some data are below or above [0,1] set to 0 and 1
        RGB = np.clip(RGB, a_min=0, a_max=1)
        return RGB
    
#-----------------------------------------------------------------------------.
#############################################
### Define directory where to save models ###
#############################################
def get_ParametricUMAP_dir(model_dir, 
                           n_components,
                           n_neighbors,
                           min_dist,
                           metric,
                           PCA_preprocessing,
                           PCA_n_components):
    ##------------------------------------------------------------------.
    # Define directory name
    dir_name = "_".join(['ParametricUMAP', str(n_components),
                         'nneighbors', str(n_neighbors),
                         'mindist', str(min_dist),
                         'metric', metric])
    if PCA_preprocessing is True:
        dir_name = "_".join([dir_name, 
                             'PCAncomps', str(PCA_n_components)])
    ##------------------------------------------------------------------.
    # Define directory path 
    dir_path = os.path.join(model_dir, dir_name) 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Return directory paths
    return dir_path

def get_UMAP_dir(model_dir, 
                 n_components,
                 n_neighbors,
                 min_dist,
                 metric,
                 PCA_preprocessing,
                 PCA_n_components):
    ##------------------------------------------------------------------.
    # Define directory name
    dir_name = "_".join(['UMAP', str(n_components),
                         'nneighbors', str(n_neighbors),
                         'mindist', str(min_dist),
                         'metric', metric])
    if PCA_preprocessing is True:
        dir_name = "_".join([dir_name, 
                             'PCAncomps', str(PCA_n_components)])
    ##------------------------------------------------------------------.
    # Define directory path 
    dir_path = os.path.join(model_dir, dir_name) 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Return directory paths
    return dir_path

#----------------------------------------------------------------------------.
################################## 
### Define training functions #### 
################################## 
def train_UMAP(X, 
               model_dir,
               n_components = 2,
               # UMAP settings 
               n_epochs = 200,
               n_neighbors = 15,
               min_dist = 0.1,
               metric="euclidean",
               # PCA settings
               PCA_preprocessing = False, 
               PCA_n_components = None,
               verbose = True):
    ##------------------------------------------------------------------------.
    # Record start time 
    t_i = datetime.datetime.now()
    # Default PCA_n_components
    if PCA_n_components is None: 
        PCA_n_components = min(20, X.shape[1])
    ##------------------------------------------------------------------------.
    # Define model directory and fpaths
    umap_dir = get_UMAP_dir(model_dir = model_dir, 
                            n_components = n_components,
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            metric = metric,
                            PCA_preprocessing = PCA_preprocessing, 
                            PCA_n_components = PCA_n_components)
    pca_fpath = os.path.join(umap_dir,'pca.sav')
    umap_fpath = os.path.join(umap_dir,'umap.sav')
    RGB_scaler_fpath = os.path.join(umap_dir,'RGB_scaler.sav')
    ##------------------------------------------------------------------------.
    ### Perform PCA before UMAP 
    if PCA_preprocessing is True:
        if verbose is True:
            print("Start PCA preprocessing.")  
        # Fit PCA
        pca_nD = PCA(n_components=PCA_n_components)
        pca_nD.fit(X)  
        # Save PCA model
        joblib.dump(pca_nD, pca_fpath)
        # Project data
        X = pca_nD.transform(X)
    ##------------------------------------------------------------------------.
    ### Perform UMAP projection
    if verbose is True:
        print("Start UMAP training.")  
    umap_nD = UMAP(n_components = n_components,
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   metric = metric,
                   verbose = verbose)
    # Fit UMAP
    umap_nD.fit(X)  
    ##------------------------------------------------------------------------.
    ### Save UMAP models (joblib or pickle)
    joblib.dump(umap_nD, umap_fpath)
    ##------------------------------------------------------------------------.
    ### Create RGB scale if n_components == 3
    if n_components == 3:
        if verbose is True:
            print("Start creation of the RGB scaler.") 
        umap3D_latent = umap_nD.embedding_
        umap3D_RGB_scaler = latentRGB_Scaler(umap3D_latent) 
        # - Save the scaler
        joblib.dump(umap3D_RGB_scaler, RGB_scaler_fpath)
    ##------------------------------------------------------------------------. 
    # Display training time info
    t_f = datetime.datetime.now()
    dt = t_f - t_i 
    print("Training completed in", int(dt.seconds/60), "minutes")
    return  

#-----------------------------------------------------------------------------.
def train_ParametricUMAP(X, 
                         model_dir,
                         n_components = 2,
                         # UMAP settings 
                         n_neighbors = 15,
                         min_dist = 0.1,
                         metric="euclidean",
                         # NN settings 
                         n_hidden_layer = 3,
                         n_hidden_layer_neurons = 128,
                         batch_size = 1024,  
                         n_training_epochs = 50,
                         loss_report_frequency = 1,
                         # PCA settings
                         PCA_preprocessing = False, 
                         PCA_n_components = None,
                         verbose=True):
    ##------------------------------------------------------------------------.
    # Record start time 
    t_i = datetime.datetime.now()
    # Default PCA_n_components
    if PCA_n_components is None: 
        PCA_n_components = min(20, X.shape[1])
    ##------------------------------------------------------------------------.
    # Define model directory and fpaths
    umap_dir = get_ParametricUMAP_dir(model_dir = model_dir, 
                                       n_components = n_components,
                                       n_neighbors = n_neighbors,
                                       min_dist = min_dist,
                                       metric = metric,
                                       PCA_preprocessing = PCA_preprocessing, 
                                       PCA_n_components = PCA_n_components)
    pca_fpath = os.path.join(umap_dir,'pca.sav')
    RGB_scaler_fpath = os.path.join(umap_dir,'RGB_scaler.sav')
    ##------------------------------------------------------------------------.
    ### Perform PCA before ParametricUMAP 
    if PCA_preprocessing is True:
        if verbose is True:
            print("Start PCA preprocessing.")  
        # Fit PCA
        pca_nD = PCA(n_components=PCA_n_components)
        pca_nD.fit(X)  
        # Save PCA model
        joblib.dump(pca_nD, pca_fpath)
        # Project data
        X = pca_nD.transform(X)
    ##------------------------------------------------------------------------.
    ### Define NN architecture 
    # - Define optimizer
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # - Define input shape to the encoder 
    dims = X.shape[1:] # (n_columns,)
    # - Define NN encoder  
    encoder_nD = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=dims),

        tf.keras.layers.Dense(units=n_hidden_layer_neurons, activation="relu"),
        tf.keras.layers.Dense(units=n_hidden_layer_neurons, activation="relu"),
        tf.keras.layers.Dense(units=n_hidden_layer_neurons, activation="relu"),
        tf.keras.layers.Dense(units=n_components, name="z"),
    ])
    encoder_nD.summary()
    ##------------------------------------------------------------------------.
    ### Train parametric UMAP 
    if verbose is True:
        print("Start ParametricUMAP training.")  
    umap_nD = ParametricUMAP(n_components = n_components,
                            ## UMAP options
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            metric=metric, 
                            parametric_embedding=True, 
                            parametric_reconstruction=False,
                            verbose=verbose, 
                            ## NN options
                            optimizer=optimizer,
                            batch_size=batch_size,
                            n_training_epochs = n_training_epochs,
                            dims=dims,
                            encoder=encoder_nD, 
                            loss_report_frequency = loss_report_frequency)
    # - Train
    umap_nD.fit(X)
    # - Save
    umap_nD.save(umap_dir, verbose=True)
    ##------------------------------------------------------------------------.
    ### Create RGB scale if n_components == 3
    if n_components == 3:
        if verbose is True:
            print("Start creation of the RGB scaler.")  
        umap3D_latent = umap_nD.embedding_
        umap3D_RGB_scaler = latentRGB_Scaler(umap3D_latent) 
        # - Save the scaler
        joblib.dump(umap3D_RGB_scaler, RGB_scaler_fpath)
    ##------------------------------------------------------------------------.
    # Display training time info
    t_f = datetime.datetime.now()
    dt = t_f - t_i 
    print("Training completed in", int(dt.seconds/60), "minutes")
    return

#-----------------------------------------------------------------------------.
#############################
### Define Model Loaders ####
#############################

class load_UMAP():
    def __init__(self, 
                 model_dir,
                 n_components,
                 n_neighbors, 
                 min_dist, 
                 metric,
                 PCA_preprocessing, 
                 PCA_n_components):
        ##--------------------------------------------------------.
        # Retrieve directory and filepath of models  
        umap_dir = get_UMAP_dir( model_dir = model_dir,
                                 n_components = n_components,
                                 n_neighbors = n_neighbors, 
                                 min_dist = min_dist, 
                                 metric = metric,
                                 PCA_preprocessing = PCA_preprocessing, 
                                 PCA_n_components = PCA_n_components)
        pca_fpath = os.path.join(umap_dir,'pca.sav')
        umap_fpath = os.path.join(umap_dir,'umap.sav')
        RGB_scaler_fpath = os.path.join(umap_dir,'RGB_scaler.sav')
        ##--------------------------------------------------------.
        # Check and load PCA   
        if PCA_preprocessing is True:
            self.PCA_preprocessing = True
            self.PCA = joblib.load(pca_fpath)
        else:
            self.PCA_preprocessing = False
        ##--------------------------------------------------------.
        # Load UMAP 
        self.UMAP = joblib.load(umap_fpath)
        self.n_components = n_components
        ##--------------------------------------------------------.
        # Load RGB scaler if n_components = 3
        if n_components == 3:
            self.RGB_scaler = joblib.load(RGB_scaler_fpath)
        ##--------------------------------------------------------.
        
    def transform(self, X):
        # Preprocess with PCA
        if self.PCA_preprocessing:
            X = self.PCA.transform(X)
        # Project with UMAP
        return self.UMAP.transform(X)

    def transform_to_RGB(self, X):
        if self.n_components != 3:
            raise ValueError("UMAP must project to 3D latent space!")
        return self.RGB_scaler.transform(self.transform(X))
        

class load_ParametricUMAP():
    def __init__(self, 
                 model_dir,
                 n_components,
                 n_neighbors, 
                 min_dist, 
                 metric,
                 PCA_preprocessing, 
                 PCA_n_components,
                 verbose = True):
        ##--------------------------------------------------------.
        # Retrieve directory and filepath of models  
        umap_dir = get_ParametricUMAP_dir(model_dir = model_dir,
                                          n_components = n_components,
                                          n_neighbors = n_neighbors, 
                                          min_dist = min_dist, 
                                          metric = metric,
                                          PCA_preprocessing = PCA_preprocessing, 
                                          PCA_n_components = PCA_n_components)
        pca_fpath = os.path.join(umap_dir,'pca.sav')
        RGB_scaler_fpath = os.path.join(umap_dir,'RGB_scaler.sav')
        ##--------------------------------------------------------.
        # Check and load PCA   
        if PCA_preprocessing is True:
            self.PCA_preprocessing = True
            self.PCA = joblib.load(pca_fpath)
        else:
            self.PCA_preprocessing = False
        ##--------------------------------------------------------.
        # Load ParametricUMAP 
        self.ParametricUMAP = Original_ParametricUMAP_Loader(umap_dir, verbose=verbose) 
        self.n_components = n_components
        ##--------------------------------------------------------.
        # Load RGB scaler if n_components = 3
        if n_components == 3:
            self.RGB_scaler = joblib.load(RGB_scaler_fpath)
        ##--------------------------------------------------------.
        
    def transform(self, X):
        # Preprocess with PCA
        if self.PCA_preprocessing:
            X = self.PCA.transform(X)
        # Project with ParametricUMAP
        return self.ParametricUMAP.transform(X)

    def transform_to_RGB(self, X):
        if self.n_components != 3:
            raise ValueError("ParametricUMAP must project to 3D latent space!")
        return self.RGB_scaler.transform(self.transform(X))

 

    
 











 