#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:05:24 2020

@author: ghiggi
"""
import xarray as xr
import pandas as pd

def reshape_df_to_ds(df, s_coords, t_coords):
    # Check s_coords are column name of df 
    if (len(s_coords) != 2):
        raise ValueError('Provide valid name for x and y dimensions')
    s_valid = [s in df.columns for s in s_coords]
    if not all(s_valid): 
        raise ValueError('Provide valid name for the spatial dimensions')
    # Check t_coords is a column name of df 
    if not (isinstance(t_coords, str)):
        raise ValueError('Provide t_coords as string')
    if not (t_coords in df.columns): 
        raise ValueError('Provide valid name for the time dimension')
    # Set indexes (space + time)
    keys = s_coords + [t_coords]
    df = df.set_index(keys=keys)   
    # df.index
    # Reshape to xarray Dataset
    ds = df.to_xarray()     
    return ds 
    
def reshape_to_RGB_composite(df, RGB_matrix, dims=None, name="Composite"):
    # Check same number of rows 
    if (RGB_matrix.shape[0] != df.shape[0]):
        raise ValueError('df and RGB_matrix must have same number of rows')
    # Check RGB matrix has 3 columns 
    if (RGB_matrix.shape[1] != 3):
        raise ValueError('The RGB matrix must have 3 columns')
    # Check if dims is required and its validity
    # - At least 2 levels (x and y) are required for 2D image
    if ((df.index.nlevels == 1)):  
        if dims is None: 
            raise ValueError("Specify valid dims names to reshape to xarray")
    if dims is not None: 
        # Remove from dims what is already an index 
        # - TODO 
        # Check that remaining are columns of pandas
        valid_dims = [dim in df.columns for dim in dims]
        if not all(valid_dims): 
            raise ValueError('Provide valid dimensions names')
    #-------------------------------------------------------------------------.
    # Add RGB matrix to df 
    df["R"] = RGB_matrix[:,0]
    df["G"] = RGB_matrix[:,1]
    df["B"] = RGB_matrix[:,2]
    # Set indexes if not yet specified 
    if ((df.index.nlevels == 1) or (dims is not None)):
        df = df.set_index(keys=dims)   
    # Reshape to xarray Dataset
    ds_RGB = df[['R','G','B']].to_xarray()   
    # Ensure x y and time are sorted
    for dim in ds_RGB.dims:
        ds_RGB = ds_RGB.sortby(dim)
    # Create RGB DataArray
    da_RGB = ds_RGB.to_stacked_array(new_dim="RGB", sample_dims=ds_RGB.dims, name=name)
    # Return RGB data array
    return da_RGB
