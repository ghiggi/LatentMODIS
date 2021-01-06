#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:23:20 2020

@author: ghiggi
"""
import os 
import glob 
import numpy as np

def find_MODIS_L1B_paths(data_dir,
                         satellite = "Aqua",
                         product = "MYD021KM",
                         collection = "61"):
    """
    Return L1B radiance and geolocation data filepaths for MODIS

    Parameters
    ----------
    data_dir : str
        Directory path containing the MODIS folder. 
        
    Returns
    -------
    geo_filepaths : list
        List of path of MYD03 geolocation files
    l1b_filepaths : list
        List of paths of L1B radiance (i.e. MYD021KM) files
    """
    #-------------------------------------------------------------------------. 
    # Getting all geolocation files
    geo_filepaths_raw = glob.glob(os.path.join(data_dir, 'MODIS',satellite,collection,'MYD03',
                                      '[1-2][0-9]*/*/*.hdf'))
    geo_filepaths_raw = np.array(sorted(geo_filepaths_raw))
    # Getting all L1B radiance files
    l1b_filepaths_raw = glob.glob(os.path.join(data_dir, 'MODIS',satellite,collection, product,
                                      '[1-2][0-9]*/*/*.hdf'))
    l1b_filepaths_raw = np.array(sorted(l1b_filepaths_raw))
    #-------------------------------------------------------------------------.
    # Extracting the part of filename to check
    check_str_geo = []
    for i_f_geo, f_geo in enumerate(geo_filepaths_raw):
        check_str_geo.append('.'.join(f_geo.split('.')[-5:-2]))
    check_str_l1b = []
    for i_f_l1b, f_l1b in enumerate(l1b_filepaths_raw):
        check_str_l1b.append('.'.join(f_l1b.split('.')[-5:-2]))

    check_str_geo = np.array(check_str_geo)
    check_str_l1b = np.array(check_str_l1b)
    #-------------------------------------------------------------------------.
    # Finding common entries among the part to check
    common, idx_comm1, idx_comm2 = np.intersect1d(check_str_geo, check_str_l1b, return_indices=True)
    #-------------------------------------------------------------------------.
    # Returning filepaths for L1B data with available geolocation data
    return geo_filepaths_raw[idx_comm1].tolist(), l1b_filepaths_raw[idx_comm2].tolist()

def find_MODIS_L2_paths(data_dir,
                        satellite = "Aqua",
                        product = "MYD035_L2",
                        collection = "61"):
    """
    Return L2 data filepaths for MODIS

    Parameters
    ----------
    data_dir : str
        Directory path containing the MODIS folder. 
        
    Returns
    -------
    l2_filepaths : list
        List of paths of L2 specified product (i.e. MYD035_L2) files
    """
    #-------------------------------------------------------------------------. 
    # Getting all L2 product files
    l2_filepaths_raw = glob.glob(os.path.join(data_dir, 'MODIS',satellite,collection, product,
                                      '[1-2][0-9]*/*/*.hdf'))
    l2_filepaths_raw = sorted(l2_filepaths_raw)
    return(l2_filepaths_raw)