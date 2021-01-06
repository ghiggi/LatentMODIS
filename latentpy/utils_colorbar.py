#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:09:26 2020

@author: ghiggi
"""
import numpy as np

# skimage.color

def rgb_to_hex(rgb, alpha=False):
    # Assume 0 255
    if alpha is False:
        rgb = rgb.reshape(3)
        return '#{:02X}{:02X}{:02X}'.format(*rgb)
    if alpha is True:
        rgb = rgb.reshape(4)
        return '#{:02X}{:02X}{:02X}{:02x}'.format(*rgb)

# 0 to 1 
# matplotlib.colors.to_hex([ 0.47, 0.0, 1.0, 0.5 ], keep_alpha=True)


def hex_to_rgb(hex_str: str) -> np.ndarray:
    hex_str = hex_str.strip()

    if hex_str[0] == '#':
        hex_str = hex_str[1:]

    if len(hex_str) != 6:
        raise ValueError('Input #{} is not in #RRGGBB format.'.format(hex_str))

    r, g, b = hex_str[:2], hex_str[2:4], hex_str[4:]
    rgb = (int(n, base=16) for n in (r, g, b))
    return np.array(rgb)

# def vectorize_rgb(rgb): 
#     ## Take matrix with three columns  and
#     np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) 
#     ## To divide by 255
    
def get_c_cmap_from_color_dict(color_dict, labels): 
    """
    # Retrieve c and cmap argument for plt.scatter provided a custom color dictionary 
    # assign_dict_colors = lambda x : campaign_colors_dict[x]
    # c_names = list(map(assign_dict_colors, experiments))
    """
    c_names = [color_dict[x] for x in labels]
    # Retrieve c integer values 
    c, c_unique_name = pd.factorize(c_names, sort=False)
    # Create cmap
    cmap = mpl.colors.ListedColormap(c_unique_name)
    # Return object 
    return[c, cmap]


# Define functions to color each datapoint to the associated MNIST label 
def get_c_cmap_from_color_dict(color_dict, labels): 
    """
    # Retrieve c and cmap argument for plt.scatter provided a custom color dictionary 
    # assign_dict_colors = lambda x : campaign_colors_dict[x]
    # c_names = list(map(assign_dict_colors, experiments))
    """
    import pandas as pd
    import matplotlib as mpl
    c_names = [color_dict[x] for x in labels]
    # Retrieve c integer values 
    c, c_unique_name = pd.factorize(c_names, sort=False)
    # Create cmap
    cmap = mpl.colors.ListedColormap(c_unique_name)
    # Return object 
    return[c, cmap]

def get_legend_handles_from_colors_dict(colors_dict, marker='o'):
    """
    Retrieve 'handles' for the matplotlib.pyplot.legend
    # marker : "s" = filled square, 'o' = filled circle
    # marker : "PATCH" for filled large rectangles 
    """
    import matplotlib as mpl
    if (marker == 'PATCH'):
        # PATCH ('filled large rectangle')
        handles = []
        for key in colors_dict:
            data_key = mpl.patches.Patch(facecolor=colors_dict[key],
                                         edgecolor=colors_dict[key],
                                         label=key)
            handles.append(data_key)    
    else:
        # Classical Markers
        handles = []
        for key in colors_dict:
            data_key = mpl.lines.Line2D([0], [0], linewidth=0, 
                                        marker=marker, label=key, 
                                        markerfacecolor=colors_dict[key], 
                                        markeredgecolor=colors_dict[key], 
                                        markersize=3)
            handles.append(data_key)    
    return(handles)
 
