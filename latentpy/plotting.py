#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:37:41 2020

@author: ghiggi
"""
import vaex 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d as mplot3d  
import napari
import cartopy.crs as ccrs
from grid_strategy import strategies

from .utils_colorbar import get_c_cmap_from_color_dict
from .utils_colorbar import get_legend_handles_from_colors_dict

def convert_to_numpy(arr):
    if isinstance(arr, vaex.dataframe.DataFrameArrays):
        arr = np.stack(arr.to_arrays()).transpose()
        return arr
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise ValueError("Provide either vaex.dataframe.DataFrameArrays or a np.ndarray")
    return 

# def RGB_from_latent3D(embedding, scaler=None):
#     """Return RGB color between 0 and 1."""
#     from sklearn.preprocessing import MinMaxScaler
#     # from vaex.ml import MinMaxScaler
#     flag_scaler_provided = False
#     #-------------------------------------------------------------------------.
#     # Checks has 3 columns 
#     if (embedding.shape[1] != 3):
#         raise ValueError("Provide array/dataframe with 3 columns")
#     # Check a valid scaler is provided 
#     if (scaler is not None):
#         flag_scaler_provided = True  
#         valid_scaler = "<class 'sklearn.preprocessing._data.MinMaxScaler'>"
#         if not str(type(scaler)) == valid_scaler:
#             raise ValueError("Provide a sklearn.preprocessing.MinMaxScaler object")
#     #-------------------------------------------------------------------------.
#     if scaler is None:
#         # Define min max scaler
#         scaler = MinMaxScaler()
#         scaler = scaler.fit(embedding)
#     #-------------------------------------------------------------------------. 
#     # Scale the data        
#     RGB = scaler.transform(embedding)
#     #-------------------------------------------------------------------------. 
#     # If some data are below or above [0,1] set to 0 and 1
#     RGB = np.clip(RGB, a_min=0, a_max=1)
#     #-------------------------------------------------------------------------. 
#     if (flag_scaler_provided is True):
#         return RGB
#     else: 
#         return scaler, RGB
#     #-------------------------------------------------------------------------. 
 
    
 
def minmax(x):
   return [np.min(x),np.max(x)] 

def plot_scatter2D(embedding, color, title="", xlab="",ylab=""):    
    embedding = convert_to_numpy(embedding)
    # Plot scatter 
    # plt.figure(figsize=cm2inch(20, 20), dpi=1000)
    # plt.style.use('dark_background')
    plt.scatter(embedding[:, 0], embedding[:, 1], \
                c=color,
                marker='.',
                s=0.8, 
                edgecolors='none')
    plt.xlim(minmax(embedding[:,0]))
    plt.ylim(minmax(embedding[:,1]))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    #  plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=12)
    # cbar = plt.colorbar() # aspect= ... for width
    # cbar.set_label(variable, rotation=270)
    # cbar.ax.get_yaxis().labelpad = 15         
    
def plot_pca2D(embedding, color, title="PCA 2D embedding"):    
    plot_scatter2D(embedding, color, title=title, xlab="PC1", ylab="PC2")

def plot_umap2D(embedding, color, title="UMAP 2D embedding"): 
    plot_scatter2D(embedding, color, title=title, 
                   xlab="Latent dimension 1",
                   ylab="Latent dimension 2")

def plot_scatter3D(embedding, color, azim=40, elev=10, title=""): 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.patch.set_facecolor('xkcd:black')
    ax.set_facecolor('xkcd:black')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(embedding[:, 0], 
               embedding[:, 1],
               embedding[:, 2],
               c = color,
               s = 0.1,
               marker = '.')
    ax.set_xlim(minmax(embedding[:, 0]))
    ax.set_xlim(minmax(embedding[:, 1]))
    ax.set_xlim(minmax(embedding[:, 2]))
    ax.set_title(title)
    # ax.axis('off')

def plot_scatter3D_napari(embedding, color, name, axis_labels, size = 0.05):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3, axis_labels=axis_labels)   
        viewer.add_points(embedding,
                          symbol = 'o',
                          size = size, 
                          face_color = color,
                          name = name)
        
def plot_scatter2D_napari(embedding, color, name, axis_labels, size = 0.05):
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=2, axis_labels=axis_labels)   
        viewer.add_points(embedding,
                          symbol = 'o',
                          size = size, 
                          face_color = color,
                          name = name)  
        
##-----------------------------------------------------------------------------.
def plot_2D_category(embedding, labels, category_colors_dict,  
                     legend_title = "Categories", title=""):
    embedding = convert_to_numpy(embedding)
    # Display embeddings colored by labels (aka classes
    c, cmap = get_c_cmap_from_color_dict(category_colors_dict, labels=labels)  
    # plt.figure(figsize=cm2inch(20, 20), dpi=1000)
    # plt.style.use('dark_background')
    plt.scatter(embedding[:, 0], embedding[:, 1],  
                c=c, cmap=cmap,  
                marker='.', s=0.8, edgecolors='none')
    plt.xlim(minmax(embedding))
    plt.ylim(minmax(embedding))
    plt.gca().set_aspect('equal', 'datalim')
    box = plt.gca().get_position() # Shrink axis on the left to create space for the legend
    plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(handles=get_legend_handles_from_colors_dict(category_colors_dict), 
               title=legend_title,  
               bbox_to_anchor=(1,0.5), loc="center left")
    plt.title(title, fontsize=12)      

#-----------------------------------------------------------------------------.
### Xarray 
def plot_hourly_variability_boxplot(da, ylab=''):
    name = da.name
    df = da.to_dataframe().reset_index()
    
    fig, ax = plt.subplots(figsize=(12,5))  
    # ax = sns.boxplot(x='time', y="1", data=df_tmp) # 
    ax = sns.boxplot(x=df.time.dt.hour, y=name, data=df)
    ax.set_xlabel('Hour') 
    ax.set_ylabel(ylab) 
    ax.set_title(name)

def plot_variability_boxplot(da, ylab=''):
    name = da.name
    df = da.to_dataframe().reset_index()
    
    fig, ax = plt.subplots(figsize=(12,5))  
    ax = sns.boxplot(x=df.time, y=name, data=df)
    plt.xticks(rotation=90)
    ax.set_xlabel('Overpass') 
    ax.set_ylabel(ylab) 
    ax.set_title(name)

def plot_bands(ds, bands, timestep, title_prefix="Channel ", fname=None):
    ## TODO
    # scale_colorbar_to_map_size 
    # remove space in between
    # extent based on ds lat/lon  
    # ------------------------------------------------------------------------.
    # - Figure options
    PE_lat, PE_lon = -71.949944, 23.347079
    crs_WGS84 = ccrs.PlateCarree()  
    crs_proj = ccrs.SouthPolarStereo()
    # - Subset data
    ds_tmp = ds.isel(time=timestep)
    timestr_tmp = str(ds_tmp.time.values)
    # ------------------------------------------------------------------------.
    # Define figure
    figsize = (12,6) # (width, height) [in inches]
    fig = plt.figure(figsize = figsize) 
    # Define plot arrangement 
    n_plots = len(bands)
    strat = strategies.RectangularStrategy()
   
    l_GridSpec = strat.get_grid(n_plots)
    l_ax = [None for i in range(n_plots)]
    # ------------------------------------------------------------------------.
    # # - Plot each band
    for i, band in enumerate(bands):
        # - Define vmin and vmax over time 
        vmin_tmp = ds[band].min().compute().values.tolist()
        vmax_tmp = ds[band].max().compute().values.tolist()  
        # - Create subplot
        l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
        l_ax[i].get_gridspec().update(wspace=0) # hspace=0,
        # - Plot the image 
        ds_tmp[band].plot(ax=l_ax[i],
                          transform=crs_proj, 
                          add_colorbar = False,
                          vmin=vmin_tmp, vmax=vmax_tmp)
        # - Set the title 
        l_ax[i].set_title(title_prefix + band)
        # - Add coastlines
        l_ax[i].coastlines() 
        #- Add PE marker 
        l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
        
        #- Add gridlines 
        gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--') 
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
         
        # gl.xlabels_bottom = False
        # gl.xlabels_top = False
        # gl.xlabels_left = False
        # gl.xlabels_right = False
        # gl.ylabels_right = False
    # Add supertitle   
    fig.suptitle(timestr_tmp, fontsize=12)
    # Fig tight layout 
    # fig.tight_layout()
    # Plot or save the fig 
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()    
    return 

## ---------------------------------------------------------------------------.
def plot_latent_spectral_space(da_RGB, df, df_latent, df_latent_colors, t_coords, time_idx = 0):
    # Retrieve data for a specific timestep 
    da = da_RGB.isel({t_coords: time_idx})
    timestep = da[t_coords].values
    idx_df_timestep = df[t_coords] == timestep
    # Retrieve xlim and ylim
    xlim = [ df_latent[:,0].min(), df_latent[:,0].max() ]
    ylim = [ df_latent[:,1].min(), df_latent[:,1].max() ]
    ## -----------------------------------------------------------------------.
    # Define projections 
    crs_ref = ccrs.PlateCarree()       # crs reference
    crs_proj = ccrs.SouthPolarStereo() # crs plot projection
    PE_lat, PE_lon = -71.949944, 23.347079 # Princess Elizabeth station coordinates in WGS84
    ## -----------------------------------------------------------------------.
    fig = plt.figure(figsize = (16,8))
    (ax1, ax2) = fig.subplots(1,2, subplot_kw={'projection': crs_proj})
    ## -----------------------------------------------------------------------.
    ## Plot the map 
    da.plot.imshow(ax=ax1, x='x', y='y', transform=crs_proj)
    # plt.plot(PE_lon, PE_lat, color="red", marker='o', transform=crs_ref)
    ax1.coastlines() 
    ## -----------------------------------------------------------------------.
    ## Plot scatter 
    # - Update the axes projection (to default) 
    ax2.remove()       
    ax2 = fig.add_subplot(1,2,2, projection=None) # 1-based indexing !
    plot_scatter2D(df_latent[idx_df_timestep], color=df_latent_colors[idx_df_timestep], 
                   xlab="Latent dimension 1", ylab="Latent dimension2",
                   title="2D Embedding") 
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    # - Show plot
    fig.tight_layout() 
    plt.show() 
## ---------------------------------------------------------------------------.

def plot_RGB_DataArray(da, timestep=0):
    # - Select timestep
    da_tmp = da.isel(time=timestep)
    ##----------------------------------------------------------------------------.
    # - Define map settings 
    PE_lat, PE_lon = -71.949944, 23.347079
    crs_WGS84 = ccrs.PlateCarree()  
    crs_proj = ccrs.SouthPolarStereo()
    ##----------------------------------------------------------------------------.
    # - Create figure 
    plt.figure(figsize=(10,8))
    ax = plt.subplot(projection=crs_proj)
    da_tmp.plot.imshow(x='x', y='y',transform=crs_proj)
    # - Add coastlines
    ax.coastlines() 
    ## - Add PE marker 
    ax.scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
    ## - Add gridlines 
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--') 
    gl.xlabels_bottom = False
    gl.xlabels_left = False
    gl.xlabels_right = False
    gl.ylabels_right = True
    plt.show() 
    
def plot_RGB_DataArrays(da, title="", time_dim="time", fname=None):
    # ------------------------------------------------------------------------.
    # - Define figure
    figsize = (12,12) # (width, height) [in inches]
    fig = plt.figure(figsize = figsize) 
    # ------------------------------------------------------------------------.
    # - Define map settings
    PE_lat, PE_lon = -71.949944, 23.347079
    crs_WGS84 = ccrs.PlateCarree()  
    crs_proj = ccrs.SouthPolarStereo()
    # ------------------------------------------------------------------------.
    # - Define number of plots
    n_plots = len(da[time_dim])
    # ------------------------------------------------------------------------.
    # - Define plot arrangement 
    strat = strategies.RectangularStrategy()
    l_GridSpec = strat.get_grid(n_plots)
    l_ax = [None for i in range(n_plots)]
    # ------------------------------------------------------------------------.
    # - Plot each timestep
    for i in range(0, n_plots):
        # - Select timestep
        da_tmp = da.isel({time_dim: i})
        # - Select title
        tmp_title = np.datetime_as_string(da_tmp[time_dim].values, unit='s')   
        # - Create subplot
        l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
        l_ax[i].get_gridspec().update(wspace=0) # hspace=0,
        # - Plot the image 
        da_tmp.plot.imshow(x='x', y='y',transform=crs_proj, ax=l_ax[i])
        l_ax[i].set_title(tmp_title)
        # - Add coastlines
        l_ax[i].coastlines() 
        # - Add PE marker 
        l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
        # - Add gridlines 
        gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=2, color='gray', alpha=0.5, linestyle='--') 
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        # Below deprecated
        # gl.xlabels_bottom = False
        # gl.xlabels_left = False
        # gl.xlabels_right = False
        # gl.ylabels_right = True
    #-------------------------------------------------------------------------.
    # - Add supertitle   
    fig.suptitle(title, fontsize=12)
    # Fig tight layout 
    # fig.tight_layout()
    # Plot or save the fig 
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()    
    return


def plot_DataArrayPanels(da, title="", loop_over="time", fname=None, figsize=(12,12), levels=None):
    # ------------------------------------------------------------------------.
    # - Define figure
    #figsize = (12,12) # (width, height) [in inches]
    fig = plt.figure(figsize = figsize) 
    # ------------------------------------------------------------------------.
    # - Define map settings
    PE_lat, PE_lon = -71.949944, 23.347079
    crs_WGS84 = ccrs.PlateCarree()  
    crs_proj = ccrs.SouthPolarStereo()
    # ------------------------------------------------------------------------.
    # - Define number of plots
    n_plots = len(da[loop_over])
    # ------------------------------------------------------------------------.
    # - Define plot arrangement 
    strat = strategies.RectangularStrategy()
    l_GridSpec = strat.get_grid(n_plots)
    l_ax = [None for i in range(n_plots)]
    # ------------------------------------------------------------------------.
    # - Plot each timestep
    for i in range(0, n_plots):
        # - Select timestep
        da_tmp = da.isel({loop_over: i})
        # - Select title
        tmp_title = np.datetime_as_string(da_tmp[loop_over].values, unit='s')   
        # - Create subplot
        l_ax[i] = fig.add_subplot(l_GridSpec[i], projection=crs_proj) 
        l_ax[i].get_gridspec().update(wspace=0) # hspace=0,
        # - Plot the image 
        if levels is None:
            da_tmp.plot(x='x', y='y',transform=crs_proj, ax=l_ax[i])
        else:
            da_tmp.plot(x='x', y='y',transform=crs_proj, ax=l_ax[i], levels=levels)

        l_ax[i].set_title(tmp_title)
        # - Add coastlines
        l_ax[i].coastlines() 
        # - Add PE marker 
        l_ax[i].scatter(PE_lon, PE_lat, c="red", transform=crs_WGS84)
        # - Add gridlines 
        gl = l_ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                               linewidth=2, color='gray', alpha=0.5, linestyle='--') 
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        # Below deprecated
        # gl.xlabels_bottom = False
        # gl.xlabels_left = False
        # gl.xlabels_right = False
        # gl.ylabels_right = True
    #-------------------------------------------------------------------------.
    # - Add supertitle   
    fig.suptitle(title, fontsize=12)
    # Fig tight layout 
    # fig.tight_layout()
    # Plot or save the fig 
    if fname is not None:
        fig.savefig(fname)
    else:
        plt.show()    
    return
