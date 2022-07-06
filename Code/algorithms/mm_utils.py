# -*- coding: utf-8 -*-
# Fielname = fmm_bin.py

"""
Algorithm Wrapper
Created on 2022-06-27
@author: gjgress
This python module is a wrapper which includes basic functions for evaluating and plotting. If an algorithm does not (choose to) implement their own methods, this should work as long as one adheres to the format standards.
"""

# import
import os
import struct
import platform
import math
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString # To create line geometries that can be used in a GeoDataFrame
from ctypes import *
from ctypes import wintypes
from scipy import spatial # for proximity fuse

# globals
VERSION = '1.0'
    
### Required     
def plot(self, network, input_data, results):
    '''
    Args:
        network: a MultiDigraph, preferably directly from OSM
        input_data: a dataset that matplotlib can plot. Needs to share the same CRS as network
        results: a dataset that matplotlib can plot. Needs to share the same CRS as network
    '''
    fig, ax = ox.plot_graph(network,show=False, close=False)
    input_data.plot(ax=ax)
    results[1].plot(ax=ax, color="red")

def evaluate(self, prediction, gt, match = 'geometry'):
    '''
    Args:
        prediction: a GeoDataFrame consisting of the matches found by the algorithm
        gt: a GeoDataFrame consisting of the actual network nodes/edges traversed (ground truth). Should be indexed similar to prediction. (Note: ground truth has a specific meaning in literature, which we misuse here. gt does not need to be the literal ground truth, but simply the route taken as truth for comparison)
        match: the column with which to match prediction and gt. Standard method is by matching geometries, but indexing columns can be superior if you know they are consistent between prediction and gt
    '''
    evalint = gt.loc[np.intersect1d(gt[matchid], pred_edges[matchid], return_indices=True)[2]]
    evalxor = gt.overlay(evalint, how="difference").append(self.results.overlay(evalint, how = "difference")) # This may seem similar to directly using overlay (symmetric difference), but not so. The difference here is that the intersection is found by matchid; then we take the geometries in the prediction/gt and subtract it out.
        # Why do this? Suppose that somehow your geometries were truncated, i.e (1.00001, 1.00001) => (1,1). Technically, these geometries are distinct, so overlay would put these in the xor GDF. But if we have a column like id which we are certain aligns with both datasets, then we can match that way, and ensure they are correctly put into the evalint set, and our evalxor length is corrected.

    d0 = np.sum(gt['length'])
    ddiff = np.sum(evalxor['length'])

    # This error formula was created in https://doi.org/10.1145/1653771.1653818
    return ddiff/d0 # Lower is better (zero is perfect)

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    '''
    A fantastic utility provided by fmm. This function converts a MultiDigraph (specifically from OSM) to a ShapeFile, while also including information like one ways. This is immensely useful if your algorithm is designed for ShapeFiles.
    Args:
    G: A MultiDigraph network, preferably from OSM 
    filepath: filepath to export the ShapeFile
    encoding: encoding to use as formatting  
    '''
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

def point_to_traj(input_nodes, columns=None):
    '''
    Oftentimes the raw data is a sequence of coordinates/timestamps.
    Directly converting to GeoDataFrame can be unhelpful, as you will simply have a sequence of Points.
    Often, algorithms would rather work with a 'trajectory', as in, a sequence of points WITH edges.
    This is especially the case for curve-to-curve methods.
    This function is a short util which converts a GDF consisting of solely Points into this format.
    Furthermore, if you have IMU data, you'd like a way to assign that data to edges as well.
    By supplying these columns, this function does its best to assign to edges these values.
    Args:
    input_nodes: The sequence of Points (in order) as a GeoDataFrame
    columns: a dictionary consisting of column data you'd like to assign, with key values being one of the following: 'first' (use first node values), 'average' (average between nodes), or 'last' (use last node values)
    
    Ex: mm_utils.point_to_traj(df_nodes, {'speed', 'average'})
    '''
    
    input_nodes = input_nodes[input_nodes.geom_type == 'Point'] # Just in case you are silly and try to include non-Point geometries
    input_edges = gpd.GeoDataFrame(columns = input_nodes.columns) # Initialize

    biglinestring = [xy for xy in zip(input_nodes['geometry'].x, input_nodes['geometry'].y)] # Form a single huge LineString from input_nodes
    line_segments = list(map(LineString, zip(biglinestring[:-1], biglinestring[1:]))) # Separate the line_segments into pieces

    # Now we will format the other columns in the input_data to be included in our final GeoDataFrame
    coldict = {}
    for column in columns:
        
        # Make sure the column exists
        if np.any(input_nodes.columns.values == column):
            dat = input_nodes[column].to_numpy()

            # Depending on the method, create the adjusted data
            if columns[column] == 'first':
                datval = dat[:-1]
            elif columns[column] == 'average':
                datval = (dat[:-1] + dat[1:])/2
            elif columns[column] == 'last':
                datval = dat[1:]
            else: # I only have the above methods! Don't try to use something else
                print(columns[column] + ' is not a valid assignment method. Ignoring this column.')

            coldict[column] = datval
        else:
            print(column + ' is not a valid column in input nodes. Ignoring this column.')

    coldict['geometry'] = line_segments
    input_edges = input_edges.append(gpd.GeoDataFrame(coldict)) # Put the separated LineStrings into a GeoDataFrame, along with any other columns supplied
    
    return input_edges

def fuse(main_data, data, fuse_method):
    '''
    A barebones method to fuse data measured asynchronously.
    main_data: the data you'd like to align to, as a (geo)pandas (geo)dataframe. This dataframe won't be changed; the other data will be aligned to it instead.
    datalist: a dataframe or a list of dataframes which you would like to align to main_data
    fuse_method: a str or list of str. must be one of: 'proximity', 'average', 'blah'.
If a list of str is provided, it must have the same length as datalist.
    
    Note: it is essential that the first column for all datasets be the timestamps, and that they all share the same relative formatting (i.e. 1000 needs to mean the same thing to both data sets). If you aren't sure if your time format will work, the simplest way to do this is to convert everything to UNIX time.
    '''
        
    # Now we check to make sure everything is formatted correctly
    if (type(data)==list and type(fuse_method)==list):
        if len(data) != len(fuse_method):
            raise Exception('Data list and method list do not have equal length!')
    
    df = main_data
    
    if type(data)==list:
        for i in range(len(data)):
            if type(fuse_method) == list:
                fuse = fuse_method[i]
            else:
                fuse = fuse_method
            df = fuse_simple(df,data[i],fuse_m=fuse)
    else:
        df = fuse_simple(df,data,fuse_method)
    
    return df

# Note-- you can but probably shouldn't use this function directly. fuse() will pass to this helper function as needed
def fuse_simple(df1,df2,fuse_m):
    if fuse_method = 'proximity':
        # Make a kd-tree
        tree = spatial.KDTree(df2.iloc[:,0].values.tolist())
        aligndf2 = 
        ## Finish this
    elif fuse_method = 'average':
        
    elif fuse_method = 'blah':
        
    else:
        raise Exception("Not a valid fusion method (must be 'proximity', 'average', or 'blah')!")
        