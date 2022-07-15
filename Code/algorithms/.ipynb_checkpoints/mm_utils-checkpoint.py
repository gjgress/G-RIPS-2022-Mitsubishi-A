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
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString # To create line geometries that can be used in a GeoDataFrame
from ctypes import *
from ctypes import wintypes
from scipy import spatial # for proximity fuse

# globals
VERSION = '1.0'
    
def run(sim, tracks_edges_list = None, tracks_nodes_list = None, network_edges_list = None, network_nodes_list = None, parallel=False, workers = os.cpu_count()):
    '''
    This is a higher level function which takes a list (or tuple) of map-matching datasets and runs them sequentially through the given sim. This way you can write your algorithm for the simplest situation, and use this when you want to test on a bigger scale
    Args:
        sim: a map-matching algorithm object. Needs to have a run() function which can handle the other variables
        network_nodes_list: a list containing network node datasets (optional; if your algorithm requires it)
        network_edges_list: a list containing network edge datasets (optional; if your algorithm requires it)
        tracks_nodes_list: a list containing track nodes datasets (optional; if your algorithm requires it)
        tracks_edges_list: a list containing track edge datasets (optional; if your algorithm requires it)
    All of the lists must be of the same length
    '''
    tn = tracks_nodes_list
    te = tracks_edges_list
    nn = network_nodes_list
    ne = network_edges_list

    arglist = [te, tn, nn, ne]
    arglist = list(filter(None, arglist))
    

    results = []
    for i in range(len(arglist[0])):
        arglisti = []
#            for j in range(len(arglist)): # Also broken for now
#                arglisti.append(arglist[j][i])
        results.append(sim.run(te[i],tn[i],ne[i],nn[i], return_results=True))
    
    return results
    
def plot(network, input_data, results, fs = (8,8)):
    '''
    Args:
        network: a MultiDigraph, preferably directly from OSM
        input_data: a dataset that matplotlib can plot. Needs to share the same CRS as network
        results: a dataset that matplotlib can plot. Needs to share the same CRS as network
    '''
    fig, ax = ox.plot_graph(network,show=False, close=False, figsize= fs)
    input_data.plot(ax=ax)
    results.plot(ax=ax, color="red")
    
def evaluate(prediction, gt, matchid = 'index'):
    '''
    Args:
        prediction: a GeoDataFrame consisting of the matches found by the algorithm
        gt: a GeoDataFrame consisting of the actual network nodes/edges traversed (ground truth). Should be indexed similar to prediction. (Note: ground truth has a specific meaning in literature, which we misuse here. gt does not need to be the literal ground truth, but simply the route taken as truth for comparison)
        match: the column with which to match prediction and gt. Standard method is by matching geometries, but indexing columns can be superior if you know they are consistent between prediction and gt
    '''
    if (type(prediction) == list or type(prediction) == tuple):
        error = []
        for i in range(len(prediction)):
            error.append(evaluate(prediction[i], gt[i], matchid))
    else:
        if type(gt) == list:
            gt = gt[0] # Dask Delayed returns a list of one element...
            # Temporary workaround
        try:
            np.intersect1d(gt[matchid], prediction[matchid], return_indices=True)
        except:
            raise Exception('Currently only array-like objects can be matched; please choose a different match column.')
        else:
            evalint = gt.loc[np.intersect1d(gt[matchid], prediction[matchid], return_indices=True)[1]]
            evalxor = pd.concat([gt.overlay(evalint, how="difference"), prediction.overlay(evalint, how = "difference")]) # This may seem similar to directly using overlay (symmetric difference), but not so. The difference here is that the intersection is found by matchid; then we take the geometries in the prediction/gt and subtract it out.
                # Why do this? Suppose that somehow your geometries were truncated, i.e (1.00001, 1.00001) => (1,1). Technically, these geometries are distinct, so overlay would put these in the xor GDF. But if we have a column like id which we are certain aligns with both datasets, then we can match that way, and ensure they are correctly put into the evalint set, and our evalxor length is corrected.
            d0 = sum(gt.length)
            ddiff = sum(evalxor.length)

            error = ddiff/d0
    # This error formula was created in https://doi.org/10.1145/1653771.1653818
    return error # Lower is better (zero is perfect)


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
    if columns:
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
    input_edges = pd.concat([input_edges,gpd.GeoDataFrame(coldict)]) # Put the separated LineStrings into a GeoDataFrame, along with any other columns supplied
    
    return input_edges

def fuse(main_data, unfused_data, timestampcol, fuse_method):
    '''
    A barebones method to fuse data measured asynchronously.
    main_data: the data you'd like to align to, as a (geo)pandas (geo)dataframe. This dataframe won't be changed; the other data will be aligned to it instead.
    unfused_data: a dataframe or a list of dataframes which you would like to align to main_data
    timestampcol: a str containing the name of the timestamp column in main_data and unfused_data
    fuse_method: a str or list of str. must be one of: 'nearest neighbor', 'average', 'blah'.
If a list of str is provided, it must have the same length as unfused_data.
    
    Note: it is essential that the first column for all datasets be the timestamps, and that they all share the same relative formatting (i.e. 1000 needs to mean the same thing to both data sets). If you aren't sure if your time format will work, the simplest way to do this is to convert everything to UNIX time.
    '''
    result = main_data
    # Now we check to make sure everything is formatted correctly
    if (type(unfused_data)==list and type(fuse_method)==list):
        if len(unfused_data) != len(fuse_method):
            raise Exception('Data list and method list do not have equal length!')
    
    if type(unfused_data)==list:
        for i in range(len(unfused_data)):
            if type(fuse_method) == list:
                fuse = fuse_method[i]
            else:
                fuse = fuse_method
            result = fuse_simple(result,unfused_data[i],timestampcol,fuse_m=fuse)
    else:
        result = fuse_simple(result,unfused_data, timestampcol, fuse_method)
    return result

# Note-- you can but probably shouldn't use this function directly. fuse() will pass to this helper function as needed
def fuse_simple(df1,df2,timestampcol,fuse_m):
        # I think this works fine but may be good to create a test case
    if fuse_m == 'nearest neighbor':
        base_df2 = df2[timestampcol].to_numpy()   
        vals = []
        for i in range(len(df1)):
            k = np.argmin(np.abs(base_df2 - df1[timestampcol][i]))
            vals.append(df2.iloc[k,1])
        result = df1.copy()
        result[df2.columns[1]] = vals
        return result
    
    elif fuse_m == 'average':
        # I think this works fine but may be good to create a test case
        df2_time = df2[timestampcol].to_numpy() 
        df2_vals = df2.iloc[:,1].to_numpy() 
        vals = []
        for i in range(len(df1)):
            if i == 0:
                prox_vals = df2_vals[df2_time <= df1[timestampcol][i]+(df1[timestampcol][i+1]-df1[timestampcol][i])/2]
                if len(prox_vals) == 0: # If there are no points in the interval, just pick the closest one
                    k = np.argmin(np.abs(df2_time - df1[timestampcol][i]))
                    newval = df2_vals[k]
                else:
                    newval = np.average(prox_vals)
                vals.append(newval)
            elif i == len(df1)-1:
                prox_vals = df2_vals[df2_time >= df1[timestampcol][i-1]+(df1[timestampcol][i]-df1[timestampcol][i-1])/2]
                if len(prox_vals) == 0: # If there are no points in the interval, just pick the closest one
                    k = np.argmin(np.abs(df2_time - df1[timestampcol][i]))
                    newval = df2_vals[k]
                else:
                    newval = np.average(prox_vals)
                vals.append(newval)
            else:
                prox_vals = df2_vals[(df2_time <= df1[timestampcol][i]+(df1[timestampcol][i+1]-df1[timestampcol][i])/2) & (df2_time >= df1[timestampcol][i-1]+(df1[timestampcol][i]-df1[timestampcol][i-1])/2)]
                if len(prox_vals) == 0: # If there are no points in its interval, just pick the closest one
                    k = np.argmin(np.abs(df2_time - df1[timestampcol][i]))
                    newval = df2_vals[k]
                vals.append(newval)
        result = df1.copy()
        result[df2.columns[1]] = vals
        return result
    
    ## Other methods to consider implementing:
    # - K-Means
    # - PDA (Probabilistic Data Association)
    # - JPDA (Joint PDA)
    # - MHT (Multiple Hypothesis Test)
    # - JPDA-D (Distributed JPDA)
    # - MHT-D (Distributed MHT)
    # - Graphical models?
    # - See more from here: https://www.hindawi.com/journals/tswj/2013/704504/
    elif fuse_m == 'blah':
        return df1
    else:
        raise Exception("Not a valid fusion method (must be 'nearest neighbor', 'average', or 'blah')!")
        
def df_to_network(df, buffer = 0.002, ntype = 'drive', as_gdf = True, *args):
    '''
    A helper function which takes a GeoDataFrame (coordinates, lines, anything) and downloads a road network from OSM for the surrounding area
    Args:
    df: a GeoDataFrame from which to generate the network
    buffer: a buffer distance (in lat/lon) from which to expand the bbox (optional, default 0.002)
    ntype: network type to download (optional, default 'drive')
    as_gdf: Boolean, returns as gdf if True, and as OSM network if false
    *args: optional arguments to pass to osmnx.graph_from_bbox()
    '''
    miny, minx, maxy, maxx = df.geometry.total_bounds
    network = ox.graph_from_bbox(maxx+buffer, minx-buffer, maxy+buffer, miny-buffer, network_type=ntype, *args)
    if as_gdf:
        networknodes, networkedges = ox.graph_to_gdfs(network)
        return [networknodes, networkedges]
    else:
        return network