# -*- coding: utf-8 -*-
# Fielname = fmm_bin.py

"""
FMM Algorithm Wrapper
Created on 2022-06-27
Last Modified: 2022-07-21
@author: gjgress
This python module is a wrapper which allows one to use fmm more simply.
It still requires fmm to be installed separately.

Update (2023-06-19): The python module for FMM is no longer functional due to deprecated dependencies. So, one must use Docker to use FMM. Rewriting this python module to be compatible with the Docker implementation of FMM would require a whole restructuring, so this module is deprecated indefinitely.
"""

# import
import os
import shutil
import struct
import platform
import math
import numpy as np
import osmnx as ox
import geopandas as gpd
import pandas as pd
from ctypes import *
from ctypes import wintypes
import tempfile
from fmm import Network
from fmm import NetworkGraph
from fmm import FastMapMatch
from fmm import UBODTGenAlgorithm
from fmm import UBODT
from fmm import FastMapMatchConfig
from fmm import GPSConfig
from fmm import ResultConfig
import sys

# globals
VERSION = '1.0'

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "network_nodes.shp")
    filepath_edges = os.path.join(filepath, "network_edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

def extract_cpath(cpath):
        if (cpath==''):
            return []
        elif type(cpath) == float:
            if math.isnan(float(cpath)):
                return []
            else:
                return [int(cpath)]
        else:
            return [int(s) for s in cpath.split(',')]
    
### Required    
class FMM(object):
    '''
    '''
    def __init__(self, cfg):
        '''
        Args:
            None
        '''
        # algorithm description
        self.input = ['geometry']
        self.output = ['geometry']
        self.batch = True
        self.results = None
        self.config = cfg
        # algorithm vars
        this_dir = os.path.dirname(__file__)

    ### Required
    def run(self, input_edges, input_nodes=None, network_edges=None, network_nodes=None,   return_results=False):
        '''
        main procedure of the algorithm
        Args:
            input_edges: a GeoDataFrame (and so must have a 'geometry' column) consisting of LineStrings
            input_nodes*: a GeoDataFrame (and so must have a 'geometry' column) consisting of Points
            network edges/nodes: FMM is picky about the network structure, so its simpler to ignore the given network and download from OSM... however, if given, it will try its best to match its results with the network edges/nodes supplied
            return_results: By default the run() function simply passes the results to a local variable, but sometimes it is preferable to also return the results directly
            * FMM doesn't actually use input_nodes, so the inclusion here is superfluous
        '''
        #input_edges = set_of_inputs[set_of_inputs.geom_type == 'LineString'].geometry
        
        
#        if not os.path.exists("temp/"):
#            os.makedirs("temp/")
#        else: 
#            print()

        # So, Dask Delayed can be silly, and when computed, gives us a list (of 1 element)
        # Obviously we want the element and not the list, so we need to unpack it if so
        # Typically the way to handle this is to set nout when initializing the Dask Delayed
        # But because we are converting a Dask Bag to Dask Delayed, we can't actually do that....
        # I don't know of any other way to fix this yet.
        if type(input_edges) == list:
            try:
                input_edges = input_edges[0]
            except:
                raise Exception('Cannot unpack list')
        if type(input_nodes) == list:
            try:
                input_nodes = input_nodes[0]
            except:
                raise Exception('Cannot unpack list')
        if type(network_edges) == list:
            try:
                network_edges = network_edges[0]
            except:
                raise Exception('Cannot unpack list')        
        if type(input_edges) == list:
            try:
                network_nodes = network_nodes[0]
            except:
                raise Exception('Cannot unpack list')
    
        # Write to Shapefile (just make a copy)
        with tempfile.NamedTemporaryFile() as tmp1:
            
            #outfp = "temp/input_edges.shp"
            input_edges.geometry.to_file(tmp1.name + '.shp')
        
        if network_edges is None:
            networkDigraph = df_to_network(input_edges, as_gdf = False)    
            save_graph_shapefile_directional(networkDigraph, filepath='temp')
            network = Network('temp/network_edges.shp', 'fid', 'u', 'v')
            
            network_nodes, network_edges = ox.graph_to_gdfs(networkDigraph)
            
            # Import our network edges for later
            network_gdf = gpd.read_file('temp/network_edges.shp')
            network_gdf.fid = network_gdf.fid.astype(int)
        else:
            network_edges.index.names = ['fid']
            with tempfile.NamedTemporaryFile() as tmp2:
                network_edges.to_file(tmp2.name + '.shp',index = True)
                network = Network(tmp2.name + '.shp', "fid", "u", "v")
                
                # Import our network edges for later
                network_gdf = gpd.read_file(tmp2.name + '.shp')
                network_gdf.fid = network_gdf.fid.astype(int)
#        print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
        
        graph = NetworkGraph(network)

    ### Precompute an UBODT table

    # Can be skipped if you already generated an ubodt file
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        with tempfile.NamedTemporaryFile() as tmp3:
            status = ubodt_gen.generate_ubodt(tmp3.name, 0.02, binary=False, use_omp=True)
#        print(status)

        ### Read UBODT

            ubodt = UBODT.read_ubodt_csv(tmp3.name)

        ### Create FMM model
        model = FastMapMatch(network,graph,ubodt)

        input_config = GPSConfig()
        input_config.file = tmp1.name + '.shp'
        input_config.id = "FID"

        result_config = ResultConfig()
        with tempfile.NamedTemporaryFile() as tmp4:
            result_config.file = tmp4.name + '.csv'
            result_config.output_config.write_opath = True
#        print(result_config.to_string())

            result = model.match_gps_file(input_config, result_config, self.config)

    # Now we post-process

            resultdf = pd.read_csv(tmp4.name + '.csv', sep=";")

            # Extract cpath as list of int
            resultdf["cpath_list"] = resultdf.apply(lambda row: extract_cpath(row.cpath), axis=1)

            # Extract all the edges matched
            all_edge_ids = np.unique(np.hstack(resultdf.cpath_list)).tolist()

        edges_df = network_gdf[network_gdf.fid.isin(all_edge_ids)].reset_index()
        edges_df["points"] = edges_df.apply(lambda row: len(row.geometry.coords), axis=1)
        
        # fmm does not identify matched nodes, but only matched edges. So we will only export matched edges
        output = edges_df[edges_df.geom_type == 'LineString']
        
        ## results
        # If a network was provided, then we need to find the nearest match from our downloaded network
        ##### OK, this is much harder than I thought, so for now this is WIP.
        # While ideally we'd simply do a nearest neighbors search, match the edges 1-1, and return the matched edges,this doesn't actually work
        # This is because a given road may appear multiple times in a network...
        # For example, take a divided highway-- technically the lanes are part of the same road
        # But they connect very differently, so OSM treats them as separate roads occupying the same space
        # If you naively try a nearest neighbor search, you'll get both of these routes
        # Of course, only one of the routes was actually traversed, but it is rather difficult to algorithmically ascertain which one
        # And if you return the wrong one, it would appear as a 'invalid guess' to your evaluation method
        # Which would erroneously increase your error
        # I haven't figured out a robust solution yet (maybe require ids to align with osm ids?)
        # So for now I'm just returning the OSM matched route
        if network_edges is None:
            self.results = output
        else:
            #matched_nodes = output.sjoin_nearest(network_nodes, how = 'right')
            self.results = output
        self.network = network_edges # This is often used in other utilities
        if input_nodes is None:
            self.input_data = input_edges # This is often used in other utilities
        else:
            self.input_data = pd.concat([input_nodes,input_edges]) # This is often used in other utilities

            
        if return_results:
            return self.results
        
        
        
    def plot(self):
    
        fig, ax = ox.plot_graph(self.network,show=False, close=False)
        self.input_data.plot(ax=ax)
        self.results[1].plot(ax=ax, color="red")
    
    def evaluate(self, gt, match = 'geometry'):
        '''
        Author: Gabriel Gress
        '''
        evalint = gt.loc[np.intersect1d(gt[match], pred_edges[match], return_indices=True)[2]]
        evalxor = pd.concat([gt.overlay(evalint, how="difference"),self.results.overlay(evalint, how = "difference")]) # This may seem similar to directly using overlay (symmetric difference), but not so. The difference here is that the intersection is found by matchid; then we take the geometries in the prediction/gt and subtract it out.
            # Why do this? Suppose that somehow your geometries were truncated, i.e (1.00001, 1.00001) => (1,1). Technically, these geometries are distinct, so overlay would put these in the xor GDF. But if we have a column like id which we are certain aligns with both datasets, then we can match that way, and ensure they are correctly put into the evalint set, and our evalxor length is corrected.
        d0 = np.sum(gt['length'])
        ddiff = np.sum(evalxor['length'])
        
        # This error formula was created in https://doi.org/10.1145/1653771.1653818
        return ddiff/d0 # Lower is better (zero is perfect)

    ###Required
    def results(self):
        '''
        Returns:
            algorithm results as specified in self.output
        '''
        return self.results
    
def df_to_network(df, buffer = 0.002, ntype = 'drive', as_gdf = True, *args):
    '''
    A helper function which takes a GeoDataFrame (coordinates, lines, anything) and downloads a road network from OSM for the surrounding area
    Args:
    df: a GeoDataFrame from which to generate the network
    buffer: a buffer distance (in lat/lon) from which to expand the bbox (optional, default 0.002)
    ntype: network type to download (optional, default 'drive')
    as_gdf: Boolean, returns as gdf if True, and as OSM network if false
    *args: optional arguments to pass to osmnx.graph_from_bbox()
    
    Author: Gabriel Gress
    '''
    miny, minx, maxy, maxx = df.geometry.total_bounds
    network = ox.graph_from_bbox(maxx+buffer, minx-buffer, maxy+buffer, miny-buffer, network_type=ntype, *args)
    if as_gdf:
        networknodes, networkedges = ox.graph_to_gdfs(network)
        return [networknodes, networkedges]
    else:
        return network# -*- coding: utf-8 -*-
# Fielname = fmm_bin.py
