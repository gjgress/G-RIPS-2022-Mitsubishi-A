# -*- coding: utf-8 -*-
# Fielname = fmm_bin.py

"""
FMM Algorithm Wrapper
Created on 2022-06-27
@author: gjgress
This python module is a wrapper which allows one to use fmm more simply.
It still requires fmm to be installed separately.
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
from fmm import Network,NetworkGraph,FastMapMatch,UBODTGenAlgorithm,UBODT,FastMapMatchConfig,GPSConfig,ResultConfig

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
    def run(self, network_edges, network_nodes, input_edges, input_nodes=None ):
        '''
        main procedure of the algorithm
        Args:
            network_edges is a GeoDataFrame object preferably created from a MultiDiGraph (the default format for OSM), consisting of LineStrings            
            network_nodes is a GeoDataFrame object preferably created from a MultiDiGraph (the default format for OSM), consisting of Points
            input_edges: a GeoDataFrame (and so must have a 'geometry' column) consisting of LineStrings
            input_nodes*: a GeoDataFrame (and so must have a 'geometry' column) consisting of Points
            * FMM doesn't actually use input_nodes, so the inclusion here is superfluous
        '''
        #input_edges = set_of_inputs[set_of_inputs.geom_type == 'LineString'].geometry
        
        if not os.path.exists("temp/"):
            os.makedirs("temp/")
        else: 
            print()
        
        # Write to Shapefile (just make a copy)
        outfp = "temp/input_edges.shp"
        input_edges.geometry.to_file(outfp)
        
        networkDigraph = ox.graph_from_gdfs(network_nodes, network_edges)
        save_graph_shapefile_directional(networkDigraph, filepath='temp')
        network = Network("temp/network_edges.shp", "fid", "u", "v")
        print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
        graph = NetworkGraph(network)
        
        ### Precompute an UBODT table

        # Can be skipped if you already generated an ubodt file
        ubodt_gen = UBODTGenAlgorithm(network,graph)
        status = ubodt_gen.generate_ubodt("temp/ubodt.txt", 0.02, binary=False, use_omp=True)
        print(status)
        
        ### Read UBODT

        ubodt = UBODT.read_ubodt_csv("temp/ubodt.txt")
        
        ### Create FMM model
        model = FastMapMatch(network,graph,ubodt)
        
        input_config = GPSConfig()
        input_config.file = "temp/input_edges.shp"
        input_config.id = "FID"

        result_config = ResultConfig()
        result_config.file = "temp/mr.csv"
        result_config.output_config.write_opath = True
        print(result_config.to_string())

        result = model.match_gps_file(input_config, result_config, self.config)
        
        # Now we post-process
        
        resultdf = pd.read_csv("temp/mr.csv", sep=";")
        
        # Extract cpath as list of int
        resultdf["cpath_list"] = resultdf.apply(lambda row: extract_cpath(row.cpath), axis=1)

        # Extract all the edges matched
        all_edge_ids = np.unique(np.hstack(resultdf.cpath_list)).tolist()

        # Import our network edges
        network_gdf = gpd.read_file("temp/network_edges.shp")
        network_gdf.fid = network_gdf.fid.astype(int)

        edges_df = network_gdf[network_gdf.fid.isin(all_edge_ids)].reset_index()
        edges_df["points"] = edges_df.apply(lambda row: len(row.geometry.coords), axis=1)
        
        # fmm does not identify matched nodes, but only matched edges. So we will only export matched edges
        
        # results
        self.results = edges_df[edges_df.geom_type == 'Point'], edges_df[edges_df.geom_type == 'LineString']
        self.network = networkDigraph # This is often used in other utilities
        if not input_nodes.empty:
            self.input_data = pd.concat([input_nodes,input_edges]) # This is often used in other utilities
        else:
            self.input_data = input_edges # This is often used in other utilities
        
        shutil.rmtree('temp/') # Clean up files made
        
    def plot(self):
    
        fig, ax = ox.plot_graph(self.network,show=False, close=False)
        self.input_data.plot(ax=ax)
        self.results[1].plot(ax=ax, color="red")
    
    def evaluate(self, gt, match = 'geometry'):
        
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
    
