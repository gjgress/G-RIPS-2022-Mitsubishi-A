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
    def run(self, input_nodes, input_edges, network_nodes, network_edges):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a GeoDataFrame (and so must have a 'geometry' column)
            network is a GeoDataFrame object created from a MultiDiGraph (the default format for OSM)
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
        
        def extract_cpath(cpath):
            if (cpath==''):
                return []
            return [int(s) for s in cpath.split(',')]
        
        # Extract cpath as list of int
        resultdf["cpath_list"] = resultdf.apply(lambda row: extract_cpath(row.cpath), axis=1)

        # Extract all the edges matched
        all_edge_ids = np.unique(np.hstack(resultdf.cpath_list)).tolist()

        # Import our network edges
        network_gdf = gpd.read_file("temp/network_edges.shp")
        network_gdf.fid = network_gdf.fid.astype(int)

        edges_df = network_gdf[network_gdf.fid.isin(all_edge_ids)].reset_index()
        edges_df["points"] = edges_df.apply(lambda row: len(row.geometry.coords), axis=1)
        
        # fmm does not identify matched nodes, but only matched edges. So we will "export" both, but the nodes df will be empty
        
        # results
        self.results = edges_df[edges_df.geom_type == 'Point'], edges_df[edges_df.geom_type == 'LineString']
        self.network = networkDigraph
        self.input_data = input_nodes.append(input_edges)
        
        shutil.rmtree('temp/')
        
    def plot(self):
    
        fig, ax = ox.plot_graph(self.network,show=False, close=False)
        self.input_data.plot(ax=ax)
        self.results[1].plot(ax=ax, color="red")
    
    def evaluate(self, gt):
        return

    ###Required
    def results(self):
        '''
        Returns:
            algorithm results as specified in self.output
                algorithm time step, sec
                Euler angles [yaw pitch roll], rad
        '''
        return self.results
    
