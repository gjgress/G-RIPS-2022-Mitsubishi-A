# -*- coding: utf-8 -*-
# Fielname = fmm_bin.py

"""
(Generalized) Metric-based Map Matching Algorithm
Created on 2022-07-25
@author: gjgress

This is a modular algorithm designed to use arbitrary metric-based functions in conjunction with nearest neighbors. Candidate routes and GPS tracks are dissected into n and m nodes respectively; we then calculate the Euclidean distance to the (k-)nearest neighbor for both the candidate route and the GPS track. Then the user can apply a loss function of their choice onto this data. We provide two samples in this module: least squares, and inverse squares.

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
import sys
from functools import reduce
from algorithms import mm_utils
from shapely.geometry import Point
import dask

### Required    
class Sim(object):
    '''
    '''
    def __init__(self, route_inner_f = None, route_outer_f = None, gps_inner_f = None, gps_outer_f = None, wrapper_f = None, loss_function = None, trajectory = None, candidate_route_nodes = None, candidate_routes = None):
        '''
        Args:
            None
        '''
        # algorithm description
        self.input = ['geometry']
        self.output = ['geometry']
        self.batch = True
        self.results = None
#        self.config = cfg
        
        self.route_inner_f = route_inner_f
        self.route_outer_f = route_outer_f
        self.gps_inner_f = gps_inner_f
        self.gps_outer_f = gps_outer_f
        self.wrapper_f = wrapper_f
        self.trajectory = trajectory
        self.candidate_route_nodes = candidate_route_nodes
        self.candidate_routes = candidate_routes
        
        if loss_function is None:
            try:
                self.loss_function = wrapper_f(route_inner_f, route_outer_f, gps_inner_f, gps_outer_f)
            except:
                raise Exception('The functions provided could not be parsed into a proper loss function!')
        else:
            self.loss_function = loss_function
        
        # algorithm vars
        this_dir = os.path.dirname(__file__)

    ### Required
    def preprocessing(self, input_edges, network_edges=None, input_nodes=None, network_nodes=None, candidate_routes = None, n = 100, m = 100, *args):
        '''
        A necessary preprocessing step that prepares the input data for use in self.run()
        Args:
        
            input_edges: a GeoDataFrame (and so must have a 'geometry' column) consisting of LineStrings (GPS data)
            network edges: a GeoDataFrame (and so must have a 'geometry' column) consisting of LineStrings (network data)
            (this is not optional, but if its not given, we use OSMnx to get it)
            input_nodes: This is not used, but included for consistency
            network_nodes: This is not used, but included for consistency
            candidate_routes: A list of GeoDataFrames (of LineStrings) representing candidate routes. While you can let the algorithm determine it, there are many parameters which are difficult to determine in advance; therefore, we recommend you provide it directly.
            n: The number of interpolation points to obtain for each edge of the trajectory. Higher better approximates the integral, but may increase computation time (usually the computation time increase is negligible). If 0 is passed, it will return the nodes between the LineStrings
            m: The number of interpolation points to obtain for each edge of the candidate route. Higher better approximates the integral, but may increase computation time (usually the computation time increase is negligible). If 0 is passed, it will return the nodes between the LineStrings
            args*: If provided, passes the parameters into the candidate route constructor.
        '''
        
        gps_nodes = []
        
        for row in input_edges['geometry']:
            edge = []
            for i in range(n+1):
                edge.append(row.interpolate(i/n, normalized=True))
            edge = gpd.GeoDataFrame(edge, columns = ['geometry'], crs = 'EPSG:4326') # Maybe project crs?
            if len(gps_nodes) == 0:
                gps_nodes = edge
            else:
                gps_nodes = pd.concat([gps_nodes,edge])            
        
        self.trajectory = gps_nodes.drop_duplicates(subset=['geometry']).reset_index(drop=True)
        
        if network_edges is None:
            _, network_edges = mm_utils.df_to_network(input_edges)
        
        if candidate_routes is None:
            qry_pts = [y for sublist in [x.coords[:] for x in network_edges['geometry']] for y in sublist]            
            source_index, _ = mm_utils.get_nearest([(gps_nodes['geometry'].iloc[0].x, gps_nodes['geometry'].iloc[0].y)], qry_pts, k_neighbors = 1)
            source_index = source_index[0][0]
            target_index, _ = mm_utils.get_nearest([(gps_nodes['geometry'].iloc[-1].x, gps_nodes['geometry'].iloc[-1].y)], qry_pts, k_neighbors = 1)
            target_index = target_index[0][0]

            source = Point(qry_pts[source_index])
            target = Point(qry_pts[target_index])
                                
            candidates = mm_utils.get_nearest_edges(gps_nodes, network_edges)
            all_candidate_edges = reduce(lambda left,right: pd.concat([left, right]).drop_duplicates(subset=['geometry'])
, candidates)
            candidate_routes, _ = mm_utils.dijkstra(source, target, all_candidate_edges, *args)
        
        self.candidate_routes = candidate_routes
        
        candidate_route_nodes = []
        
        for route in candidate_routes:
            route_nodes = []
            for row in route['geometry']:
                edge = []
                for i in range(m+1):
                    edge.append(row.interpolate(i/m, normalized=True))
                edge = gpd.GeoDataFrame(edge, columns = ['geometry'], crs = 'EPSG:4326')#.to_crs('EPSG:3395') # We will need to be in a projected CRS to properly do length calculations; the choice of CRS does not greatly matter
                if len(route_nodes) == 0:
                    route_nodes = edge
                else:
                    route_nodes = pd.concat([route_nodes,edge]).drop_duplicates(subset=['geometry']).reset_index(drop=True)
            candidate_route_nodes.append(route_nodes)
        
        # Now we should have a list of GeoDataFrames of interpolated nodes!
        
        self.candidate_route_nodes = candidate_route_nodes
        
        # This is all we need to run our algorithm.
        
        
    def run(self, k1 = 1, k2 = 1, return_full = False, return_results=False, parallel = True, preprocessed = True, **kwargs):
        '''
        main procedure of the algorithm
        Args:
            
            k1: The number of Nearest Neighbors to use when obtaining the distance between a GPS node and the candidate route nodes; note that increasing this number does not necessarily increase accuracy-- it just means that more points of the polyline are taken into consideration. You may not wish to compare every node of one polyline with the entirety of the other! If -1 is given, computes max
            k2: The number of Nearest Neighbors to use when obtaining the distance between a candidate route node and the GPS nodes. If -1 is given, computes max
            
        By default it assumes minimum less is optimal. If you want to instead maximize loss, include a -1* term in your wrapper function (turning a max into a min)
        '''
        
        if not preprocessed:
            self.preprocessing(**kwargs)    
            
        loss = []
        
        if parallel:
            for i in range(len(self.candidate_route_nodes)):
                in_pts = [(x,y) for x,y in zip(self.candidate_route_nodes[i].to_crs("EPSG:3395").geometry.x , self.candidate_route_nodes[i].to_crs("EPSG:3395").geometry.y)]
                qry_pts =  [(x,y) for x,y in zip(self.trajectory.to_crs("EPSG:3395").geometry.x , self.trajectory.to_crs("EPSG:3395").geometry.y)]
                _, routeloss = mm_utils.get_nearest(in_pts, qry_pts, k_neighbors = k1)
                _, gpsloss = mm_utils.get_nearest(qry_pts, in_pts, k_neighbors = k2)
                loss.append(dask.delayed(self.loss_function)(routeloss, gpsloss))  
            loss = dask.compute(*loss)
            
        else:
            for i in range(len(self.candidate_route_nodes)):
                in_pts = [(x,y) for x,y in zip(self.candidate_route_nodes[i].to_crs("EPSG:3395").geometry.x , self.candidate_route_nodes[i].to_crs("EPSG:3395").geometry.y)]
                qry_pts =  [(x,y) for x,y in zip(self.trajectory.to_crs("EPSG:3395").geometry.x , self.trajectory.to_crs("EPSG:3395").geometry.y)]
                _, routeloss = mm_utils.get_nearest(in_pts, qry_pts, k_neighbors = k1)
                _, gpsloss = mm_utils.get_nearest(qry_pts, in_pts, k_neighbors = k2)
                loss.append(self.loss_function(routeloss, gpsloss))
        
        
        loss_ind = np.argsort(loss)
        ix = [i for i, j in enumerate(loss) if j == min(loss)][0]
        
        if return_full:
            self.results = [self.candidate_routes, loss]
        
        else:
            self.results = [self.candidate_routes[ix], loss[ix]]
    
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
# -*- coding: utf-8 -*-