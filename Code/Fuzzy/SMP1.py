# load dependencies'
import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import osmnx as ox
import networkx as nx
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from urllib.parse import urljoin
from shapely.geometry import Point, LineString, Polygon
import pyproj 
import mm_utils
from function_util import *
from FIS2 import FIS2
from datetime import datetime

def SMP1(curr_loc, curr_edge, prev_loc, last_matched, nodes_utm, edges_utm, gdf_utm):
    
    start_node, end_node = node_direction(curr_edge, nodes_utm, curr_loc)

    # finding angle alpha, which is the angle between previous matched point and current postion 
    # find pos_bearing1, bearing between last matched point and current point
    # convert two points into tuple for function input purposes
    last_matched_tuple = (last_matched['lon_lat'].x.iloc[0], last_matched['lon_lat'].y.iloc[0])
    cur_loc_tuple = curr_loc['lon_lat'].iloc[0].x, curr_loc['lon_lat'].iloc[0].y
    pos_bearing1 = get_bearing(last_matched_tuple, cur_loc_tuple)
    # convert to 0, 2pi 
    pos_bearing1 = conv_angle(pos_bearing1)

    # find bearing of the edge 
    # convert lat lon into tupple coordinate 
    start_node_tuple = (start_node.lon, start_node.lat)
    end_node_tuple = (end_node.lon, end_node.lat)
    # bearing between start to end
    start_bearing = get_bearing(start_node_tuple, end_node_tuple)
    # convert to 0 ,2*pi
    start_bearing = conv_angle(start_bearing)
    alpha = abs(pos_bearing1 - start_bearing)

    # find beta, angle between end node and current position 
    # bearing from end to start 
    end_bearing = get_bearing(end_node_tuple, start_node_tuple)
    end_bearing = conv_angle(end_bearing)
    #pos_bearing2 beating between end node and current position 
    pos_bearing2 = get_bearing(end_node_tuple, cur_loc_tuple)
    pos_bearing2 = conv_angle(pos_bearing2)
    # beta equal to abs(end_bearing minus pos_bearing2)
    beta = abs(end_bearing - pos_bearing2)


    # Heading Increment  
    HI = curr_loc['GPS Bearing'].iloc[0] - prev_loc['GPS Bearing'].iloc[0]

    # distance travelled from last position fix to the end nodes
    d = end_node['geometry'].distance(curr_loc['geometry']).iloc[0]

    # distance travelled since last position fix 
    t = curr_loc['time'].iloc[0] - prev_loc['time'].iloc[0]
    d2 = (curr_loc['speed_mps'].iloc[0] )* t.seconds

    delta_d = d - d2

    speed = curr_loc['speed_mps'].iloc[0] 
    hdop = curr_loc['GPS HDOP'].iloc[0] 

    # rearrange new data to the input of fis1  
    new_data = np.array([speed,hdop, alpha, beta, delta_d, HI, HI]).T

    res = FIS2(new_data, plot = False)
    
    return res