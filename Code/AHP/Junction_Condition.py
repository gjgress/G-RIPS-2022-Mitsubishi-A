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
from datetime import datetime
import JCfunc

def adjust_angle(angle):
    # this function converts the angle so that if the angle is greater than pi it replaced with 2pi-angle
    if angle > 180:
        angle = 360 - angle
    return angle

def point_matching(curr_loc, curr_edge):
    # matched position to the current edge 
    # input need to be panda series
    # curr loc need attribute geometry point
    # curr_edge need attribute 'geometry' lines
    # output a point that matched to the current edge
    crs_utm = curr_loc.crs
    dist = curr_edge['geometry'].project(curr_loc['geometry']).iloc[0]
    matched_point = list(curr_edge['geometry'].interpolate(dist).coords)
    matched_point = gpd.GeoDataFrame(geometry=gpd.points_from_xy([matched_point[0][0]], [matched_point[0][1]]), crs= crs_utm)
    return matched_point

def JC(nodes_data, edges_data, current_position, previous_position, previous_edge):

    end_node = nodes_data.loc[previous_edge.index[0][1]]
    start_node = nodes_data.loc[previous_edge.index[0][0]]

    
    # angle distance between the heading of the current point and the heading of the previous point
    # it should lie between 0 to pi
    HI = abs(current_position['GPS Bearing'].iloc[0] - previous_position['GPS Bearing'].iloc[0])
    delta_h = adjust_angle(HI)
    
    # distance traveled from the last position fix to the end nodes
    projected = point_matching(previous_position, previous_edge.iloc[0])

    d_1 = end_node['geometry'].distance(projected['geometry']).iloc[0]

    # distance traveled since the last position fix 
    t = current_position['time'].iloc[0] - previous_position['time'].iloc[0]
    d_2 = (current_position['speed_mps'].iloc[0] )* t.seconds
    
    delta_d = d_1 - d_2
    
    speed = current_position['speed_mps'].iloc[0]
    
    if JCfunc.MML(delta_d, delta_h, speed):
        # print(['d_1 ',d_1, ' ,d_2 ',d_2, ' delta_d ',delta_d, ' ,delta_h ', delta_h])
        return True
    else:
        return False

