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

def JC(trajectory_data, nodes_data, edges_data, current_position, previous_edge, iter):

    end_node = nodes_data.loc[previous_edge.index[0][1]]
    # start_node = nodes_data.loc[previous_edge.index[0][0]]
    
    # angle distance between the heading of the current point and the heading of the previous point
    # it should lie between 0 to pi
    delta_h = adjust_angle(abs(trajectory_data['GPS Bearing'].iloc[iter] - trajectory_data['GPS Bearing'].iloc[iter - 1]))
    
    # distance traveled from the last position fix to the end nodes
    d_1 = end_node['geometry'].distance(current_position['geometry']).iloc[0]
    
    # distance traveled since the last position fix 
    t = trajectory_data['time'].iloc[iter] - trajectory_data['time'].iloc[iter - 1]
    d_2 = (trajectory_data['speed_mps'].iloc[iter])* t.seconds
    
    delta_d = d_1 - d_2
    
    speed = current_position['speed_mps'].iloc[0]
    
    return JCfunc.MML(delta_d, delta_h, speed)
