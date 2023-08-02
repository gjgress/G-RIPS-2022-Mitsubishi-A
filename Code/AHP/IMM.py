# load dependencies'
import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Point
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
import IMMfunction
from Map_Environment import ME

# some functions
def get_bearing(point1, point2):
    # this code calculates the bearing of any given pair of longitude, latitude  
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth,back_azimuth,distance = geodesic.inv(point1[0], point1[1], point2[0], point2[1])
    return fwd_azimuth

def edge_bearing(edge):
    # this function calculates the bearing from the starting and ending node of each road segment
    bearing = get_bearing(edge[0], edge[len(edge) - 1])
    return bearing
    
def conv_angle(angle):
    # this function converts the angle from [-pi, pi] to [0, 2pi]
    if angle < 0 :
        angle = angle + 360
    return angle

def adjust_angle(angle):
    # this function converts the angle so that if the angle is greater than pi it replaced with 2pi-angle
    if angle > 180:
        angle = 360 - angle
    return angle

def conc(a):
    #function to convert list or integer in osmid into a unique string id 
    if type(a) is int:
        return str(a)
    ans = ",".join(map(str, a))
    return ans

def err_polygon(curr_loc, err_size):
    crs_utm = curr_loc.crs
    # function that output shapely polygon for point error bound
    x = curr_loc['geometry'].iloc[0].x
    y = curr_loc['geometry'].iloc[0].y
    
    err_coord = [[x - err_size, y + err_size], 
                 [x + err_size, y + err_size],
                 [x + err_size, y - err_size],
                 [x - err_size, y - err_size]]

    poly_coord = Polygon(err_coord)
    # #print(ply_coord)
    df = {'Attribute' : ['name1'], 'geometry':poly_coord}

    #projected to UTM 31 
    err_poly = gpd.GeoDataFrame(df, geometry = 'geometry', crs = crs_utm)
    
    return err_poly

def IMM(trajectory_data, edges_data, iter):
    # initialization for IMM
    stop_iter = False
    err_size = 38
        
    while stop_iter == False :
        # extract current location at given iteration 
        curr_loc = trajectory_data.iloc[[iter]]

    
        # if the vehicle speed is less than 3m/s we skip it because the data at the time is less reliable
        if curr_loc['speed_mps'].iloc[0] < 3:
            # print(['the vehicle speed is less than 3m/s at iteration number', iter + 1])
            iter = iter + 1
        else:   
            # input should be location and error size 
            # create rectangular polygon 
            err_poly = err_polygon(curr_loc, err_size)
        
            # to plot error polygon for debugging
            # err_poly.plot()
            
            # Check for intersection and containment using geopandas
            intersects = gpd.sjoin(err_poly, edges_data, predicate='intersects')
        
            if (len(intersects)) <= 0:
                # print(['no edeges intersects with error bound at iteration number', iter + 1])
                iter = iter + 1
            else:    
                stop_iter = True
                # perform IMP only when there is edge intersects with error bound
                # print(['edges found at iteration number', iter + 1])
        
                # extract index from edges that intersect with error polygon 
                index = intersects[['index_right0', 'index_right1', 'index_right2']]
        
        
                # initialize candidate edges 
                appended_edge = []
        
                # extract candidate eges  
                for i in range(len(index)):
                    edge_list = (index['index_right0'].iloc[i], index['index_right1'].iloc[i], 0 )
                    appended_edge.append(edge_list)
        
                candidate_link = edges_data.loc[appended_edge]
        
                # calculate perpendicular distance 
                # initialize list that hold perpendicular distance between points and edges
                p_dist = []
        
                # calculate perpendicular distance between current point and candidate edges
                for i in range(len(candidate_link)):
                    p_dist.append(candidate_link['geometry'].iloc[i].distance(curr_loc['geometry']).iloc[0])
        
                # attach perpendicular distance to candidate link 
                candidate_link["perp_dist"] = p_dist
        
                # calculate heading error
                # convert lat lon into tuple coordinate 
                candidate_link['lon_lat_pair'] = candidate_link.lon_lat.apply(lambda geom: list(geom.coords))
        
                # calculate bearing from start and end node for each candidate link (see notes below)
                bearing_raw = candidate_link['lon_lat_pair'].apply(edge_bearing)
        
                # convert bearing from -pi, pi to 0, 2pi range
                candidate_link['edge_heading'] = bearing_raw.apply(conv_angle)
        
                # heading difference = abs(gps heading - edge bearing)
                heading_diff = abs(candidate_link['edge_heading'] - trajectory_data['GPS Bearing'].iloc[iter])
                
                # convert heading difference so that all its values lie from 0 to pi because the contribution of angle x and 2pi-x should be equal.
                candidate_link['heading_error'] = heading_diff.apply(adjust_angle)
        
        
                # input for IMM
                PD = candidate_link['perp_dist'].to_list()
                HE = candidate_link['heading_error'].to_list()
                map_enviroment = ME(edges_data, curr_loc)
                
                # find the index corresponding to the highest weight edge
                imm_res = IMMfunction.IMMfunc(PD, HE, map_enviroment)               

                a = candidate_link.iloc[[imm_res]]

                iter = iter + 1

                return iter, a
                
    
        































