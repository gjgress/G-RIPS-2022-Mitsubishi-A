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
import MMJfunction
from Map_Environment import ME

def get_bearing(point1, point2):
    # this code calculate the bearing of any given pair of longitude, latitude  
    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth,back_azimuth,distance = geodesic.inv(point1[0], point1[1], point2[0], point2[1])
    return fwd_azimuth

def edge_bearing(edge):
    # this function calculate the bearing from the starting and ending node of each road segment
    bearing = get_bearing(edge[0], edge[len(edge) - 1])
    return(bearing)
    
def conv_angle(angle):
    # this function convert angle from -pi,pi to 0,2*pi
    if angle < 0 :
        angle = angle + 360
    return(angle)

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


def check_connect(link):
    # this function check connectivity of candidate link with the current edge
    # needed to convert key to column and the previous end node saved in the dataframe
    # 1 if its connect 0 if elsewhere
    if (link['u'] == link['prev_end_node']):
        a = 1
    else : 
        a = 0
    return a

def MMJ(trajectory_data, nodes_data, edges_data, current_position, previous_edge, iter):
    err_size = 38
    # extract node id
    edge_node1 = nodes_data.loc[previous_edge.index[0][0]]
    edge_node2 = nodes_data.loc[previous_edge.index[0][1]]
    
    edge_link = []
    
    # select the end node 
    if (current_position['GPS Bearing'].iloc[0] > 0 and current_position['GPS Bearing'].iloc[0] < 180):
        # if object movint to the right of map select the one with larger longitude
        if edge_node1.x > edge_node2.x:
            end_node = edge_node1
            start_node = edge_node2
        else:
            end_node = edge_node2
            start_node = edge_node1
    else:
        # if object is moving to the left select node with smaller longitude
        if edge_node1.x < edge_node2.x:
            end_node = edge_node1
            start_node = edge_node2
        else:
            end_node = edge_node2
            start_node = edge_node1
    
    # get edges inside the error region 
    err_poly = err_polygon(current_position, err_size)
    
    intersects = gpd.sjoin(err_poly, edges_data, predicate='intersects')
    contains = gpd.sjoin(err_poly, edges_data, predicate='contains')
    # print('len(intersects))
    if (len(intersects) + len(contains)) > 0:
        # extract index from edges that intersect with error polygon 
        int_index = intersects[['index_right0', 'index_right1', 'index_right2']]
        # extract index from edges that contained in the error polygon 
        cont_index = contains[['index_right0', 'index_right1', 'index_right2']]
    
        # merge index
        index = pd.concat([int_index, cont_index])
        # drop duplicate
        index = index.drop_duplicates()
    
        # initialize candidate edges 
        appended_edge = []
    
        # extract candidate eges  
        for i in range(len(index)):
            edge_list = (index['index_right0'].iloc[i], index['index_right1'].iloc[i], 0 )
            appended_edge.append(edge_list)
    
        candidate_link = edges_data.loc[appended_edge]
        # store previous end link info for connectivity checking 
        candidate_link['prev_end_node'] = np.repeat(end_node.name, len(candidate_link))
        # put u and v into column for connectivity checking
        candidate_link_uv = candidate_link.reset_index()
        
        # calculate perpendicular distance 
        # initialize list that hold perpendicular distance between points and edges
        p_dist = []
        # initialize list that hold connectivity 
        conn = []
        # calculate perpendicular distance between current point and connectivity
        for i in range(len(candidate_link)):
            p_dist.append(candidate_link['geometry'].iloc[i].distance(current_position['geometry']).iloc[0])
            conn.append(check_connect(candidate_link_uv.iloc[i]))
        
        # attach perpendicular distance to candidate link 
        candidate_link['perp_dist'] = p_dist
        
        # attach connectivity 
        candidate_link['connectivity'] = conn
    
        # print(candidate_link)
    
        # calculate heading error
        # convert lat lon into tupple coordinate 
        candidate_link['lon_lat_pair'] = candidate_link.lon_lat.apply(lambda geom: list(geom.coords))
    
        # calculate bearing frome start and end node for each candidate link (see notes below)
        bearing_raw = candidate_link['lon_lat_pair'].apply(edge_bearing)
    
        # convert bearing from -pi, pi to 0, 2pi range
        candidate_link['edge_heading'] = bearing_raw.apply(conv_angle)
    
        # heading difference = abs(gps heading - edge bearing)
        heading_diff = abs(candidate_link['edge_heading'] - current_position['GPS Bearing'].iloc[0])
                
        # convert heading difference so that all its values lie from 0 to pi because the contribution of angle x and 2pi-x should be equal.
        candidate_link['heading_error'] = heading_diff.apply(adjust_angle)    
        
        # initialize input for MMJ
        PD = candidate_link['perp_dist'].to_list()
        HE = candidate_link['heading_error'].to_list()
        TR = candidate_link['connectivity'].to_list()
        map_enviroment = ME(edges_data, current_position)
        
        mmj_res = MMJfunction.MMJfunc(PD, HE, TR, map_enviroment)
        edge_link.append(candidate_link['osmid'].iloc[mmj_res])

        # loc = np.where(edges_data["str_id"] == conc(edge_link[0]))
        # a = edges_data.iloc[loc]
        
        a = candidate_link.iloc[[mmj_res]]
        
        iter = iter + 1

        return iter, a
    else:
        return (iter + 1), previous_edge

















