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
from FIS3 import *
from function_util import *
from datetime import datetime

def SMP2(curr_loc, curr_edge, prev_loc, last_matched, err_size, nodes_utm, edges_utm, gdf_utm, wt_matrix = np.identity(10), plot = False, score = False):
    
    # get start and end node 
    start_node, end_node = node_direction(curr_edge, nodes_utm, curr_loc)

    # get edges inside the error region 
    err_poly = err_polygon(curr_loc, err_size)

    intersects = gpd.sjoin(err_poly, edges_utm, op='intersects')
    contains = gpd.sjoin(err_poly, edges_utm, op='contains')

    if (len(intersects) + len(contains)) >0:
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

        candidate_link = edges_utm.loc[appended_edge]
        # store previous end link info for connectivity checking 
        candidate_link['prev_end_node'] = np.repeat(end_node.name, len(candidate_link))
        # put u and v into column for connectivity checking
        candidate_link_uv = candidate_link.reset_index()

        # calculate perpendicular distance 
        # initialize list that hold perpendicular distance between points and edges
        p_dist = []
        # initialize list that hold connectivity 
        conn = []
        # grab unique edges only 
        #unique_candidate_edge = candidate_link.loc[candidate_link['osmid'].drop_duplicates().index]

        # Extract candidate node from the candidate link 
        # merge index
        node = pd.concat([candidate_link_uv['u'], candidate_link_uv['v']])
        # drop duplicate
        node_list = node.drop_duplicates()
        candidate_node = nodes_utm.loc[node_list]
        #print(candidate_node)
        # # Build Network Graph from the candidate node and candidate edge  
        G = ox.graph_from_gdfs(candidate_node, candidate_link)

        origin_edge = curr_edge.iloc[0]

        # project all the candidate link 
        route_list = []
        for i in range(len(candidate_link)):
            target_edge  = candidate_link.iloc[i]
            point_proj = point_matching(curr_loc, target_edge)
            origin_point = (prev_loc['geometry'].iloc[0].x, prev_loc['geometry'].iloc[0].y)
            target_point = (point_proj['geometry'].iloc[0].x, point_proj['geometry'].iloc[0].y)
            route = shotest_path(target_edge, target_point, origin_edge, origin_point, G, nodes_utm)
            route_list.append(route)
            # calculate perpendicular distance between current point and connectivity
            p_dist.append(candidate_link['geometry'].iloc[i].distance(curr_loc['geometry']).iloc[0])
            conn.append(check_connect(candidate_link_uv.iloc[i]))

        route_gdf = pd.concat(route_list)

        # attach perpendicular distance to candidate link 
        candidate_link["perp_dist"] = p_dist

        # attach connectivity 
        candidate_link["connectivity"] = conn
        candidate_link['d_n'] = route_gdf['dist_path'].to_list()
        # print(candidate_link)

        # calculate heading error
        # convert lat lon into tupple coordinate 
        candidate_link['lon_lat_pair'] = candidate_link.lon_lat.apply(lambda geom: list(geom.coords))

        # calculate bearing frome start and end node for each candidate link (see notes below)
        bearing_raw = candidate_link['lon_lat_pair'].apply(edge_bearing)

        # convert bearing from -pi, pi to 0, 2pi range
        candidate_link['edge_heading'] = bearing_raw.apply(conv_angle)

        # heading error = abs(gps heading - edge bearing)
        candidate_link['heading_error'] = abs(candidate_link['edge_heading'] - curr_loc['GPS Bearing'].iloc[0])


        # initialize input for FIS
        PD = candidate_link['perp_dist']
        HE = candidate_link['heading_error']
        speed = np.repeat(curr_loc['speed_mps'], len(candidate_link))
        hdop = np.repeat(curr_loc['GPS HDOP'], len(candidate_link))

        # distance travelled since last position fix 
        t = curr_loc['time'].iloc[0] - prev_loc['time'].iloc[0]
        d = np.repeat((curr_loc['speed_mps'].iloc[0])* t.seconds, len(candidate_link))

        # distance error d - dn 
        dist_err = abs(d - candidate_link['d_n'])
        # convert into list
        dist_err = dist_err.to_list()

        # rearrange new data to the input of fis3  
        new_data = np.array([speed, HE, PD, hdop, conn, dist_err]).T

        # perform FIS3 
        pred =[]
        for i in range(len(new_data)):
            pred.append(FIS3(new_data[i,:], method = 1, wt_matrix = wt_matrix, plot = False))

        # pick candidate link based on highest FIS value
        index = pred.index(max(pred))

        # selected edge and its point matched 
        next_edge = candidate_link.iloc[[index]]
        
        next_point_matching = point_matching(curr_loc, next_edge.iloc[0])
        
        # plot SMP 
        if plot == True:
            #%matplotlib tk
            # This is how we  visualize edges and error bound 

            # find the last two position for IMP
            poly_1 = err_polygon(curr_loc, err_size)

            # plotting edges and starting point together 
            f, ax = plt.subplots()

            # location for all point
            #locs_utm.plot(ax=ax)
            point_locs = gdf_utm['geometry'].to_frame()
            point_locs.iloc[12:15].plot(ax = ax)

            #err coord 
            # better if we just take location at the last and use error bound function 
            poly_1.plot(ax=ax, facecolor="none")

            # this plot all the road system 
            #edges_utm.plot(ax=ax)
            candidate_link.iloc[[0]].plot(ax = ax, color = 'Red')
            candidate_link.iloc[[3]].plot(ax = ax, color = 'Yellow')
            candidate_link.iloc[4:6].plot(ax = ax, color = 'Blue')
            candidate_link.iloc[6:8].plot(ax = ax)
            # this plot the selected edge at time point 

            # matched point plot
            last_matched.plot(ax = ax, color = "Grey")
            next_point_matching.plot(ax = ax, color = "Green")

            # matched_edge 
            next_edge.plot(ax = ax, color = "Black")

            # # plot closest node
            # closest_node.plot(ax = ax, color = "Black")
            # closest_edge.plot(ax = ax , color = "Black")
            # # debuging for djiksta shortest path
            # origin_node.plot(ax = ax, color = "Black")
            # target_node.plot(ax = ax , color = "Black")
        
        if score == True:
            return next_edge, max(pred)
        else:
            return next_edge