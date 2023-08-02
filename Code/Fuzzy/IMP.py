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
from function_util import*
from FIS1 import FIS1

def IMP(point_index, gdf_utm, edges_utm, n, wt_matrix = np.identity(6),plot = False):
    # n i s the number of same points detected to edges needed to stop IMP
    # function return index of location noted as starting point. 
    # initialization for IMP
    count = 0 
    same_link = 0
    stop_iter = False
    iter = point_index
    err_size = 38

    # saving answer for debugging purposes 
    # edge_link saves all the candidate link name for each iteration 
    # final answer is stored in the edge_link variable 
    edge_link = []
    # store fixed points : points projection to edge
    fixed_points = []
    # fis_res saves the output of FIS algorithm at every iteration 
    fis_res = []
    # HE_iter saves the heading error values for each candidate edge at any given iteration 
    HE_iter = [] 
    #curr_pos
    curr_pos_list = []
    # save candidate link name each iteration  
    candidate_link_res = []

    while stop_iter == False :
        # extract current location at given iteration 
        curr_loc = gdf_utm.iloc[[iter]]
        # save the iteration current position as a list
        curr_pos_list.append(curr_loc)

        #-----------------------------------------------------------------------
        # input should be location and error size 
        # create rectangular polygon 
        err_poly = err_polygon(curr_loc, err_size)

        # to plot error polygon for debugging
        # err_poly.plot()

        #---------------------------------------------------------------------------

        # Check for intersection and containment using geopandas
        intersects = gpd.sjoin(err_poly, edges_utm, op='intersects')
        contains = gpd.sjoin(err_poly, edges_utm, op='contains')

        if (len(intersects) + len(contains)) <=0:
            print(['no edeges intersects with error bound at iteration number', iter + 1])
        else:    
            # perform IMP only when there is edge intersects with error bound
            print(['edges found at iteration number', iter + 1])

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

            #save candidate link name 
 

            # calculate perpendicular distance 
            # initialize list that hold perpendicular distance between points and edges
            p_dist = []

            # calculate perpendicular distance between current point and 
            for i in range(len(candidate_link)):
                p_dist.append(candidate_link['geometry'].iloc[i].distance(curr_loc['geometry']).iloc[0])

            # attach perpendicular distance to candidate link 
            candidate_link["perp_dist"] = p_dist

            # print(candidate_link)

            # calculate heading error
            # convert lat lon into tupple coordinate 
            candidate_link['lon_lat_pair'] = candidate_link.lon_lat.apply(lambda geom: list(geom.coords))

            # calculate bearing frome start and end node for each candidate link (see notes below)
            bearing_raw = candidate_link['lon_lat_pair'].apply(edge_bearing)

            # convert bearing from -pi, pi to 0, 2pi range
            candidate_link['edge_heading'] = bearing_raw.apply(conv_angle)

            # heading error = abs(gps heading - edge bearing)
            candidate_link['heading_error'] = abs(candidate_link['edge_heading'] - gdf_utm['GPS Bearing'].iloc[iter])

            # initialize input for FIS
            PD = candidate_link['perp_dist']
            HE = candidate_link['heading_error']
            speed = np.repeat(gdf_utm['speed_mps'][iter]/ 3.6, len(candidate_link))
            hdop = np.repeat(gdf_utm['GPS HDOP'][iter], len(candidate_link))

            # save HE value every iter 
            HE_iter.append(HE)
            # rearrange new data to the input of fis1  
            new_data = np.array([speed, HE, PD, hdop]).T

            # calculating FIS
            pred =[]
            for i in range(len(new_data)):
                pred.append(FIS1(new_data[i,:], method = 1,wt_matrix = wt_matrix, plot = False))

            # print(pred)
            # save fis result 
            fis_res.append(pred)

            # pick candidate link based on 
            index = pred.index(max(pred))
            
            edge_link.append(candidate_link.iloc[index].name)

            # check if the current position and previous position is in the same edge
            if count > 0:
                if edge_link[count] == edge_link[count - 1]:
                    same_link = same_link + 1
                else:
                    same_link = 0

            # check to stop the for loop if n points belong to the same edge
            if same_link == (n - 1):
                print(['Starting SMP phase at',iter])
                stop_iter = True
                # find projection point 
                matched_link = candidate_link.iloc[[index]]
                matched_point = point_matching(curr_loc, matched_link.iloc[0])
            else:
                count = count + 1

        #update iteration 
        iter = iter + 1

        
    if plot == True:
        #%matplotlib tk
        # This is how we  visualize edges and error bound 
        # find which edges is selected at time point
        # find index of the edge id

        # find the last two position for IMP
        poly_1 = err_polygon(curr_pos_list[iter - 2], err_size)
        poly_2 = err_polygon(curr_pos_list[iter - 3], err_size)


        # plotting edges and starting point together 
        f, ax = plt.subplots()

        # location for all point
        #locs_utm.plot(ax=ax)
        point_locs = gdf_utm['geometry'].to_frame()
        point_locs.iloc[0:(iter), :].plot(ax = ax)

        #err coord 
        # better if we just take location at the last and use error bound function 
        poly_1.plot(ax=ax, facecolor="none")
        poly_2.plot(ax=ax, facecolor="none")

        # this plot all the road system 
        edges_utm.plot(ax=ax)

        # this plot the selected edge at time point 
        matched_link.plot(ax=ax, color = "Red")

        # matched point plot
        matched_point.plot(ax = ax, color = "Green")

        
    return (iter - 1), matched_link 