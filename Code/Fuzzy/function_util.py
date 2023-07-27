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

# this python script contains all the functions needed for IMP, SMP and SMP2 steps 
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

def check_connect(link):
    # this function check connectivity of candidate link with the current edge
    # needed to convert key to column and the previous end node saved in the dataframe
    # 1 if its connect 0 if elsewhere
    if (link['u'] == link['prev_end_node']):
        a = 1
    else : 
        a = 0
    return a

def node_direction(edge, nodes_list, loc):
    # function that give the start and end nodes given the current location and edge both are geopanda dataframe
    # nodes_list are gpd dataframe that stores all the information of the nodes dat
    # extract node id
    edge_node1 = nodes_list.loc[edge.index[0][0]]
    edge_node2 = nodes_list.loc[edge.index[0][1]]

    # select the end node 
    if (loc['GPS Bearing'].iloc[0] > 0 and loc['GPS Bearing'].iloc[0] < 180):
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
    
    return start_node, end_node

def shotest_path(target_edge, target_point, origin_edge, origin_point, G, nodes_utm):
    # function to find shortes path between two point in a graph origin point and target point
    # input : 
    # target edge is the edge where the target point located
    # origin edge is the edge where the origin point located
    # G is graph network 
    # nodes_utm is the dataframe that stores all the projected nodes
    # output :
    # length_m is the length of path through the closest nodes
    # dist_path is the shortest path between origin and target point
    # not used for now 
    # route nodes is the list of nodes travelled through the shortest path 
    
    # get the closet nodes to the start and target point
    origin_node, origin_dist =  ox.distance.nearest_nodes(G, origin_point[0], origin_point[1], return_dist =True)
    target_node, target_dist =  ox.distance.nearest_nodes(G, target_point[0], target_point[1], return_dist = True)

    # Retrieve the rows from the nodes GeoDataFrame
    o_closest = nodes_utm.loc[origin_node]
    t_closest = nodes_utm.loc[target_node]

    #print(origin_edge['str_id'] == target_edge['str_id'])

    if (o_closest.name == t_closest.name):
        # start and end points shared same closest node
        route_geom = gpd.GeoDataFrame([[o_closest.geometry]],geometry='geometry', crs=nodes_utm.crs,columns=['geometry'])
        route_geom['length_m'] = 0 
        if origin_edge['str_id'] == target_edge['str_id']:
            # case 1 origin and target share same edge, then distance travelled is equal to difference between target and origin distance to node
            dist_path = abs(target_dist - origin_dist)
            #print(dist_path)
        else:
            # case 2 origin and target are adjacent edge, distance travelled is equal to sum of target and origin distance
            dist_path = target_dist + origin_dist
    else : 
        # when start and end points doesnt share same closest node, need to find shortest path 
        # Create a GeoDataFrame from the origin and target points
        od_nodes = gpd.GeoDataFrame([o_closest, t_closest],geometry='geometry', crs=nodes_utm.crs,)


        # Calculate the shortest path
        route = nx.shortest_path(G, source=origin_node, target=target_node, weight='length')


        # Get the nodes along the shortest path
        route_nodes = nodes_utm.loc[route]
        # convert route into linestring so we can calculate distance 
        route_line = LineString(list(route_nodes.geometry.values))

        # Create a GeoDataFrame
        route_geom = gpd.GeoDataFrame([[route_line]], geometry='geometry', crs=nodes_utm.crs, columns=['geometry'])

        # Calculate the route length
        route_geom['length_m'] = route_geom.length
        
        dist_path = route_geom['length_m']
        # fixing distance from origin to nearest nodes
        if origin_edge.name[0] == route[0]:
            if(origin_edge.name[1] == route[1]):
                # origin edge is inside the route, start edge is equal to start of route
                origin_start_node = origin_edge.name[0]
                origin_end_node = origin_edge.name[1]
                dist_path = dist_path - origin_dist
            else:
                # origin edge is outside the route, end edge is equal to start of route
                origin_end_node = origin_edge.name[0]
                origin_start_node = origin_edge.name[1]
                dist_path = dist_path + origin_dist
        elif origin_edge.name[1] == route[0]:
            if(origin_edge.name[0] == route[1]):
                # origin edge is inside the route, start edge is equal to start of route
                origin_start_node = origin_edge.name[1]
                origin_end_node = origin_edge.name[0]
                dist_path = dist_path - origin_dist
            else:
                # origin edge is outside the route, end edge is equal to start of route
                origin_end_node = origin_edge.name[1]
                origin_start_node = origin_edge.name[0]
                dist_path = dist_path + origin_dist

        # fixing distance from targert to nearest nodes 
        if target_edge.name[0] == route[-1]:
            if(target_edge.name[1] == route[-2]):
                # target edge is inside the route, end edge is equal to end of route
                target_start_node = target_edge.name[1]
                target_end_node = target_edge.name[0]
                dist_path = dist_path - target_dist
            else:
                #target edge is outside the route, end route is start of edge
                target_end_node = target_edge.name[1]
                target_start_node = target_edge.name[0]
                dist_path = dist_path + target_dist
        elif target_edge.name[1] == route[-1]:
            if(target_edge.name[0] == route[-2]):
                # target edge is inside the route, end edge is equal to end of route
                target_start_node = target_edge.name[0]
                target_end_node = target_edge.name[1]
                dist_path = dist_path - target_dist
            else:
                #target edge is outside the route, end route is start of edge
                target_end_node = target_edge.name[0]
                target_start_node = target_edge.name[1]
                dist_path = dist_path + target_dist


    route_geom['dist_path'] = dist_path
    # route_geom['target_start_node'] = target_start_node
    # route_geom['target_end_node'] = target_end_node
    # route_geom['origin_start_node'] = origin_start_node
    # route_geom['origin_end_node'] = origin_end_node
    # route_geom['route_nodes'] = [tuple(route)]


    return route_geom 
