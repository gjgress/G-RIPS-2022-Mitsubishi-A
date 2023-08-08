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


# this function draws the circle of radius 200m centered at the current location.
def circle_area(current_position):
    crs_utm = current_position.crs
    
    x = current_position['geometry'].iloc[0].x
    y = current_position['geometry'].iloc[0].y

    p1 = Point((x,y))
    df = pd.DataFrame({'Attribute' : ['name1']})
    circle_ragion = gpd.GeoDataFrame(df, geometry = [p1], crs = crs_utm)
    circle_ragion['geometry'] = circle_ragion.geometry.buffer(200)

    return circle_ragion


    
# this function outputs the map environment around the current position
# 0: urban, 1: suburban, 2: rural
def ME(edges_data, current_position):

    circle_ragion = circle_area(current_position)

    intersects = gpd.sjoin(circle_ragion, edges_data, predicate='intersects')
    contains = gpd.sjoin(circle_ragion, edges_data, predicate='contains')
    
    inter_index = intersects[['index_right0', 'index_right1', 'index_right2']].values.tolist()
    intersects_edges = edges_data.loc[map(tuple, inter_index)]

    # count the number of junctions; sorry this code is very messy because we rewrote them so that we do not use "osmid".
    temp = []
    for i in intersects_edges.index.tolist():   
        j = list(i)
        del j[2]
        temp.append(set(j))

    result = []
    for item in temp:
        if item not in result:
            result.append(item)
    
    ent = []
    for i in result:
        j = list(i)
        if len(j) >= 2:
            ent.append(j[0])
            ent.append(j[1])
        elif len(j) == 1:
            ent.append(j[0])
    
    temp2 = np.unique(ent, return_counts=True)
    number_junctions = np.count_nonzero(temp2[1] >= 3)

    # calculate the total length; 
    # but this does not calculate the exact total length of the roads within the area. 
    # just take the average of the total length of the roads in intersections and the total length of the roads in contains.
    if len(contains) == 0:
        return 2
    
    total_length_intersects = 0
    for i in range(len(intersects)):
        if intersects.iloc[i]['oneway'] == False:
            total_length_intersects += intersects.iloc[i]['length']/2
        else:
            total_length_intersects += intersects.iloc[i]['length']

    total_length_contains = 0
    for i in range(len(contains)):
        if intersects.iloc[i]['oneway'] == False:
            total_length_contains += contains.iloc[i]['length']/2
        else:
            total_length_contains += contains.iloc[i]['length']
    
    # km to m
    total_length = (total_length_contains + total_length_intersects)/ 2000
    
    result = number_junctions / total_length


    if result < 2.88:
        return 2
    elif result > 6.81:
        return 0
    else:
        return 1




















    