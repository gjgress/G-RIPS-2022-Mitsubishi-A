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
    
    # unique_intersects = intersects_edges.loc[intersects_edges['osmid'].drop_duplicates().index]
    # temp = unique_intersects.reset_index()
    # list = pd.concat([temp['u'], temp['v']])
    # list = list.to_list()
    # temp2 = np.unique(list, return_counts=True) 

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
    # total_length = total_length_contains / 1000
    
    result = number_junctions / total_length

    # print(len(contains))
    # print(number_junctions)
    # print('diff',total_length_intersects-total_length_contains)
    # print('cont',total_length_contains)
    # print('int',total_length_intersects)
    # print('total',total_length)
    # print(result)
    # print('-----------------------------------------------------------------------------')
    if result < 2.88:
        return 2
    elif result > 6.81:
        return 0
    else:
        return 1




















    