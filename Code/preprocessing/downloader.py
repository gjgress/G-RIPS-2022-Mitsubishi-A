import requests
import urllib.parse
import json
import os

import geopandas as gpd
import gpxpy
import gpxpy.gpx
from shapely.geometry import Point, shape
import numpy as np
import osmnx as ox

from preprocessing.constants import EPSG4326

def read_file_content(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

def GPX_to_GeoDataFrame(gpx):
    """
    Convert GPX into GeoDataFrame

    Args:
        gpx: GPX data

    Returns:
        GeoDataFrame: A GeoDataFrame of the GPX data
    """

    data = {"geometry": [], "time": [], "elevation": []}
    for segment in gpx.tracks[0].segments:
        for p in segment.points:
            data["geometry"].append(Point(p.longitude, p.latitude))
            data["time"].append(p.time)
            data["elevation"].append(p.elevation)
    return gpd.GeoDataFrame(data=data, geometry="geometry", crs=EPSG4326) # EPSG:4326 is the commonly used CRS for GPS [https://epsg.io/4326]

def to_utm_gdf(gdf):
    utm_crs = gdf.estimate_utm_crs()
    # print('utm_crs: ', utm_crs)
    # print('gdf: ', gdf)
    utm_gdf = gdf.to_crs(crs=utm_crs)
    # print('utm_gdf: ', utm_gdf)
    utm_gdf['lon_lat']  = gdf['geometry']
    return utm_gdf

def download_from_envirocar(id, use_cache=True, cache_dir='cache/envirocar', extension='geojson', threhold=10):
    """
    Download trajectory data from envirocar.org.

    Args:
        id: a id of dataset on envirocar.org

    Returns:
        GeoDataFrame: trajectory data
    """
    
    def make_cache_path():
        return os.path.join(cache_dir, f"envirocar-{id}.{extension}")
    
    if use_cache:
        cache_path = make_cache_path()
        if os.path.isfile(cache_path):
            gdf = gpd.read_file(cache_path)
            if len(gdf) <= threhold:
                return None
            return to_utm_gdf(gdf)
    
    # Download trajectory data from envirocar.org.
    url = urllib.parse.urljoin('https://envirocar.org/api/stable/tracks/', id)
    response = requests.get(url)
    data = json.loads(response.text)

    # Convert the data into a GeoDataFrame
    geometries = []
    attributes = []
    for feature in data["features"]:
        geometry = shape(feature["geometry"])
        geometries.append(geometry)
        attributes.append(feature["properties"])
    
    gdf = gpd.GeoDataFrame(data=attributes, geometry=geometries)
    gdf = gdf.set_crs(EPSG4326)
    
    if use_cache:
        cache_path = make_cache_path()
        gdf.to_file(cache_path) # Save to the file. It may be better to specify parameters when saving. The official doc is here: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html
        
    if threhold is not None and len(gdf) <= threhold:
        return None
    
    return to_utm_gdf(gdf)

import pandas as pd

def save_trajecotory_from_envirocar_as_npz(id, dir, compressed):
    utm_gdf = download_from_envirocar(id, use_cache=False)
    utm_gdf['time'] = pd.to_datetime(utm_gdf['time'])
    start_time = utm_gdf['time'][0]
    
    x = []
    y = []
    t = []
    speed = []
    direction = []
    HDOP = []
    default_HDOP = 100 # Note: higer values have lower precision
    print(id)
    for i in range(len(utm_gdf)):
        if 'Speed' not in utm_gdf['phenomenons'][i]:
            # print(f'Skipped: {id} (a trajectory point does not have the speed field)')
            return
        x.append(utm_gdf['geometry'][i].x)
        y.append(utm_gdf['geometry'][i].y)
        t.append((pd.to_datetime(utm_gdf['time'][i]) - start_time).seconds)
        speed.append(utm_gdf['phenomenons'][i]['Speed']['value'])
        HDOP.append(utm_gdf['phenomenons'][i]['GPS HDOP']['value'] if 'GPS HDOP' in utm_gdf['phenomenons'][i].keys() else default_HDOP)
        
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    speed = np.array(speed)
    direction = np.array(direction)
    HDOP = np.array(HDOP)
    file_name = f'envirocar-{id}.npz'
    file_path = os.path.join(dir, file_name)
    if compressed:
        np.savez_compressed(file_path, x, y, t, speed, direction, HDOP)
    else:
        np.savez(file_path, x, y, t, speed, direction, HDOP)
    # print(f'Saved: {file_path}')
    
    # # Test whether the loaded information is the same as the saved information
    # npzfile = np.load(file_path)
    # print(npzfile)
    # lx, ly, lt, ls, ld, lh = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3'], npzfile['arr_4'], npzfile['arr_5']
    # assert np.array_equal(x, lx)
    # assert np.array_equal(y, ly)
    # assert np.array_equal(t, lt)
    # assert np.array_equal(speed, ls)
    # assert np.array_equal(direction, ld)
    # assert np.array_equal(HDOP, lh)

def load_npz(file_path):
    npzfile = np.load(file_path)
    x, y, t, speed, direction, HDOP = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3'], npzfile['arr_4'], npzfile['arr_5']
    return x, y, t, speed, direction, HDOP
    
def save_trajectories_from_envirocar_as_npz(num_trajectories, dir, compressed):
    """
    Download GPS trajectories from envirocar and save them as npz files.
    
    Args:
        num_trajectories: # of downloaded trajectories
        dir: where to save the npz files
        compressed: if save the GPS trajectories as compressed npz files
    
    Notes:
        It may happen that some GPS trajectories are unavailable or lack the speed field in some trajectories points.
        In these cases, this functions just skips those GPS trajectories.
        Thus, the number of downloaded GPS trajectories may less than the num_trajectories parameter.
    """
    ids = fetch_ids_on_envirocar(num_trajectories)
    for id in ids:
        save_trajecotory_from_envirocar_as_npz(id, dir, compressed)

def fetch_ids_on_envirocar(num_trajectories):
    """
    Fetch GPS trajectory ids from envirocar.
    
    Args:
        num_trajectories: # of ids fetched from envirocar
    
    Returns:
        list[str]: the list of the fetched ids
    """
    url = f'https://envirocar.org/api/stable/tracks?limit={num_trajectories}'
    response = requests.get(url)
    data = json.loads(response.text)
    ids = []
    for track in data['tracks']:
        ids.append(track['id'])
    return ids

def load_GPStrace_from_OSM(id, save_local=True):
    """
    Download a GPS trace from Open Street Map, or load it from the local storage in case there exists the cache.

    Args:
        id: the id of the GPS trace data. For example, "8504293" in https://www.openstreetmap.org/user/sunnypilot/traces/8504293
        save_local: if True, then the GPS trace data is save as GPX into the local storage when the request was successful

    Returns:
        Optional[GPX]: the GPX data of the GPS trace (Note: None is returned when the request failed)
    """

    cache_dir = "cache/osm-gps-traces"
    cache_path = os.path.join(cache_dir, f"osm-gps-trace-{id}")

    if os.path.isfile(cache_path):
        gdf = GPX_to_GeoDataFrame(gpxpy.parse((read_file_content(cache_path)))).set_crs(EPSG4326)
        return to_utm_gdf(gdf)

    base_url = "https://www.openstreetmap.org/trace/"
    url = urllib.parse.urljoin(base_url, f"{id}/data")
    response = requests.get(url)

    if response.status_code == 404:
        return None

    with open(cache_path, "w") as file:
        file.write(response.text)

    gdf = GPX_to_GeoDataFrame(gpxpy.parse(response.text)).set_crs(EPSG4326)
    return to_utm_gdf(GPX_to_GeoDataFrame(gpxpy.parse(response.text)).to_crs(EPSG4326))

def to_fuzzy_AHP_input(gdf):
    """
    Create the input for the fuzzy logic MM and AHP MM from the given GeoDataFrame
    
    Args:
        gdf: GeoDataFrame
    
    Returns:
        tuple[GeoDataFrame, GeoDataFrame, GeoDataFrame]: (gdf, nodes, edges)
    """
    def conc(a):
        #function to convert list or integer in osmid into a unique string id 
        if type(a) is int:
            return str(a)
        ans = ",".join(map(str, a))
        return ans

    #### phenomenons contains trajectory information of each time point in dictionary format
    # create array that save trajectory information that we want to extract 
    key_list = ['GPS Speed', 'GPS HDOP', 'GPS Bearing']

    # Initialize data frame 
    df = pd.DataFrame()

    # loop to get trajectory infos for each point 
    for key in key_list:
        temp = []
        for i in range(gdf.shape[0]):
            dict_temp = gdf['phenomenons'][i]
            if key in dict_temp:
                temp.append(dict_temp[key]['value'])
            else:
                temp.append(float("nan"))
        df[key] = temp

    # combine geodata frame with the new extracted info    
    gdf = pd.concat([gdf, df],axis = 1)

    
    #### get road network 
    # Get the bounding box
    # bbox = gdf.total_bounds
    bbox = gdf.to_crs(EPSG4326).total_bounds

    # 'total_bounds' returns a tuple with (minx, miny, maxx, maxy) values
    minx, miny, maxx, maxy = bbox

    # Download a map by specifying the bounding box
    print("Downloading the road network graph")
    G = ox.graph.graph_from_bbox(maxy, miny, maxx, minx, network_type='all_private', retain_all=True, truncate_by_edge=True) 
    print("Finished downloading")
    
    graph_proj = ox.project_graph(G)
    nodes_utm, edges_utm = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)

    # extract road info 
    nodes, edges = ox.graph_to_gdfs(G)

    # append latitude and longitude to utm edges 
    edges_utm['lon_lat'] = edges['geometry']

    # convert osmid into unique string id 
    edges_utm['str_id'] = edges_utm['osmid'].apply(conc)

    # append latitude and longitude to utm edges 
    nodes_utm['lon_lat'] = nodes['geometry']


    #### convert gdf to utm projection
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(EPSG4326)
        gdf = dl.to_utm_gdf(gdf)
    else:
        crs = gdf.crs
        is_utm = crs.is_projected and crs.axis_info[0].name == 'Easting' and crs.axis_info[1].name == 'Northing'
        if not is_utm:
            gdf = dl.to_utm_gdf(gdf)
    
    # gdf['lon_lat'] = [Point(p.y, p.x) for p in gdf['lat_lon']]
    # gdf = gdf.drop('lat_lon', axis=1)

    # convert time str to datetime
    gdf['time'] = pd.to_datetime(gdf['time'])
    
    gdf['speed_mps'] = gdf['GPS Speed']/3.6
    
    return gdf, nodes_utm, edges_utm