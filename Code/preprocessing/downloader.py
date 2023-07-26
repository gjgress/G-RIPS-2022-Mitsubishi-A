import requests
import urllib.parse
import json
import os

import geopandas as gpd

import preprocessing.constants

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
    utm_gdf = gdf.to_crs(crs=utm_crs)
    utm_gdf['lat_lon']  = gdf['geometry']
    return utm_gdf

def download_from_envirocar(id):
    """
    Download trajectory data from envirocar.org.

    Args:
        id: a id of dataset on envirocar.org

    Returns:
        GeoDataFrame: trajectory data
    """
    
    cache_dir = "cache/envirocar"
    cache_path = os.path.join(cache_dir, f"envirocar-{id}.geojson")
    if os.path.isfile(cache_path):
        gdf = gpd.read_file(cache_path)
        if len(gdf) <= 10:
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
    # Save to the file. It may be better to specify parameters when saving. The official doc is here: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_file.html
    gdf.to_file(cache_path)
    if len(gdf) <= 10:
        return None
    return to_utm_gdf(gdf)

def all_ids_on_envirocar():
    # Currently only downloads 100 tracks
    url = 'https://envirocar.org/api/stable/tracks/'
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