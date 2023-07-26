import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

import preprocessing.constants

def plot_trajectory(
    trajectory, title=None, plot_trajectory_edges=False, plot_clusters=False, color_mapping=None
):
    """
    Plot trajectory points.
    Args:
        trajectory: a GeoDataFrame object which represents trajectory points.
        title: the title of the plot
    """
    # Get the bounding box
    bbox = trajectory.to_crs(EPSG4326).total_bounds

    # 'total_bounds' returns a tuple with (minx, miny, maxx, maxy) values
    minx, miny, maxx, maxy = bbox

    # Download a map by specifying the bounding box, and draw the graph
    try:
        G = ox.graph.graph_from_bbox(
            maxy, miny, maxx, minx, network_type="all_private"
        )  # The order is north, south, east, west [https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_bbox]
    except (ValueError, nx.NetworkXPointlessConcept) as e:
        # TODO: Often a "ValueError: Found no graph nodes within the requested polygon" error or
        #       "NetworkXPointlessConcept: Connectivity is undefined for the null graph" error happens.
        #       This should be fixed if possible.
        print(e)
        return
    fig, ax = ox.plot_graph(G, figsize=(16, 16), show=False, close=False) # Switch the colors for the roads and the background

    # Set the title of ax
    ax.set_title(title)
    
    geographic_gdf = trajectory.to_crs(EPSG4326)
    
    # print('trajectory[\'geometry\']\n', trajectory[trajectory['cluster'] == 1])
    # print('geographic_gdf[\'geometry\']\n', geographic_gdf[trajectory['cluster'] == 1])

    if plot_trajectory_edges:
        # Plot the trajectory edges
        tripdata_edges = mm_utils.point_to_traj(
            geographic_gdf, columns={"ele": "average", "timestamp": "first", "sat": "first"}
        )
        tripdata_edges.plot(ax=ax, linewidth=1.5)

    if plot_clusters:
        # Get unique cluster labels
        unique_labels = geographic_gdf["cluster"].unique()

        # Generate a random color for each unique cluster label
        if color_mapping is None:
            color_mapping = {
                label: "#" + "%06x" % random.randint(0, 0xFFFFFF) for label in unique_labels
            }
        
        # print(color_mapping)
        
        # Color the noises by orange
        if -1 in unique_labels:
            color_mapping[-1] = "red"

        for cluster_label in unique_labels:
            cluster_points = geographic_gdf[geographic_gdf["cluster"] == cluster_label]
            color = color_mapping[cluster_label]
            cluster_points.plot(
                ax=ax, color=color, label=f"Cluster {cluster_label}"
            )
    else:
        # Plot the trajectory points
        geographic_gdf.plot(ax=ax, color="red")

    # Draw the whole plot in the campus!
    campus = ox.features.features_from_place("Somewhere", tags={"name": True})
    campus.plot(ax=ax, alpha=0.5)
    plt.legend()
    plt.show()