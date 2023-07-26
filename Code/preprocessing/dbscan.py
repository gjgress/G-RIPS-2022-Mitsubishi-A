import math

import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
import geopandas as gpd

import preprocessing.plotter as plotter

def adjust_scale(x, r):
    return np.array(x) * r

def rotate_point(x_coords, y_coords, angle_degrees, origin=(0, 0)):
    """
    Rotate a point (or list of points) around the origin (or specified point) in a 2D plane.

    Parameters:
        x_coords (list or numpy.ndarray): List of x-coordinates of the points.
        y_coords (list or numpy.ndarray): List of y-coordinates of the points.
        angle_degrees (float): Angle of rotation in degrees.
        origin (tuple, optional): Origin point (default is (0, 0)).

    Returns:
        tuple or numpy.ndarray: Tuple or array of rotated (x, y) coordinates.
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # Shift the coordinates to the origin
    x_coords_shifted = x_coords - origin[0]
    y_coords_shifted = y_coords - origin[1]

    # Perform rotation
    x_rotated = cos_theta * x_coords_shifted - sin_theta * y_coords_shifted
    y_rotated = sin_theta * x_coords_shifted + cos_theta * y_coords_shifted

    # Shift the coordinates back to the original position
    x_rotated += origin[0]
    y_rotated += origin[1]

    return x_rotated, y_rotated

def to_points(trajectory):
    """
    Convert trajectory points into a vector (I have to check the type) of points

    Args:
        trajectory: a GeoDataFrame represents trajectory points

    Returns:
        pandas.core.series.Series: a series of points representing trajectory points
    """
    return trajectory["geometry"].apply(lambda point: [point.x, point.y])

def knn_distance(gdf, point_index, k):
    # Calculate kNN distance
    point = gdf.geometry.iloc[point_index]
    return gdf.geometry.distance(point).sort_values().iloc[k]

def knn_distances(gdf, k):
    # Compute the points for the k-NN distance plot
    knn_distances = []
    for i in range(len(gdf)):
        knn_distances.append(knn_distance(gdf, i, k))
    return sorted(knn_distances)

def plot_skewed(x, y, W, H, degree):
    skewed_x, skewed_y = skew(x, y, W, H)
    plt.scatter(skewed_x, skewed_y, label="Skewed sorted K-NN distances", color="cyan")
    plt.title("Skewed sorted K-NN distances")
    draw_polynomial_fitting(skewed_x, skewed_y, degree)
    draw_local_minimals(skewed_x, skewed_y, degree)
    plt.show()

def DBSCAN_eps_candidates(gdf, slope, k=3, degree=9, sorted_knn_distances=None):
    if sorted_knn_distances is None:
        sorted_knn_distances = knn_distances(gdf, k)
    x = range(len(sorted_knn_distances))

    # Find the "elbows" in the skewed graph
    minimals_x, minimals_y = skewed_minimals(
        x, sorted_knn_distances, len(x), max(sorted_knn_distances), degree, slope
    )
    
    return sorted(minimals_y)

def plot_elbow_detection(gdf, slope, k=3, degree=9):
    sorted_knn_distances = knn_distances(gdf, k)
    x = range(len(sorted_knn_distances))
    
    # Find the "elbows" in the skewed graph
    minimals_x, minimals_y = skewed_minimals(
        x, sorted_knn_distances, len(x), max(sorted_knn_distances), degree, slope
    )
    
    plt.clf()
    plt.cla()
    
    # Plot the original data
    plt.scatter(x, sorted_knn_distances, label="Sorted K-NN distances")
    plt.xlabel("Sorted indexes")
    plt.ylabel("K-NN distance")
    plt.title(
        "Plot of pairs of (index, K-NN distance) by the ascending order of K-NN distance"
    )
    plt.grid(True)
    draw_polynomial_fitting(
        x, sorted_knn_distances, degree
    )  # Draw the polynomial fitting for the original data
    plt.scatter(
        minimals_x,
        minimals_y,
        s=40,
        color="Magenta",
        label="Elbows obtained by the polyfit for the skewed data",
    )
    plt.legend()
    plt.show()

def skew(x, y, x_scale, y_scale, slope):
    if x_scale < y_scale:
        x = adjust_scale(x, y_scale / x_scale)
    else:
        y = adjust_scale(y, x_scale / y_scale)
    return rotate_point(x, y, -slope)


def inv_skew(x, y, x_scale, y_scale, slope):
    x, y = rotate_point(x, y, slope)
    if x_scale < y_scale:
        x = adjust_scale(x, x_scale / y_scale)
    else:
        y = adjust_scale(y, y_scale / x_scale)
    return x, y


def draw_polynomial_fitting(x, y, degree):
    # Perform polynomial fitting
    coefficients = np.polyfit(x, y, degree)

    # Generate fitted polynomial curve
    y_fit = np.polyval(coefficients, x)
    plt.plot(x, y_fit, color="red", label=f"Polynomial Fitted Curve. degree = {degree}")


def draw_local_minimals(x, y, degree):
    # Perform polynomial fitting
    coefficients = np.polyfit(x, y, degree)

    # Find the local minimal_s
    min_x = np.min(x)
    max_x = np.max(x)
    local_minimal_x, local_minimal_y = local_minimals(coefficients, min_x, max_x)
    
    print('local_minimal_x', local_minimal_x)
    print('local_minimal_y', local_minimal_y)
    
    # Plot
    plt.scatter(
        local_minimal_x,
        local_minimal_y,
        color="Magenta",
        label=f"Local Minimals. degree = {degree}",
    )


def local_minimals(coefficients, min_x, max_x):
    # Calculate the first derivative of the polynomial
    derivative = np.polyder(coefficients, 1)

    # Find the local minimal points
    roots = [root.real for root in np.roots(derivative) if np.isreal(root)]

    # Find the local minimum points which are inside the domain
    local_minimal_x = [
        x
        for x in roots
        if np.polyval(derivative, x - 1e-5) < 0
        and np.polyval(derivative, x + 1e-5) > 0
        and min_x <= x
        and x <= max_x
    ]
    local_minimal_y = np.polyval(coefficients, local_minimal_x)
    return (local_minimal_x, local_minimal_y)


def skewed_minimals(x, y, W, H, degree, slope):
    # Skew the data points
    skewed_x, skewed_y = skew(x, y, W, H, slope)

    # Perform polynomial fitting for the skewed pairs
    coefficients = np.polyfit(skewed_x, skewed_y, degree)

    # Find the local minimum points of the fitting polynomial which are inside the domain of the skewed plot
    min_x = np.min(skewed_x)
    max_x = np.max(skewed_x)
    local_minimal_x, local_minimal_y = local_minimals(coefficients, min_x, max_x)

    # Extract points which are in the domain of the original plot
    original_x, original_y = inv_skew(local_minimal_x, local_minimal_y, W, H, slope)
    original_xy = [
        (xy[0], xy[1]) for xy in zip(original_x, original_y) if 0 <= xy[0] and xy[0] < W
    ]
    result_x = [xy[0] for xy in original_xy]
    result_y = [xy[1] for xy in original_xy]
    return (result_x, result_y)

# TODO: research good ways to decide eps and min_samples.
def trajectory_DBSCAN(trajectory, eps, min_samples): 
    """
    Cluster trajectory points using the DBSCAN algorithm.

    Args:
        trajectory: A GeoDataFrame represents trajectry points with a projected CRS
        eps: the radius of the circle used in the DBSCAN algorithm
        min_samples: the threhold used in the DBSCAN algorithm. If the number of points inside the eps-circle is greater than or equal to min_samples, then the ponit is marked as core point.

    Returns:
        numpy.ndarray: the result of clustering
    """
    # Create a pyproj.Geod object based on the CRS
    geod = trajectory.crs.get_geod()
    
#     def metric(p, q):
#         _, _, distance = geod.inv(p[0], p[1], q[0], q[1])
#         return distance
    
#     # Extract the coordinate information
#     # print(['trajectory', trajectory.columns.tolist()])
#     # points = to_points(trajectory).tolist()
#     points = [(p.y, p.x) for p in trajectory['lat_lon']]
#     # print(['points', points])
#     # print(['type(points)', type(points)])
    # def metric(p, q):
    #     print(p.distance(q))
    #     return p.distance(q)
    points = to_points(trajectory).tolist()
    # Calculate pairwise distances using your custom distance function
    # distances = pairwise_distances(to_points(trajectory).tolist())
    # print(['distances', distances])

    # Create a DBSCAN object with your customized distance metric
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    # labels = dbscan.fit_predict(points)
    return labels

def mitigate_stay_points(gdf):
    # Find out clusters by DBSCAN
    eps_candidates = DBSCAN_eps_candidates(gdf)
    eps = eps_candidates[0]
    min_samples = 6 #4
    gdf["cluster"] = trajectory_DBSCAN(
            gdf, eps=eps, min_samples=min_samples
        )

    # Average each cluster
    def average_times(x):
        return pd.to_datetime(x).astype('int64').mean()

    # Fix this function if its perfomance is critical
    def average_points(x):
        x_mean = x.map(lambda p: p.x).mean()
        y_mean = x.map(lambda p: p.y).mean()
        return Point(x_mean, y_mean)

    excluded_gdf = gdf[gdf['cluster'] == -1]
    averaged_gdf  = gdf[gdf['cluster'] != -1].groupby('cluster').agg(
            id=('id', 'first'),
            time=('time', average_times),
            phenomenons=('phenomenons', 'first'),
            geometry=('geometry', average_points),
            lat_lon=('lat_lon', average_points),
            cluster=('cluster', 'first')
    )

    merged_df = pd.concat([averaged_gdf, excluded_gdf])
    return gpd.GeoDataFrame(data=merged_df, geometry='geometry', crs=gdf.crs)#.drop('cluster', axis=1)

def eliminate_outliers(gdf, slope, min_samples):
    # Find out clusters by DBSCAN
    gdf = gdf.copy()
    
    # Minimum value for eps
    # min_samples = 3 #4
    max_outlier_rate = 0.05
    sorted_knn_distances = knn_distances(gdf, min_samples)
    min_eps = sorted_knn_distances[math.ceil(len(gdf) * (1 - max_outlier_rate) - 1)]
    
    # Finding eps candidates by the elbow method
    eps_candidates = DBSCAN_eps_candidates(gdf, slope, sorted_knn_distances=sorted_knn_distances)
    eps = max(eps_candidates[-1], min_eps) if len(eps_candidates) > 0 else min_eps
    # if len(eps_candidates) == 0:
    #     gdf['cluster'] = -1
    #     return gdf
    
    # Decide the eps by taking minimum
    # eps = max(eps_candidates[-1], min_eps)
    
    # Eliminating outliers by DBSCAN
    gdf["cluster"] = trajectory_DBSCAN(
            gdf, eps=eps, min_samples=min_samples
        )
    return gdf[gdf['cluster'] != -1]

def empty_gdf():
    # Create an empty DataFrame
    empty_df = pd.DataFrame()

    # Add a geometry column to the DataFrame (can be any name you prefer)
    geometry_column_name = 'geometry'
    empty_df[geometry_column_name] = None
    return gpd.GeoDataFrame(empty_df, geometry=geometry_column_name)

def visualize_outlier_detection(perturbated_gdf, slope, min_samples):
    # perturbated_gdf = perturbate_selected_rows(gdf, perturbation_percentage=0.05, mean=0, std_dev=1, label_column = 'noise')
    eliminated_gdf = eliminate_outliers(perturbated_gdf, slope=slope, min_samples=min_samples).drop('cluster', axis=1)
    
    # Now, the 'noise' column contains True for the selected rows and False for the rest.
    
    # Perform the spatial difference overlay operation
    difference_gdf = perturbated_gdf.overlay(eliminated_gdf, how='difference')
    eliminated_outliers_gdf = difference_gdf[difference_gdf['noise'] == 1].copy()
    eliminated_nonoutliers_gdf = difference_gdf[difference_gdf['noise'] == 0].copy()
    
    # For visualization
    outliers_gdf = perturbated_gdf[perturbated_gdf['noise'] == 1]
    not_eliminated_outliers_gdf = outliers_gdf.overlay(eliminated_outliers_gdf, how = 'difference') if len(outliers_gdf) > 1 else empty_gdf()
    others_gdf = perturbated_gdf[perturbated_gdf['noise'] == 0].overlay(eliminated_nonoutliers_gdf, how = 'difference')
    
    num_total_outliers = len(perturbated_gdf[perturbated_gdf['noise'] == 1])
    num_eliminated_outliers = len(eliminated_outliers_gdf)
    num_total_nonoutliers = len(perturbated_gdf[perturbated_gdf['noise'] == 0])
    num_eliminated_nonoutliers = len(eliminated_nonoutliers_gdf)
    
    print('# of trajectory points', len(perturbated_gdf))
    print('# of outliers: ', num_total_outliers)
    if num_total_outliers > 0:
        print(f'# of detected outliers: {num_eliminated_outliers} ({num_eliminated_outliers / num_total_outliers * 100}) %)')
    else:
        print(f'# of detected outliers: {num_eliminated_outliers} (100 %)')
    print(f'# of misdetected non-outliers: {num_eliminated_nonoutliers} ({num_eliminated_nonoutliers / num_total_nonoutliers * 100} %)')
    
    eliminated_outliers_gdf['cluster'] = 'eliminated_outliers'
    not_eliminated_outliers_gdf['cluster'] = 'not_eliminated_outliers'
    eliminated_nonoutliers_gdf['cluster'] = 'eliminated_nonoutliers'
    others_gdf['cluster'] = 'others'
    color_mapping = {'eliminated_outliers': '#cc6633', 'not_eliminated_outliers': '#cc2400', 'eliminated_nonoutliers': '#56aeff', 'others': '#8f8b99'}
    
    # eliminated outliers, non eliminated outliers, eliminated non-outliers, others
    visualization_gdf = gpd.GeoDataFrame(pd.concat([eliminated_outliers_gdf, not_eliminated_outliers_gdf, eliminated_nonoutliers_gdf, others_gdf]), geometry='geometry')
    
    plotter.plot_trajectory(visualization_gdf, title = f'Diff (slope={slope}, min_samples={min_samples})', plot_clusters = True, color_mapping = color_mapping)