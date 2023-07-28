
import numpy as np
from scipy.stats.mstats import gmean

# construct the pairwise comparison matrix for distance
def pairmat_dist(dist_data):
    
    M_dist = np.empty((len(dist_data), len(dist_data)))

    # definition of each component of the matrix
    for i in range(len(dist_data)):
        for j in range(len(dist_data)):
            if dist_data[j] - dist_data[i] < -15:
                M_dist[i, j] = 1/9
            elif -15 <= dist_data[j] - dist_data[i] < -13:
                M_dist[i, j] = 1/8
            elif -13 <= dist_data[j] - dist_data[i] < -11:
                M_dist[i, j] = 1/7
            elif -11 <= dist_data[j] - dist_data[i] < -9:
                M_dist[i, j] = 1/6
            elif -9 <= dist_data[j] - dist_data[i] < -7:
                M_dist[i, j] = 1/5
            elif -7 <= dist_data[j] - dist_data[i] < -5:
                M_dist[i, j] = 1/4
            elif -5 <= dist_data[j] - dist_data[i] < -3:
                M_dist[i, j] = 1/3
            elif -3 <= dist_data[j] - dist_data[i] < -1:
                M_dist[i, j] = 1/2
            elif -1 <= dist_data[j] - dist_data[i] <= 1:
                M_dist[i, j] = 1
            elif 1 < dist_data[j] - dist_data[i] <= 3:
                M_dist[i, j] = 2
            elif 3 < dist_data[j] - dist_data[i] <= 5:
                M_dist[i, j] = 3
            elif 5 < dist_data[j] - dist_data[i] <= 7:
                M_dist[i, j] = 4
            elif 7 < dist_data[j] - dist_data[i] <= 9:
                M_dist[i, j] = 5
            elif 9 < dist_data[j] - dist_data[i] <= 11:
                M_dist[i, j] = 6
            elif 11 < dist_data[j] - dist_data[i] <= 13:
                M_dist[i, j] = 7
            elif 13 < dist_data[j] - dist_data[i] <= 15:
                M_dist[i, j] = 8
            else:
                M_dist[i, j] = 9

    return M_dist
    
# construct the pairwise comparison matrix for direction
def pairmat_dir(dir_data):
    
    M_dir = np.empty((len(dir_data), len(dir_data)))

    # definition of each component of the matrix
    for i in range(len(dir_data)):
        for j in range(len(dir_data)):
            if dir_data[j] - dir_data[i] < -150:
                M_dir[i, j] = 1/9
            elif -150 <= dir_data[j] - dir_data[i] < -130:
                M_dir[i, j] = 1/8
            elif -130 <= dir_data[j] - dir_data[i] < -110:
                M_dir[i, j] = 1/7
            elif -110 <= dir_data[j] - dir_data[i] < -90:
                M_dir[i, j] = 1/6
            elif -90 <= dir_data[j] - dir_data[i] < -70:
                M_dir[i, j] = 1/5
            elif -70 <= dir_data[j] - dir_data[i] < -50:
                M_dir[i, j] = 1/4
            elif -50 <= dir_data[j] - dir_data[i] < -30:
                M_dir[i, j] = 1/3
            elif -30 <= dir_data[j] - dir_data[i] < -10:
                M_dir[i, j] = 1/2
            elif -10 <= dir_data[j] - dir_data[i] <= 10:
                M_dir[i, j] = 1
            elif 10 < dir_data[j] - dir_data[i] <= 30:
                M_dir[i, j] = 2
            elif 30 < dir_data[j] - dir_data[i] <= 50:
                M_dir[i, j] = 3
            elif 50 < dir_data[j] - dir_data[i] <= 70:
                M_dir[i, j] = 4
            elif 70 < dir_data[j] - dir_data[i] <= 90:
                M_dir[i, j] = 5
            elif 90 < dir_data[j] - dir_data[i] <= 110:
                M_dir[i, j] = 6
            elif 110 < dir_data[j] - dir_data[i] <= 130:
                M_dir[i, j] = 7
            elif 130 < dir_data[j] - dir_data[i] <= 150:
                M_dir[i, j] = 8
            else:
                M_dir[i, j] = 9

    return M_dir

# construct the pairwise comparison matrix for link connectivity
def pairmat_turnrest(turnrest_data):

    M_turnrest = np.empty((len(turnrest_data), len(turnrest_data)))

    # definition of each component of the matrix
    for i in range(len(turnrest_data)):
        for j in range(len(turnrest_data)):
            if turnrest_data[i] == 1 and turnrest_data[j] == 0:
                M_turnrest[i, j] = 9
            elif turnrest_data[i] == 0 and turnrest_data[j] == 1:
                M_turnrest[i, j] = 1/9
            else:
                M_turnrest[i, j] = 1

    return M_turnrest
    
# calculate a weight vector from the pairwise comparison matrix
def calc_weight(comparison_mat):

    # calculate the geometric mean of each row
    row_gmean = gmean(comparison_mat, axis=1)

    # obtain a weight vector
    S = sum(row_gmean)
    weight_vec = row_gmean / S

    return weight_vec

# return the index corresponding to the highest weight edge
def MMJfunc(dist_data, dir_data, turnrest_data, map_environment):
    # calculate a weight for each data
    weight_dist = calc_weight(pairmat_dist(dist_data))
    weight_dir = calc_weight(pairmat_dir(dir_data))
    weight_turnrest = calc_weight(pairmat_turnrest(turnrest_data))


    # relative importance vector
    Z = np.array([[0.0806, 0.372, 0.3585, 0.1894],
                  [0.4376, 0.4642, 0.0429, 0.0553],
                  [0.5563, 0.4237, 0.01, 0.01]
                 ])

    # total weight
    TW = weight_dist * Z[map_environment][0] + weight_dir * Z[map_environment][1] + weight_turnrest * (Z[map_environment][2] + Z[map_environment][3])

    return np.argmax(TW)

    





































