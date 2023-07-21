import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import osmnx as ox
import networkx as nx
import requests
import json
from scipy.optimize import curve_fit
from urllib.parse import urljoin
import numpy as np
# import fuzzy logic 
import skfuzzy as fuzz
import skfuzzy.membership as mf
from algorithms import mm_utils
# for plotting 
import matplotlib.pyplot as plt

def get_params(l, r, shape="s"):
    shape = shape.lower()
    if shape == "s":
        y = np.array([0.01, 0.5, 0.99])
    else:
        y = np.array([0.99, 0.5, 0.01])

    x = np.array([l, (l + r) / 2, r])
    slope = 1 / (r - l)

    def logistic_func(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))

    p0 = [1, slope, (l + r) / 2]
    params, _ = curve_fit(logistic_func, x, y, p0=p0)

    return params[1]

def get_mid(bounds, row):
    return (bounds.iloc[row, 0] + bounds.iloc[row, 1]) / 2

def init_vars():
    left_bounds = [3, 2, 0, 20, 25, 10, 20, 3, 4, 85, 90, 85, 90, -5, -5, 10, 15, 150, 0, 0, 5, 10]
    right_bounds = [6, 4, 2, 45, 60, 40, 50, 5, 6, 100, 120, 100, 120, 5, 10, 20, 30, 200, 1, 1, 15, 25]
    data = {"left_bounds": left_bounds, "right_bounds": right_bounds}
    df = pd.DataFrame(data, index=["speed_high", "speed_low", "speed_zero", "HE_small",
                                   "HE_large", "PD_short", "PD_long", "HDOP_good",
                                   "HDOP_bad", "alpha_low", "alpha_high", "beta_low", "beta_high",
                                   "delta_dist_neg", "delta_dist_pos",
                                   "HI_small", "HI_large", "HI_180", "connectivity_direct",
                                   "connectivity_indirect", "dist_err_small", "dist_err_large"])
    df["ID"] = range(1, len(df) + 1)
    return df


def FIS2(data_temp, plot):
    var_bounds = init_vars()
    #print(var_bounds.iloc[0,0])
    get_mid(var_bounds,0)

    m = np.empty((2, 13))
    m[:] = np.nan

    # assign each element
    # speed high
    m[0, 0] = get_params(var_bounds.iloc[0, 0], var_bounds.iloc[0, 1], "s")
    m[1, 0] = get_mid(var_bounds, 0)
    # speed low
    m[0, 1] = get_params(var_bounds.iloc[1, 0], var_bounds.iloc[1, 1], "z")
    m[1, 1] = get_mid(var_bounds, 1)
    # speed zero
    m[0, 2] = get_params(var_bounds.iloc[2, 0], var_bounds.iloc[2, 1], "z")
    m[1, 2] = get_mid(var_bounds, 2)
    # HDOP good 
    m[0, 3] = get_params(var_bounds.iloc[7, 0], var_bounds.iloc[7, 1], "z")
    m[1, 3] = get_mid(var_bounds, 7)
    # HDOP bad
    m[0, 4] = get_params(var_bounds.iloc[8, 0], var_bounds.iloc[8, 1], "s")
    m[1, 4] = get_mid(var_bounds, 8)
    # alpha low (less than 90)
    m[0, 5] = get_params(var_bounds.iloc[9, 0], var_bounds.iloc[9, 1], "z")
    m[1, 5] = get_mid(var_bounds, 9)
    # alpha high (more than 90)
    m[0, 6] = get_params(var_bounds.iloc[10, 0], var_bounds.iloc[10, 1], "s")
    m[1, 6] = get_mid(var_bounds, 10)
    # beta low
    m[0, 7] = get_params(var_bounds.iloc[11, 0], var_bounds.iloc[11, 1], "z")
    m[1, 7] = get_mid(var_bounds, 11)
    # beta high 
    m[0, 8] = get_params(var_bounds.iloc[12, 0], var_bounds.iloc[12, 1], "s")
    m[1, 8] = get_mid(var_bounds, 12)
    # delta_d_neg
    m[0, 9] = get_params(var_bounds.iloc[13, 0], var_bounds.iloc[13, 1], "z")
    m[1, 9] = get_mid(var_bounds, 13)
    # delta_d_pos
    m[0, 10] = get_params(var_bounds.iloc[14, 0], var_bounds.iloc[14, 1], "s")
    m[1, 10] = get_mid(var_bounds, 14)
    # HI small 
    m[0, 11] = get_params(var_bounds.iloc[15, 0], var_bounds.iloc[15, 1], "z")
    m[1, 11] = get_mid(var_bounds, 15)
    # HI large
    m[0, 12] = get_params(var_bounds.iloc[16, 0], var_bounds.iloc[16, 1], "s")
    m[1, 12] = get_mid(var_bounds, 16)
    
       # define range for each input 
    x_speed = np.arange(0, 50, 0.1)
    x_HDOP = np.arange(0, 20, 0.1)
    x_alpha = np.arange(0, 360, 0.1)
    x_beta = np.arange(0,360, 0.1)
    x_delta = np.arange(-500,500, 0.1)
    x_HI = np.arange(0, 360, 0.1)
    x_HI180 = np.arange(0, 360, 0.1)

    # Creating Membership function : 
    #input : range, x shift, width, 
    # note : parameter input is flipped from R function 
    speed_high = mf.sigmf(x_speed, m[1, 0], m[0, 0])
    speed_low = mf.sigmf(x_speed, m[1, 1], m[0, 1])
    speed_zero = mf.sigmf(x_speed, m[1,2], m[0,2])

    HDOP_good = mf.sigmf(x_HDOP, m[1, 3], m[0, 3])
    HDOP_bad = mf.sigmf(x_HDOP, m[1, 4], m[0, 4])

    alpha_low = mf.sigmf(x_alpha, m[1,5], m[0,5])
    alpha_high = mf.sigmf(x_alpha, m[1,6], m[0,6])

    beta_low = mf.sigmf(x_beta, m[1, 7], m[0, 7])
    beta_high = mf.sigmf(x_beta, m[1, 8], m[0, 8])

    delta_d_neg = mf.sigmf(x_delta, m[1, 9], m[0,9])
    delta_d_pos = mf.sigmf(x_delta, m[1,10], m[0,10])

    HI_small = mf.sigmf(x_HI, m[1, 11], m[0,11])
    HI_large = mf.sigmf(x_HI, m[1, 12], m[0,12])

    HI180 = mf.trapmf(x_HI180,[125, 150, 200, 225])

    if plot == True:
        # plot membership function 
        fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows = 7, figsize =(10, 25))

        ax0.plot(x_speed, speed_high, 'r', linewidth = 2, label = 'High')
        ax0.plot(x_speed, speed_low, 'g', linewidth = 2, label = 'Low')
        ax0.plot(x_speed, speed_zero, 'b', linewidth = 2, label = 'Zero')
        ax0.set_title('speed')
        ax0.legend()

        ax1.plot(x_HDOP, HDOP_good, 'r', linewidth = 2, label = 'Good')
        ax1.plot(x_HDOP, HDOP_bad, 'g', linewidth = 2, label = 'Bad')
        ax1.set_title('Horizontal Dilution of Precission')
        ax1.legend()

        ax2.plot(x_alpha, alpha_low, 'r', linewidth = 2, label = 'Below 90')
        ax2.plot(x_alpha, alpha_high, 'g', linewidth = 2, label = 'Above 90')
        ax2.set_title('alpha')
        ax2.legend()

        ax3.plot(x_beta, beta_low, 'r', linewidth = 2, label = 'Below 90')
        ax3.plot(x_beta, beta_high, 'g', linewidth = 2, label = 'Above 90')
        ax3.set_title('Beta')
        ax3.legend()

        ax4.plot(x_delta, delta_d_neg, 'r', linewidth = 2, label = 'Negative')
        ax4.plot(x_delta, delta_d_pos, 'g', linewidth = 2, label = 'Positive')
        ax4.set_title('Delta d')
        ax4.legend()

        ax5.plot(x_HI, HI_small, 'r', linewidth = 2, label = 'Small')
        ax5.plot(x_HI, HI_large, 'g', linewidth = 2, label = 'Large')
        ax5.set_title('HI')
        ax5.legend()

        ax6.plot(x_HI180, HI180, 'r', linewidth = 2, label = '180')
        ax6.set_title('HI 180')
        ax6.legend()

    speed_fit_high = fuzz.interp_membership(x_speed, speed_high, data_temp[0])
    speed_fit_low = fuzz.interp_membership(x_speed, speed_low, data_temp[0])
    speed_fit_zero = fuzz.interp_membership(x_speed, speed_zero, data_temp[0])

    HDOP_fit_good = fuzz.interp_membership(x_HDOP, HDOP_good, data_temp[1])
    HDOP_fit_bad = fuzz.interp_membership(x_HDOP, HDOP_bad, data_temp[1])

    alpha_fit_low = fuzz.interp_membership(x_alpha, alpha_low, data_temp[2])
    alpha_fit_high = fuzz.interp_membership(x_alpha, alpha_high, data_temp[2])

    beta_fit_low = fuzz.interp_membership(x_beta, beta_low, data_temp[3])
    beta_fit_high = fuzz.interp_membership(x_beta, beta_high, data_temp[3])

    delta_fit_neg = fuzz.interp_membership(x_delta, delta_d_neg, data_temp[4])
    delta_fit_pos = fuzz.interp_membership(x_delta, delta_d_pos, data_temp[4])

    HI_fit_small = fuzz.interp_membership(x_HI, HI_small, data_temp[5])
    HI_fit_large = fuzz.interp_membership(x_HI, HI_large, data_temp[5])

    HI180_fit = fuzz.interp_membership(x_HI180, HI180, data_temp[6])

    # weight for each rule 
    # need to be optimize later 
    Z = np.array([100, 10, 10, 100, 10 ,10, 10, 100, 50, 10, 50, 100])

    # initialize weight
    weight = np.zeros((1,12))
    # initialize output array 
    output = np.zeros((1, 12))

    # Create weight rules : 
    weight[0,0]= np.fmin(alpha_fit_low, beta_fit_low)
    weight[0,1] = np.fmin(delta_fit_pos, alpha_fit_high)
    weight[0,2] = np.fmin(delta_fit_pos, beta_fit_high)
    weight[0,3] = np.fmin(HI_fit_small, np.fmin(alpha_fit_low, beta_fit_low))
    weight[0,4] = np.fmin(HI_fit_small, np.fmin(delta_fit_pos, alpha_fit_high))
    weight[0,5] = np.fmin(HI_fit_small, np.fmin(delta_fit_pos, beta_fit_high))
    weight[0,6] = np.fmin(HI_fit_large, np.fmin(alpha_fit_low, beta_fit_low))
    weight[0,7] = np.fmin(HDOP_fit_good, speed_fit_zero)
    weight[0,8] = np.fmin(HDOP_fit_good, delta_fit_neg)
    weight[0,9] = np.fmin(HDOP_fit_good, delta_fit_pos)
    weight[0,10] = np.fmin(speed_fit_high, HI_fit_small)
    weight[0,11] = np.fmin(HDOP_fit_good, np.fmin(speed_fit_high, HI180_fit))

    # standardize the weigth 
    std_weight = weight / weight.sum()


    # Tsugeno-Kang method 
    output_array = np.multiply(std_weight,Z)

    output = output_array.sum()
    return output