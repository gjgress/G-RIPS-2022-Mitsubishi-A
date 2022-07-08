# -*- coding: utf-8 -*-
# Fielname = dmu380_offline_sim.py

"""
Stupid Algorithm
Created on 2022-06-27
@author: gjgress
"""

# import
import os
import struct
import platform
import math
import numpy as np
from ctypes import *
from ctypes import wintypes

# globals
VERSION = '1.0'


### Required    
class FMMSim(object):
    '''
    This is a 'stupid' algorithm which outputs an arbitrary route on a given network.
    Its purpose is for testing.
    '''
    def __init__(self):
        '''
        Args:
            None
        '''
        # algorithm description
        self.input = ['geometry']
        self.output = ['geometry']
        self.batch = True
        self.results = None
        # algorithm vars
        this_dir = os.path.dirname(__file__)

    ### Required
    def run(self, set_of_input, network):
        '''
        main procedure of the algorithm
        Args:
            set_of_input is a GeoDataFrame (and so must have a 'geometry' column)
            network is a GeoDataFrame object created from a MultiDiGraph (the default format for OSM)
        '''
        
        
        
        # results
        self.results = [time_step[0:output_len],\
                        pos[0:output_len, :],\
                        vel[0:output_len, :],\
                        euler_angles[0:output_len, :],\
                        rate_bias[0:output_len, :],\
                        accel_bias[0:output_len, :]]

    def update(self, gyro, acc, mag=np.array([0.0, 0.0, 0.0])):
        '''
        Mahony filter for gyro, acc and mag.
        Args:
        Returns:
        '''
        pass

    ###Required
    def results(self):
        '''
        Returns:
            algorithm results as specified in self.output
                algorithm time step, sec
                Euler angles [yaw pitch roll], rad
        '''
        return self.results

    ### Required/Optional?
    def reset(self):
        '''
        Reset the fusion process to uninitialized state.
        '''
        windll.kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
        windll.kernel32.FreeLibrary(self.sim_engine._handle)
        self.sim_engine = cdll.LoadLibrary(self.sim_lib)
        self.sim_engine.SimInitialize(pointer(self.sim_config))

    ### Optional
    def build_lib(self, dst_dir=None, src_dir=None):
        '''
        Build shared lib
        Args:
            dst_dir: dir to put the built libs in.
            src_dir: dir containing the source code.
        Returns:
            True if success, False if error.
        '''
        if self.ext == '.dll':
            print("Automatic generation of .dll is not supported.")
            return False
        this_dir = os.path.dirname(__file__)
        # get dir containing the source code
        if src_dir is None:
            src_dir = os.path.join(this_dir, '//home//dong//c_projects//dmu380_sim_src//')
        if not os.path.exists(src_dir):
            print('Source code directory ' + src_dir + ' does not exist.')
            return False
        # get dir to put the libs in
        if dst_dir is None:
            dst_dir = os.path.join(this_dir, './/dmu380_sim_lib//')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        algo_lib = 'libdmu380_algo_sim.so'
        sim_utilities_lib = 'libsim_utilities.so'
        # get current workding dir
        cwd = os.getcwd()
        # create the cmake dir
        cmake_dir = src_dir + '//cmake//'
        if not os.path.exists(cmake_dir):
            os.mkdir(cmake_dir)
        else:
            os.system("rm -rf " + cmake_dir + "*")
        # call cmake and make to build the libs
        os.chdir(cmake_dir)
        ret = os.system("cmake ..")
        ret = os.system("make")
        algo_lib = cmake_dir + 'algo//' + algo_lib
        sim_utilities_lib = cmake_dir + 'SimUtilities//' + sim_utilities_lib
        if os.path.exists(algo_lib) and os.path.exists(sim_utilities_lib):
            os.system("mv " + algo_lib + " " + dst_dir)
            os.system("mv " + sim_utilities_lib + " " + dst_dir)

        # restore working dir
        os.chdir(cwd)
        return True