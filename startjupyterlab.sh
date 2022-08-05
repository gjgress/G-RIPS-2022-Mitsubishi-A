#!/bin/bash

jupyter-lab --ip 172.20.21.199 --port 8888 --collaborative --ServerApp.token='mitsubishi' --ServerApp.iopub_data_rate_limit=1.0e5 --ExecutePreprocessor.timeout=-1
