#!/bin/bash

jupyter-lab --ip ${1:-localhost} --port 8888 --collaborative --ServerApp.token='mitsubishi' --ServerApp.iopub_data_rate_limit=1.0e10 --ExecutePreprocessor.timeout=-1
