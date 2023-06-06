#!/bin/bash

jupyter-lab --ip localhost --port 8889 --collaborative --ServerApp.token='mitsubishi' --ServerApp.iopub_data_rate_limit=1.0e5 --ExecutePreprocessor.timeout=-1
