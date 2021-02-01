#!/bin/bash
module load python/3.7.0
python3 -m venv worker_env
source worker_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numba
python3 -m pip install scikit-image trackpy
python3 -m pip install pandas scipy matplotlib
tar czf worker_env.tar.gz worker_env
deactivate