#!/bin/bash
module load python/3.7.0

#unpack virtual python environment and activate it
tar -xzf worker_env.tar.gz
python3 -m venv worker_env 
source worker_env/bin/activate &> ignore.txt
# python3 -m pip install scikit-learn cython scikit-image gdown &> ignore.txt
#get lib
git clone https://github.com/timtyree/care_worker.git
cd care_worker/python/worker
#compile cythonic measurements
cd lib/measure
./setup.sh &> ignore.txt
cd ../.. 

#run sim
python3 ./test_run.py
# python3 ./return_longest_unwrapped_traj.py $1 $2 $3 $4 $5 $6 $7


deactivate
