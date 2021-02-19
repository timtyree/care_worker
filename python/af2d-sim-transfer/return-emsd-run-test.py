# bin/bash/env python3
# returns the ensemble mean squared displacement of long-lived spiral tip trajectories
# Tim Tyree
# 2.19.2021
import numpy as np, pandas as pd
from lib import *
from lib.utils.get_txt import get_txt
from lib.routines.return_tip_log import return_tips_from_txt
from lib.routines.return_longest_traj import return_longest_trajectories
import random
import os,sys

def run_main(L, diffCoef, txt_id1,txt_id2, mode='FK'):
	T_min=1000#ms - the minimum lifetime for a trajectory to be considered in the EMSD calculation
	omit_time=150#ms
	worker_dir=os.getcwd()
	width=L;height=L;
	if mode=='FK':
		V_threshold=0.4
		dt=0.025
	else:
		V_threshold=-50
		dt=0.1
		# K_o=7.#5.4 higher K_o should give shorter APD#
	tmax_sec=30.
	tmax_sec=.15 #max time to integratein seconds#COMMENT_HERE
	tmax=tmax_sec * 10**3
	dsdpixel=0.025#cm/pixel  # area=width*height*dsdpixel**2
	DT = 1.   #ms between spiral tip frames
	#NOTE: assert ( save_every_n_frames*dt==DT to floating point precision)
	DS=dsdpixel
	save_every_n_frames=int(DT/dt)
	tmin=100# milliseconds
	mem=0;#memory for tracking
	sr=width*2#search range for tracking
	round_t_to_n_digits=0
	tmin_early_stopping=100
	jump_thresh=30.
	################################################################
	# Download initial conditions and compute tip locations
	################################################################
	txt= get_txt(txt_id1,txt_id2,width,height,worker_dir,mode=mode)
	#delete the mother initial condition (as she is >200MB)
	# os.remove(os.path.join('ic','ic1800x1800.npz'))#(NEVERMIND CAUSED BUG)UNCOMMENT_HERE
	df=return_tips_from_txt(
	    txt=txt,
	    h=dt,
	    tmax=tmax,
	    V_threshold=V_threshold,
	    dsdpixel=dsdpixel,
	    tmin_early_stopping=tmin_early_stopping,
	    save_every_n_frames=save_every_n_frames,mode=mode)
	del txt

	#drop columns that won't be used
	column_drop_list=list(set(df.columns.values).difference(['frame','x','y','t','particle']))
	df.drop(columns=column_drop_list,inplace=True)

	################################################################
	# Track, unwrap, and select sufficiently long lived trajectories
	################################################################
	if df is None:
		print("no tips were detected...")
		print(f'Printing Inputs:\nL={L}, diffCoef={diffCoef}, txt_id1={txt_id1}, txt_id2={txt_id2}')
	else:
		# # UNCOMMENT_HERE
		# # turn off logging
		# null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
		# # save the current file descriptors to a tuple
		# save = os.dup(1), os.dup(2)
		# # put /dev/null fds on 1 and 2
		# os.dup2(null_fds[0], 1)
		# os.dup2(null_fds[1], 2)

		#track tips, unwrap them, and then truncate at the first jump of distance greater than jump_thresh pixels
		df=return_unwrapped_trajectory(df, width, height, sr, mem, dsdpixel,round_t_to_n_digits,jump_thresh)#, **kwargs)

		# # UNCOMMENT_HERE
		# # turn on logging
		# os.dup2(save[0], 1)
		# os.dup2(save[1], 2)
		# # close the temporary fds
		# os.close(null_fds[0])
		# os.close(null_fds[1])

		# DT=compute_time_between_frames(df);#print(f"DT={DT}")#might cause problem in output
		df=get_all_longer_than(df,DT,T_min=T_min)
		#count remaining individuals
		num_individuals=len(list(set(df.particle.values)))
		if num_individuals==0:
			print(f"no spiral tips lived longer than {T_min} ms...")
			print(f'Printing Inputs:\nL={L}, diffCoef={diffCoef}, txt_id1={txt_id1}, txt_id2={txt_id2}')
		else:
			################################################################
			# Compute and return the ensemble mean squared displacement
			################################################################
			#compute emsd
			df=compute_emsd(traj=df.copy(), DT=DT, omit_time=omit_time, printing=False,DS=DS)
			#print results
			print(f'mode={mode}')
			print(f'Printing Inputs:\nL={L}, diffCoef={diffCoef}, txt_id1={txt_id1}, txt_id2={txt_id2}, N={num_individuals}')
			# print(f"num_individuals={num_individuals}")
			print(f"Printing Outputs:")#" of longest unwrapped spiral tip trajectory were:")
			# with open('out.csv') as f:
			# 	for line in f:
			# 		print(line)
			print(df.to_string())

if __name__=="__main__":
	import sys
	from random import randint
	from time import sleep
	if len(sys.argv)<4:
		print("Error: not enough arguments specified")
	else:
		# parse arguments
        L		   = int(sys.argv[1].split(',')[0])
        diffCoef   = float(sys.argv[2].split(',')[0])
        txt_id1    = int(sys.argv[3].split(',')[0])
        txt_id2    = int(float(sys.argv[4].split(',')[0]))
		# #wait a randomly selected amount of time (10-100 seconds
		# sleep(randint(10,100)))#UNCOMMENT_HERE
		run_main(L, diffCoef, txt_id1, txt_id2)
