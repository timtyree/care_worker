# bin/bash/env python3
# return_longest_unwrapped_traj returns the longest unwrapped trajectory
from lib import *
# from lib.routines.generate_tip_logs_LR_model_cy import *
# from lib.routines.kwargs_LR_model_cy import get_kwargs
from lib.utils.get_txt import get_txt
from lib.routines.return_tip_log import return_tips_from_txt
from lib.routines.return_longest_traj import return_longest_trajectories

# from lib.routines.dag_log_to_msd import *
# from lib.routines.compute_msd import get_longest_trajectories
import random
import os,sys

def run_main(txt_id1,mode='FK'):
	#randomly determine which initial condition to use
	width_in=1800
	max_area=900**2 #sqpixels
	min_area=200**2 #sqpixels
	L=int(np.floor(np.sqrt(100*random.randint(min_area/100,max_area/100))))#UNCOMMENT_HERE
	# L=112#int(np.floor(np.sqrt(random.uniform(min_area,max_area))))#COMMENT_HERE
	# N_txt_id1=4-1#the max index of the main-square
	# txt_id1=random.randint(0,N_txt_id1)
	N_txt_id2=int(np.floor((width_in-L) / L))**2-1#the max index to the sub-square
	if N_txt_id2>0:
		txt_id2=random.randint(0,N_txt_id2)
	else:
		txt_id2=0
	worker_dir=os.getcwd()#nb_dir#os.path.join(nb_dir,'worker')
	width=L;height=L;
	if mode=='FK':
		V_threshold=0.4
		dt=0.025
	else:
		V_threshold=-50
		dt=0.1
	# txt_id1=0;txt_id2=8#COMMENT_HERE
	tmax_sec=30.
	# tmax_sec=.15 #max time to integratein seconds#COMMENT_HERE
	tmax=tmax_sec * 10**3
	# K_o=7.#5.4 higher K_o should give shorter APD#
	dsdpixel=0.025#cm/pixel  # area=width*height*dsdpixel**2
	# dt = 0.1 # milliseconds
	DT = 2   #ms between spiral tip frames
	save_every_n_frames=int(DT/dt)
	tmin=100# milliseconds
	# sr=width*2;
	# mem=0;
	n_tips = 1
	round_t_to_n_digits=0
	tmin_early_stopping=100
	jump_thresh=30.
	# round_output_decimals
	################################
	# Setup file system and initial conditions
	################################
	# results_folder=f'results'
	# ic_fn=os.path.join(worker_dir,f'ic-in',f'ic{width}x{height}.{txt_id1}.{txt_id2}.npz')
	# ic_fn=os.path.join(worker_dir,'ic-in/ic312x312.0.8.npz')
	# param_fn = 'param_set_8_og.json'
	# #download and chunk initial conditions
	# os.chdir(worker_dir)
	# if not os.path.exists('ic-in'):
	# 	os.mkdir('ic-in')
	# if not os.path.exists('ic-out'):
	# 	os.mkdir('ic-out')
	# txt= get_txt(txt_id1,txt_id2,width,height,worker_dir)
	# np.savez_compressed(ic_fn,txt)
	#delete the mother initial condition (as she is >200MB)
	# os.chdir(worker_dir)
	os.remove(os.path.join('ic','ic1800x1800.npz'))#UNCOMMENT_HERE

	#initialize filesystem if not already initialized
	# cwd=os.getcwd()
	# base_folder   = '/'+os.path.join(*cwd.split('/')[:-1])
	# base_folder=worker_dir
	# try:
	# 	init_filesystem_bd(base_folder,
	# 					  results_folder=results_folder,
	# 					  subfolder_list=None)
	# except FileExistsError:
	# 	pass
		# print('file system already exists.')
	# #reset in out if ic-in is empty of npz files
	# #(may be skipped for remote)
	# enable_reset_in_out=True
	# if enable_reset_in_out:
	# 	os.chdir(base_folder)
	# 	def is_npz(s): return s[-4:]=='.npz'
	# 	retval = [fn for fn in os.listdir('ic-in') if is_npz(fn)]
	# 	if len(retval)==0:
	# 		os.rename('ic-in','ic-in2')
	# 		os.rename('ic-out','ic-in')
	# 		os.rename('ic-in2','ic-out')
	# 		# print('ic reset')
	txt= get_txt(txt_id1,txt_id2,width,height,worker_dir,mode=mode)
	df=return_tips_from_txt(
	    txt=txt,
	    h=dt,
	    tmax=tmax,
	    V_threshold=V_threshold,
	    dsdpixel=dsdpixel,
	    tmin_early_stopping=tmin_early_stopping,
	    save_every_n_frames=save_every_n_frames,mode='FK')
	del txt

	# df = return_longest_trajectories(df, width, height, dsdpixel, n_tips = n_tips, DT = DT,
	#                                 round_t_to_n_digits=round_t_to_n_digits, jump_thresh=jump_thresh, **kwargs)
	################################
	# Generate tip logs
	################################
	# def routine(ic):
	# #     kwargs = get_kwargs(ic)
	# 	kwargs = get_kwargs(ic, results_folder=results_folder)#,param_fn=param_fn)
	# #     kwargs['V_threshold'] =  -50.#mV
	# 	kwargs['dsdpixel']=dsdpixel
	# 	kwargs['h']=dt# kwargs['h']=0.01 for ds_1_param_set_8 for stability
	# 	kwargs['tmax_sec'] = tmax_sec #maximum time to be integrated in seconds
	# 	kwargs['tmax']= tmax_sec*10**3
	# 	kwargs['tmin']= tmin #millisecondds
	# 	kwargs['save_every_n_frames']=save_every_n_frames
	# 	#     kwargs['K_o']=K_o#TODO: expose K_o to this level if it should be varied extensively
	# 	kwargsout = generate_tip_logs_from_ic(ic, **kwargs)
	# 	#     print(f"completed birth_death_rates_from_ic: {ic}")
	# 	#     print(f"csv of spiral tip data stored in: {kwargs['completed_ic_dir']}")
	# 	#     return os.path.basename(ic)
	# 	return kwargsout#['data_dir_log']#output location
	# #get all .npz files in the file's directory
	# input_fn_lst=get_all_files_matching_pattern(file=ic_fn, trgt='.npz')
	# assert ( len(input_fn_lst)>0)
	#
	#generate tip logs
	# print("running simulation...")
	# retval=[routine(ic) for ic in input_fn_lst]
	# log_dir=os.path.join(worker_dir,results_folder,'Log',retval[0]['data_fn_log'])
	# #generate tip logs with multiple threads in parallel
	# b = db.from_sequence(input_fn_lst, npartitions=1).map(routine)
	# start = time.time()
	# retval = list(b)
	################################
	# Track, unwrap, and select longest trajectories
	################################
	if df is not None:
		# unwrapped_fn=log_to_unwrapped_trajectory(log_dir, use_cache=True,width=width, height=height,
		# 										 sr=sr, mem=mem)
		# UNCOMMENT_HERE
		# open 2 fds
		null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
		# save the current file descriptors to a tuple
		save = os.dup(1), os.dup(2)
		# put /dev/null fds on 1 and 2
		os.dup2(null_fds[0], 1)
		os.dup2(null_fds[1], 2)
		# *** run the function ***
		df = return_longest_trajectories(df, width, height, dsdpixel, n_tips = n_tips, DT = DT,
		                                round_t_to_n_digits=round_t_to_n_digits, jump_thresh=jump_thresh)#, **kwargs)

		# UNCOMMENT_HERE
		# restore file descriptors so I can print the results
		os.dup2(save[0], 1)
		os.dup2(save[1], 2)
		# close the temporary fds
		os.close(null_fds[0])
		os.close(null_fds[1])




		# #get longest unwrapped trajectory
		# df=get_longest_trajectories(unwrapped_fn,width=width,height=height,
		# 							n_tips = n_tips, DS = dsdpixel,DT = DT,
		# 							round_t_to_n_digits=round_t_to_n_digits)
		# #save .csv in log
		# os.chdir(worker_dir)
		# log_folder='wlog'
		# if not os.path.exists(log_folder):
		# 	os.mkdir(log_folder)
		# os.chdir(log_folder)
		# df.to_csv('out.csv', index=False)


		print(f'Printing Inputs:\nL={L}, txt_id1={txt_id1}, txt_id2={txt_id2}')
		print(f"Printing Outputs:")#" of longest unwrapped spiral tip trajectory were:")
		# with open('out.csv') as f:
		# 	for line in f:
		# 		print(line)
		print(df.to_string())
	else:
		print("no tips were detected...")
		print(f'Printing Inputs:\nL={L}, txt_id1={txt_id1}, txt_id2={txt_id2}')


if __name__=="__main__":
	import sys
	if len(sys.argv)==0:
		print("Example Usage: python3 test_run.py 1")
	else:
		# # parse arguments
		txt_id1 = int(sys.argv[1].split(',')[0])
		run_main(txt_id1)
