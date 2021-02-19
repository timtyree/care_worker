#gen_run_emsd.py
import numpy as np
diffCoef_values=10**-3*np.array([0.01,0.05,0.25,0.5,0.75,1,1.25])#cm^2/ms
L_values=np.array([480,440,400,320,280,200])
# DS=0.025
# area_values=DS**2*L_values**2
max_num_trials_per_setting=200
width_in=1800
N_txt_id1=4#the max index of the father square
# N_txt_id2_values=np.floor((width_in-L_values) / L_values)**2-1#the max index to the sub-square
# N_txt_id2_values*4 #the max number of trials for a given setting
#iterate over settings, scheduling the largest jobs first
for L in L_values:
    N_txt_id2=int(np.floor((width_in-L) / L)**2)#the max index to the sub-square
    for diffCoef in diffCoef_values:
        num_trials=0
        #iterate over txt_id2
        for txt_id2 in range(N_txt_id2):
            #iterate over txt_id1
            for txt_id1 in range(N_txt_id1):
                if num_trials<max_num_trials_per_setting:
                    print(f"{L} {diffCoef} {txt_id1} {txt_id2}")
                    num_trials+=1
