import numpy as np, pandas as pd
import re,os,sys

# def read_inputs(line,trgt_lst):
#     input_lst=[]
#     for trgt in trgt_lst:
#         i=line.find(trgt)
#         l=len(trgt)
#         val=eval(line[i+l:].split(',')[0])
#         input_lst.append(val)
#     return input_lst


def parse_inputs(line):
    result_keys = re.findall(r'\S+=', line)
    result_values = re.findall(r'=\d+', line)
    if len(result_values)==0:
        pass
        #     return None
    input_dict={}
    for k,v in zip(result_keys,result_values):
        key=k.split('=')[0].split(',')[-1]
        value=eval(v.split('=')[-1])
        input_dict[key]=value
    return input_dict

def next_line(f):
    yield f.readline()

def parse_input_params(input_fn):
    parse_following_line=False
    line_no=0
    #read through until the outputs starts
    with open(input_fn) as f:
        for line in f:
            line_no+=1
            input_dict=parse_inputs(line)
            if len(list(input_dict))>0:
                found_output=True
                if found_output and parse_following_line:
                    line_no+=1
                    next_line_dict=parse_inputs(line)
                    input_dict.update(next_line_dict)
                return line_no, input_dict
    return -1,{}

def parse_input_fn(input_fn):
    printing=False
    line_no, input_dict=parse_input_params(input_fn)
    found_output=line_no!=-1
    if found_output and printing:
        print(f'the inputs found on line {line_no} were {input_dict}.')
    try:
        #         print(line_no)
        df=pd.read_csv(input_fn,header=line_no+1,delim_whitespace=True).astype({'n':int})
    except Exception:
        #only 1 line found in file, input_fn, skip
        return None

    df.drop(columns=['index'],inplace=True)
    #reset frame number to start at zero
    # df['frame']=df['frame']-df['frame'].values[0]
    #updating df...
    for k in input_dict.keys():
        df[k]=input_dict[k]

    #(optional) if a job was done twice for the same setting, remove it using the pandas.DataFrame.index set datastructure
    # index_labels=list(input_dict.keys())
    # # index_labels=["L","txt_id1","txt_id2"]
    # # # column_labels=["frame","x","y","particle","n","grad_ux","grad_uy","grad_vx","grad_vy"]
    # df.set_index(keys=index_labels,inplace=True)#, columns=column_labels)
    # # df=df[column_labels]
    return df

def return_df_updated_with_input_folder(input_fn_lst):#,df=None):
    """    returns df with this schema,
    multindex L|txt_id1|txt_id2|frame|particle
    and columns frame,x,y,n,t,grad_ux,grad_uy,grad_vx,grad_vy,particle
    input_fn='/Users/timothytyree/Documents/GitHub/care_worker/python/Log/job.out.8051248.703'
    """
    K12_index_set={}
    df_lst=[]
    duration_lst=[]
#     if df is None:
#         del df
    for input_fn in input_fn_lst:
        retval= parse_input_fn(input_fn)
        # df.unstack(1).unstack(1).head()
        if retval is not None:
            # TODO(optional for faster runtime): update a df_out only if it does not yet contain input_dict
            # TODO(optional test): assert that whenever an output is repeated,
            # that their results are equal (to floating point precision)
            NI=4#count the number of times the first NI inputs were used
            K12_index=retval.index.values[0][:NI]
            src="{}_{}_{}_{}".format(*K12_index)
            retval['src']=src
            #count if K12_index exists in the set of current index values
            try:
                K12_index_set[K12_index]+=1
            except KeyError:
                #then add K12_index to the set
                K12_index_set[K12_index]=0
                # K12_index_set.update({K12_index:0})
                df_lst.append(retval)
                # duration=retval.t.max()-retval.t.min()#ms
                # duration_lst.append(duration)
                # print(f"the src={src} spiral tip lasted {retval.t.max()-retval.t.min()} ms.")
    #     print(K12_index_set)
    # print(f"the mean duration was {np.mean(duration_lst):.2f} ms")
    # print(f"the max duration was {np.max(duration_lst):.2f} ms")
    return df_lst,K12_index_set

if __name__=="__main__":
	# save_fn='longest_traj_by_area_pbc.csv'
    save_fn=sys.argv[1].split(',')[0]
	log_folder="osg_output/Log"
	# log_folder="/Users/timothytyree/Documents/GitHub/care_worker/python/osg_output/Log"
	os.chdir(log_folder)

	input_fn_lst=[os.path.abspath(fn) for fn in os.listdir() if fn.find('job.out.')!=-1]
	print(f"the number of output files in folder is {len(input_fn_lst)}.")

	df_lst,K12_index_set=return_df_updated_with_input_folder(input_fn_lst)
	# df.index.values
	# K12_index_set
	print(f"results were successfully parsed from {len(list(K12_index_set))} distinct trials.")

	df=pd.concat(df_lst)
	# df.head()

	#save df as csv in care
	# save_folder=os.path.join(nb_dir,'Data/cloud_results')
	# save_folder='..'
	# os.chdir(save_folder)
	df.to_csv(save_fn)
	print(f'consolidated DataFrame saved in {os.path.abspath(save_fn)}')
