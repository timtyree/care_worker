import os, numpy as np
from .chunk_array import chunk_array
from .gdown import download_file_from_google_drive

def load_buffer(data_dir,**kwargs):
	cwd=os.getcwd()
	if data_dir[-4:]=='.npy':
		txt = np.load(data_dir)
	elif data_dir[-4:]=='.npz':
		txt = np.load(data_dir,**kwargs)
		txt = txt[txt.files[0]]  #only take the first buffer because there's typically one
	elif data_dir[-4:]=='.txt':
		txt = np.loadtxt(data_dir,**kwargs)
		widthxheight,chno=txt.shape#(1800, 1800, 18)
		width=int(np.sqrt(widthxheight))
		height=width#1800
		txt = txt.reshape((width,height,chno)).copy() #only take the first buffer because there's typically one
	else:
		print(f"Warning: file format not supported {data_dir}.")
		raise Exception(f"Warning: file format not supported {data_dir}.")
	os.chdir(cwd)
	return txt

def run_downloader(gid,txt_ic_fn='ic/ic1800x1800.txt'):
	destination=txt_ic_fn
	retval=download_file_from_google_drive(gid, destination)
	return None

def get_gid(txt_id):
	import random
	def decision(probability):
	    return random.random() < probability
	#two gid's per texture lowers the load on google drive servers
	if txt_id==0:#at time, 1210
		if decision(0.5):
			gid='15Dw_ZVj1AodyqSQpBdrvggxSBkL5-YIp'
		else:
			gid='18XXdQXfjMbDKCREmOOG7Pat7eCeGaJm9'
	if txt_id==1:#at time, 2020
		if decision(0.5):
			gid='1zyAZab_5U4jD6Xn6OK_kKKSSTh65dHkX'
		else:
			gid='1e5ojFGN2mHN8YJlDvTBHxkVfJPZTZXvq'
	if txt_id==2:#at time, 2830
		if decision(0.5):
			gid='1cBQ-knunPqiDlshhptUzZlvFV6luR_Aj'
		else:
			gid='1-mRKafJypVeopGZxiEZohrMT4o8kcIyp'
	if txt_id==3:#at time, 3640
		if decision(0.5):
			gid='1EOMR7izLM0bS4GVXr9daSHo9FtsW56vo'
		else:
			gid='1Hu-w9vChqRAR4EZSuLqZbG7qh5gFgBbL'
	return gid

def download_txt(txt_id,worker_dir,rm_father_ic=True):
	'''returns the first gdrive download file found in the directory, worker_dir.'''
	os.chdir(worker_dir)
	if not os.path.exists('ic'):
		os.mkdir('ic')
	gid=get_gid(txt_id)
	run_downloader(gid=gid)
	# cmd=f'gdown https://drive.google.com/uc?id={gid} -O ic/ic1800x1800.npz'
	# os.system(cmd)#at time, 1210
	# if txt_id==0:
	# 	# run_downloader(gid='1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu')
	# 	os.system('gdown https://drive.google.com/uc?id=1OYtQNnp5KnGfKMkskk7GeDQSCe3Mo7Gu -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==1:
	# 	# run_downloader(gid='1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD')
	# 	os.system('gdown https://drive.google.com/uc?id=1td_6aQHFWzvunt1kU14ViW5DZ69rUhMD -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==2:
	# 	# run_downloader(gid='12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u')
	# 	os.system('gdown https://drive.google.com/uc?id=12dLQ_YFwSAvuuZc1lhNsKPcv4QXZB86u -O ic/ic1800x1800.npz')#at time, 1210
	# if txt_id==3:
	# 	# run_downloader(gid='14SipoA-gemvfyuA5v9tAUQRP3Firmu8G')
	os.chdir(worker_dir)
	txt=load_buffer('ic/ic1800x1800.txt')#[0]#,allow_pickle=True)
	if rm_father_ic:
		os.remove('ic/ic1800x1800.txt')
	# txt=load_buffer('ic/ic1800x1800.npz')[0]#,allow_pickle=True)
	return txt

def get_txt_lst(txt_id1,width,height,worker_dir,**kwargs):
	txt_in=download_txt(txt_id1,worker_dir)
	array_lst = chunk_array(txt_in, width, height, typeout='float64')
	return array_lst

def get_txt(txt_id1,txt_id2,width,height,worker_dir,**kwargs):
	array_lst=get_txt_lst(txt_id1,width,height,worker_dir,**kwargs)
	# N=len(array_lst)
	try:
		txt=array_lst[txt_id2]
	except IndexError as e:
		import random
		print (f'IndexError for {(width,txt_id1,txt_id2)}')
		print ( e )
		txt_id2=random.randint(0,len(array_lst)-1)
		print (f'Choosing txt_id2={txt_id2}...')
		txt=array_lst[txt_id2]#-1]
	return txt

if __name__=='__main__':
	os.get_cwd()
	for input_fn in sys.argv[1:]:
		download_txt(txt_id,worker_dir)
