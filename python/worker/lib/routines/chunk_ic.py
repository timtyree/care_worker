#! /usr/bin/env python
#Chunk Initial Conditions (ic)
#Tim Tyree
#6.6.2020
import numpy as np, os, sys
#I don't believe these are needed
# from lib.operari import search_for_file, load_buffer

#########################
#### chunking initial conditions into pieces
#########################
def load_fortran(input_fn):
	#load txt
	arr = np.loadtxt(input_fn)
	#WJ's files come to 16 digits of machine precision.
	arr = arr.astype(dtype = np.float64, casting='same_kind', order='C', copy=False)

	N = arr.shape[0]
	n = np.sqrt(N)
	if not n == np.int(n):
		raise Exception('Error: Input array is not a square matrix!')
	n = int(n)
	arr = arr.reshape(n,n,arr.shape[-1], order='C')
	return arr

def _precision(arr):
	val = arr[0,0]
	return len(str(val))-2
def _dtype(arr):
	val = arr[0,0]
	return type(val)

def chunk_1800x1800_into_ninths_npz(txt_in, save_folder,count,typeout='float64'):
	'''suppose file_name is a bare string with the extension  ".npz"
	suppose txt_in is a numpy array that is (1800,1800,The_Rest)
	count is the ic number of the most recent ic00X file saved (increments by 1 before saving).
	'''
	count+=1
	foo_file_name_out = lambda cnt:f'ic{cnt:03}.npz'
	txt=txt_in.astype(typeout).copy()
	cwd=os.getcwd()
	os.chdir(save_folder)
	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[0:600,0:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[600:1200,0:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[1200:1800,0:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[0:600,600:1200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[600:1200,600:1200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[1200:1800,600:1200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[0:600,1200:1800]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[600:1200,1200:1800]
	np.savez_compressed(save_fn,txt_out)

	save_fn = foo_file_name_out(count);count+=1
	txt_out = txt[1200:1800,1200:1800]
	np.savez_compressed(save_fn,txt_out)
	os.chdir(cwd)
	return count

def chunk_600x600_into_ninths_npz(txt_in, file_name, save_folder,typeout='float32'):
	'''suppose file_name is a bare string with the extension ".npz"
	suppose txt_in is a numpy array that is (600,600,The_Rest)
	'''
	file_name=os.path.basename(file_name)
	file_name=os.path.join(*file_name.split('.')[:-1])
	txt=txt_in.astype(typeout).copy()
	cwd=os.getcwd()
	os.chdir(save_folder)
	save_fn = file_name+'.11.npz'
	txt_out = txt[0:200,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.12.npz'
	txt_out = txt[200:400,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.13.npz'
	txt_out = txt[400:600,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.21.npz'
	txt_out = txt[0:200,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.22.npz'
	txt_out = txt[200:400,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.23.npz'
	txt_out = txt[400:600,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.31.npz'
	txt_out = txt[0:200,400:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.32.npz'
	txt_out = txt[200:400,400:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.33.npz'
	txt_out = txt[400:600,400:600]
	np.savez_compressed(save_fn,txt_out)
	os.chdir(cwd)
	return True


def chunk_600x600_into_ninths(txt,file_name, save_folder):
	'''suppose file_name is a bare string with no extension such as ".npz"
	suppose txt is a numpy array that is (600,600,The_Rest)
	'''
	os.chdir(save_folder)
	save_fn = file_name+'.11.npz'
	txt_out = txt[0:200,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.12.npz'
	txt_out = txt[200:400,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.13.npz'
	txt_out = txt[400:600,0:200]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.21.npz'
	txt_out = txt[0:200,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.22.npz'
	txt_out = txt[200:400,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.23.npz'
	txt_out = txt[400:600,200:400]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.31.npz'
	txt_out = txt[0:200,400:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.32.npz'
	txt_out = txt[200:400,400:600]
	np.savez_compressed(save_fn,txt_out)

	save_fn = file_name+'.33.npz'
	txt_out = txt[400:600,400:600]
	np.savez_compressed(save_fn,txt_out)
	return True

def parse_input_fn(input_fn):
	'''file_name, input_folder, output_folder, tmp_folder = parse_input_fn(input_fn)
	changes to input_folder in the local scope.'''
	file_name     = input_fn.split('/')[-1]
	input_folder  = '/'+os.path.join(*input_fn.split('/')[:-1])
	os.chdir(input_folder)
	if not os.path.isabs(input_folder):
		input_folder = os.getcwd()
	base_folder   = '/'+os.path.join(*input_folder.split('/')[:-1])
	tmp_folder    = base_folder+'/ic-tmp'
	output_folder = base_folder+'/ic-in'
	return file_name, input_folder, output_folder, tmp_folder

def chunk_600x600_from_fn(input_fn):
	input_fn = os.path.abspath(input_fn)
	arr = load_fortran(input_fn)
	file_name, input_folder, output_folder, tmp_folder = parse_input_fn(input_fn)
	chunk_600x600_into_ninths(txt=arr,
							  file_name=file_name.replace('600','200'),
							  save_folder=output_folder
							 )
	print(f'done chunking {input_fn}!')
	return True

if __name__=='__main__':
	for input_fn in sys.argv[1:]:
		chunk_from_fn(input_fn)
