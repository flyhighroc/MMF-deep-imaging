import numpy as np
np.random.seed(42)

import cv2
from glob import glob

def save_stack(paths,out_path):
	images = [cv2.imread(x,0) for x in paths]
	row = np.hstack([x for x in images])
	status = cv2.imwrite(out_path,row)
	print(status,out_path)

def get_matching_patches():
	base_in_path = 'tmp/all/input/'
	image_paths = glob(base_in_path+'*.jpg')
	image_paths = np.random.choice(image_paths, size=5, replace=False)

	in_paths = image_paths
	save_stack(in_paths,'tmp/in.jpg')
	
	base_out_path = 'tmp/all/output/'
	out_paths = [x.split('/')[-1] for x in in_paths]
	out_paths = [base_out_path+x for x in out_paths]
	save_stack(out_paths,'tmp/out.jpg')


def get_patches():
	# base_in_path = 'tmp/mnist_m0_op/rp/'
	base_in_path = 'tmp/mnist_m0_op/mnist/'
	image_paths = glob(base_in_path+'*.jpg')
	image_paths = np.random.choice(image_paths, size=5, replace=False)

	in_paths = image_paths
	save_stack(in_paths,'tmp/m0.jpg')

if __name__ == '__main__':
	# get_matching_patches()
	get_patches()
