import random
from glob import glob
from os.path import *
import os
import csv

def N_fold(full_list, n_fold):
    avg = len(full_list) / float(n_fold)
    out = []
    last = 0.0

    while last < len(full_list):
        out.append(full_list[int(last):int(last + avg)])
        last += avg
    return out

data_file_path = '/storage/local/rong/datasets/Brain4Cars/brain4cars_data/face_camera'
file_list = sorted(glob(join(data_file_path, '*/*/'))) ### file_list includes:"'./flownet2_face_camera/rturn/20141220_154451_747_897/', './flownet2_face_camera/rturn/20141220_161342_1556_1706/', "
random.shuffle(file_list)
print(join(data_file_path, '*/*/'))
n_fold_list = N_fold(file_list, 5) ## divide into 5 folds


for i in range (0,5):
	label = []
	subset = []
	n_frames = []
	file_location = []
	for j in range(0,5):
		this_subset = 'training'
		if j == i:
			this_subset = 'validation'
		for file in n_fold_list[j]:
			fbase = file[len(data_file_path):]
			ftarget_idx = fbase.find('/',1)
			ftarget = fbase[1:ftarget_idx]
			fnum = len(os.listdir(file)) - 2
			file_location.append(fbase[1:])
			n_frames.append(fnum)
			label.append(ftarget)
			subset.append(this_subset)

	with open('fold%d.csv'%i, 'a') as outcsv:   
		#configure writer to write standard csv file
		writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
		for (w,x,y,z) in zip (file_location,label,n_frames,subset):
			#Write item to outcsv
			writer.writerow([w,x,y,z])