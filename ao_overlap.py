# Work out how much overlap in percentage there is between sets of images.
import numpy as np
import os
import pdb
import pandas
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle

data_root = '.' # Change me to store data elsewhere
imsize = 28
overlap_tot = []
identical = 0. # 1=> perfect overlap other => degree of overlapping (/!\ does not work for the degree of overlapping /!\)

#import back the data
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
f = open(pickle_file, 'rb')
data = pickle.load(f) #upload the data in a dictionary
locals().update(data) #recover the variables 
f.close()

#Numbers of labels
Nlabels = 10 #there are 10 different possible letters

letters = ['A','B','C','D','E','F','G','H','I','J']

for i in range(0,Nlabels):
	overlap = [] # 0 = no replicate, 1 = replicated

	#extract all the similar letters in the validation and the training set
	ind_valid = np.where(valid_labels == i)[0]
	ind_test = np.where(test_labels == i)[0]
	ind_train = np.where(train_labels == i)[0]

	#Number of similar letters in the validation set (let's say A)
	Ntest = len(ind_valid)
	Ncomp = len(ind_train)

	#Loop around each of these same letters to compare to that of the training set
	for j in range(0,Ntest):
		#extract the jst validation dataset
		valid_dataset_j = valid_dataset[ind_valid[j],:,:]
		#replicate the array the number of time we have the trained letter to compare to
		rep_dataset_j = np.tile(valid_dataset_j,(Ncomp,1,1))
		#array of all the selected letter for comparison
		comp_dataset = train_dataset[ind_train,:,:]
		# if i==3:
		# 	rep_dataset_j[3,:,:] = comp_dataset[3,:,:]

		#Difference between the two
		dif_dataset = comp_dataset - rep_dataset_j

		#integrate over the square, if sum = 0 then identical
		int_dif = np.sum(dif_dataset,axis=(1,2))
		int_dif = np.absolute(int_dif)
		rang = np.max(int_dif)-np.min(int_dif)

		#is it in both dataset?
		o = np.where(int_dif <= identical*rang)[0]
		if len(o) != 0:
			overlap.append(1)
			overlap_tot.append(1)
		else:
			overlap.append(0)
			overlap_tot.append(0)

		# if i == 3:
		# 	if len(o) > 0:
		# 		pdb.set_trace()
	f_overlap = float(np.sum(overlap))/float(len(overlap))*100

	print(letters[i])
	print('fraction of overlap:',f_overlap)

f_overlap_tot = float(np.sum(overlap_tot))/float(len(overlap_tot))*100

print('total fraction of overlaping:',f_overlap_tot)
