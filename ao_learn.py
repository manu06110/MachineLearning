import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle

data_root = '.' # Change me to store data elsewhere
imsize = 28
learn_rate = 0.01
Ntrain = 10000

#import back the data
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
f = open(pickle_file, 'rb')
data = pickle.load(f) #upload the data in a dictionary
locals().update(data) #recover the variables 
f.close()

# pick an image (for test)
im = train_dataset[5,:,:]
label = train_labels[5]

# transform it in a 1D vector
im_vec = im.flatten()
Ndim = len(im_vec)
Nletter = 10

#------------------------------------------
# FUNCTIONs
#------------------------------------------
#We need to put that into a vector and define our linear model (i.e. function that returns a vector)
def linear_model(x, wij, b):
	y = np.matmul(x,wij.T) + b
	return y

#Define softmax to transform into probability
def softmax(x):
	y = np.exp(x)/np.sum(np.exp(x),axis = 0)
	return y

#Define the target (for 1 image, by hand at the moment, it's an A)
def getTargetOutput(lbl):
	target = np.zeros([10])
	target[lbl] = 1
	return target

#Update the weights depending on the error
def updateWeights(wei,learn_rate,err):
	wei += (wei.T*learn_rate*err).T
	return wei

#Define the Error
def getDiff(model,target):
	return target - model

#------------------------------------------
#------------------------------------------

# Generate a Npix x Noutput array for weights
wei = np.random.randn(Nletter,Ndim)

# Generate a vector for constant
#const = np.random.randn(Nletter)
const = np.zeros(Nletter)

#Define the output target
target = getTargetOutput(label)

dif_vec = []
#Loop for training
for i in range(0,Ntrain):
	#Apply the linear model
	y = linear_model(im_vec,wei,const)
	#Convert it into probability
	model = softmax(y)
	#Measure the difference
	diff = getDiff(model,target)
	dif_vec.append(diff[4])
	#Update the weights
	updt_wei = updateWeights(wei,learn_rate,diff)
	wei = updt_wei
	#pdb.set_trace()
	

print(np.round(model))
pdb.set_trace()