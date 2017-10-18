import numpy as np
import matplotlib.pyplot as plt
import pdb

def softmax(x):

	y = np.exp(x)/np.sum(np.exp(x),axis = 0)

	return y

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
# scores = np.array([1, 2, 3])

print softmax(scores)
pdb.set_trace()

x = np.arange(-2.0,6.0,0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

plt.plot(x, softmax(scores/10.).T, linewidth = 2)
plt.show()

