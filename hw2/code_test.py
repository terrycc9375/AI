import numpy as np


# Cross Entropy Test
"""
pred = numpy.array([
	[0.1, 0.7, 0.2], # 1st prediction
    [0.3, 0.4, 0.3], # 2nd prediction
    [0.5, 0.3, 0.2], # 3rd prediction
	[0.2, 0.2, 0.6]  # 4th prediction
])

true = numpy.array([1, 1, 0, 2]) # true labels

N, C = pred.shape
print(N, C)
cross_entropy = -numpy.log(pred[numpy.arange(N), true])
print(cross_entropy)
"""

# Softmax Test

z = np.array([[0.7, 1.2, 1.4],
			  [0.2, 0.8, 0.5]])
softmax = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
print(softmax)
