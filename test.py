import numpy as np
import BBBLearning as bb
from BBBLearning.utils.activation import *

a = np.random.randn(2,1)
print(a)
b = sigmoid(a)
c = relu(a)
d = softmax(a)
print("sigmoid: " + str(b))
print("relu: " + str(c))
print("sofmax: " + str(d))
