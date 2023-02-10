import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt


X=np.array([100,200,200,100,200,100,150,200,0,100,200,200,100,200,150])
cost = photonics.psplit(X)
print(cost)
