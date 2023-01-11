import numpy as np
import scipy.io as scio
import scipy.optimize as optim
import photonics
import algos
import matplotlib.pyplot as plt
import numpy.matlib
# Test de DE - sur un miroir de Bragg
n_couches=20
X_min=np.hstack((2*np.ones(n_couches),20*np.ones(n_couches)))
X_max=np.hstack((3*np.ones(n_couches),180*np.ones(n_couches)))

pas=0.3*np.hstack((1*np.ones(n_couches),100*np.ones(n_couches)))

de_results = []
descent_results = []
de_values=[]
steep_values=[]
for k in range(7):
    [de_best,de_convergence] =  algos.DEvol(photonics.bragg,15000,X_min,X_max,30)
    de_results.append([de_best,de_convergence])
    xmin=np.hstack((2*np.ones(n_couches),20*np.ones(n_couches)))
    xmax=np.hstack((3*np.ones(n_couches),180*np.ones(n_couches)))
    start=xmin+np.random.random_sample(2*n_couches)*(xmax-xmin)
    [x_desc,convergence]=algos.descente(photonics.bragg,pas,start,15000,xmin,xmax)
    descent_results.append([x_desc,convergence])
    if de_convergence[-1]<convergence[-1]:
        print(k,": DE wins")
    else:
        print(k,": Descent wins")
    de_values.append(de_convergence[-1])
    steep_values.append(convergence[-1])

de_sorted = np.sort(de_values)
steep_sorted = np.sort(steep_values)
plt.plot(de_sorted, 'DE')
plt.plot(steep_sorted, 'GD')
plt.legend()
plt.savefig("compa.fig")
reference = np.array([3.,2.]*10 + [600/(4*np.sqrt(3)),600/(4*np.sqrt(2))]*10)
#reference = np.hstack((np.matlib.repmat([3,2],1,int(n/2)),np.matlib.repmat([600/(4*np.sqrt(3)),600/(4*np.sqrt(2))],1,int(n/2))))
photonics.bragg(reference)
