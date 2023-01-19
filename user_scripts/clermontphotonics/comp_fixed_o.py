# Olivier's script, used as a reference, probably not fully operational
# under this form and on this repo.

import numpy as np
import scipy.io as scio
import scipy.optimize as optim
import photonics
import algos
import matplotlib.pyplot as plt
import numpy.matlib

from joblib import Parallel, delayed

# Test de DE - sur un miroir de Bragg
def do_stuff(depop, n_couches):
  X_min=np.hstack(20*np.ones(n_couches))
  X_max=np.hstack(180*np.ones(n_couches))
  pas=np.hstack(1*np.ones(n_couches))
  for budget in [10001]:
   de_results = []
   descent_results = []
   bfgs_results = []
   chain_results = []
   de_values=[]
   steep_values=[]
   bfgs_values=[]
   chain_values=[]
   for k in range(100):
    [de_best,de_convergence] =  algos.DEvol(photonics.bragg_fixed,budget,X_min,X_max,depop)
    de_results.append([de_best,de_convergence])
    de_values.append(de_convergence[-1])
    xmin=np.hstack(20*np.ones(n_couches))
    xmax=np.hstack(180*np.ones(n_couches))
    start=xmin+np.random.random_sample(n_couches)*(xmax-xmin)

#    [x_desc,convergence]=algos.descente(photonics.bragg_fixed,pas,start,budget,xmin,xmax)
#    descent_results.append([x_desc,convergence])
#    steep_values.append(convergence[-1])

    [x_desc,convergence]=algos.descente2(photonics.bragg_fixed,pas,start,budget,xmin,xmax)
    bfgs_results.append([x_desc,convergence])
    bfgs_values.append(convergence[-1])

#    [de_best,de_convergence] =  algos.DEvol(photonics.bragg_fixed,budget // 2,X_min,X_max,depop)
#    [x_desc,convergence]=algos.descente2(photonics.bragg_fixed,pas,de_best,budget - (budget // 2),xmin,xmax)
#    chain_results.append([x_desc, de_convergence + convergence])
#    chain_values.append(convergence[-1])

    de_sorted = np.sort(de_values)
#    steep_sorted = np.sort(steep_values)
    bfgs_sorted = np.sort(bfgs_values)
#    chain_sorted = np.sort(chain_values)
    print(f"bfgs ==> {bfgs_sorted}")
#    print(f"steep ==> {steep_sorted}")
    print(f"de ==> {de_sorted}")
#    print(f"chain ==> {chain_sorted}")
    plt.clf()
#    plt.yscale("log")
    plt.plot(de_sorted, label='DE')
#    plt.plot(steep_sorted, label='GD')
    plt.plot(bfgs_sorted, label='BFGS')
#    plt.plot(chain_sorted, label='Chain')
    plt.legend()
    plt.title(f'budget={budget}')
    plt.savefig(f"output100_pop{depop}_{n_couches}layers_budget{budget}.png")

    plt.clf()
    plt.hist(bfgs_sorted,bins=10)
    plt.title("Results distribution")
    plt.savefig(f"hist_{n_couches}.png")


#for depop in [10, 20, 30, 40]:
# for n_couches in [10, 20, 40, 80]:
#  do_stuff(depop, n_couches)
Parallel(n_jobs=60)(delayed(do_stuff)(depop, n_couches) for n_couches in [20,40,60,80,100,120,140] for depop in [30])


reference = np.array([3.,2.]*10 + [600/(4*np.sqrt(3)),600/(4*np.sqrt(2))]*10)
#reference = np.hstack((np.matlib.repmat([3,2],1,int(n/2)),np.matlib.repmat([600/(4*np.sqrt(3)),600/(4*np.sqrt(2))],1,int(n/2))))
photonics.bragg(reference)
