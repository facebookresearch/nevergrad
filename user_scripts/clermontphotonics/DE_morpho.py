import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def launch_optim(n_couches):
#    algo = "DE"
#    algo = "DEscol"
#    algo = "DEclip"
    algo = "DE"
    function = "morpho"
    budget = 100000
    nb_runs = 15
    X_min = np.array([0.,30.,0.,0.]*n_couches)
    X_max = np.array([300.,570.,600.,100.]*n_couches)

    results = []

    for k in range(nb_runs):
        depop = 30
        [best,convergence,recom] = algos.DEvol(photonics.Nmorpho,budget,X_min,X_max,depop)
        results.append([best,convergence])
        print(f"Run {k} with {algo} on {function} with {n_couches} layers")

    file_name = f"ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.asarray(results,dtype = object)
    np.save(file_name,results)

# Relire les données

#launch_optim(30)
Parallel(n_jobs = 18)(delayed(launch_optim)(n_couches) for n_couches in [4,5,6,7,8,9,10])
