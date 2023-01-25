import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def launch_optim(n_couches):
#    algo = "DE"
#    algo = "DEscol"
#    algo = "DEbord"
    algo = "DEstruct"
    function = "bragg"
    budget = 60000
    nb_runs = 50
    X_min=np.concatenate((np.hstack(2.*np.ones(n_couches)),np.hstack(20.*np.ones(n_couches))))
    X_max=np.concatenate((np.hstack(3.*np.ones(n_couches)),np.hstack(180.*np.ones(n_couches))))

    results = []

    for k in range(nb_runs):
        depop = 30
        [best,convergence,recom] = algos.DEvol_struct_bragg(photonics.bragg,budget,X_min,X_max,depop)
        results.append([best,convergence])
        print(f"Run {k} with {algo} on {function} with {n_couches} layers")

    file_name = f"ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.asarray(results,dtype = object)
    np.save(file_name,results)

# Relire les données

#launch_optim(30)
Parallel(n_jobs = 18)(delayed(launch_optim)(n_couches) for n_couches in [20,40,60,80,100,120,140])
