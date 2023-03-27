import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def launch_optim(n_couches):
    # A pour Antoine.
    algo = "DE"
    function = "chirpedHC[400-650.50]"
    budget = 100000
    nb_runs = 30

    X_min=np.hstack(20*np.ones(n_couches))
    X_max=np.hstack(250*np.ones(n_couches))

    results = []

    for k in range(nb_runs):
        depop = 30
        [best,convergence,recom] = algos.DEvol(photonics.chirped,budget,X_min,X_max,depop)
        results.append([best]+convergence[::100]+[convergence[-1]])
        print(f"Run {k} with {algo} on {function} with {n_couches} layers")

    file_name = f"ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.asarray(results,dtype = object)
    np.save(file_name,results)

# Relire les données

Parallel(n_jobs = 24)(delayed(launch_optim)(n_couches) for n_couches in [20,30,40,50,60])
