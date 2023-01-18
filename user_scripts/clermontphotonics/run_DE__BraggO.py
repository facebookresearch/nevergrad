import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def launch_optim(n_couches):
    # A pour Antoine.
    runner = "A"
    algo = "DE"
    function = "BraggO"
    budget = 10000
    nb_runs = 50

    X_min=np.hstack(20*np.ones(n_couches))
    X_max=np.hstack(180*np.ones(n_couches))

    results = []

    for k in range(nb_runs):
        depop = 30
        [best,convergence,recom] = algos.DEvol(photonics.bragg_origin,budget,X_min,X_max,depop)
        results.append([best,convergence])
        print(f"Run {k} with {algo} on {function} with {n_couches} layers")

    file_name = f"Res1/out_{function}_{algo}_{n_couches}_{budget}_{runner}.npy"
    results = np.asarray(results,dtype = object)
    np.save(file_name,results)

# Relire les données

Parallel(n_jobs = 24)(delayed(launch_optim)(n_couches) for n_couches in [20,40,60,80,100,120,140])
