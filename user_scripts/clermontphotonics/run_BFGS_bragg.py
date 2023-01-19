import photonics
import algos
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

def launch_optim(n_couches):
    # A pour Antoine.
    runner = "A"
    algo = "BFGS"
    function = "bragg"
    budget = 20000
    nb_runs = 50

    X_min=np.concatenate((np.hstack(2.*np.ones(n_couches)),np.hstack(20.*np.ones(n_couches))))
    X_max=np.concatenate((np.hstack(3.*np.ones(n_couches)),np.hstack(180.*np.ones(n_couches))))
    pas=np.concatenate((np.hstack(0.1*np.ones(n_couches)),np.hstack(10*np.ones(n_couches))))

    results = []

    for k in range(nb_runs):
        start=X_min+np.random.random_sample(int(2*n_couches))*(X_max-X_min)
        [best,convergence,recom] = algos.descente2(photonics.bragg,pas,start,budget,X_min,X_max)
        results.append([best,convergence])
        print(f"Run {k} with {algo} on {function} with {n_couches} layers")

    file_name = f"Res1/out_{function}_{algo}_{n_couches}_{budget}_{runner}.npy"
    results = np.asarray(results,dtype = object)
    np.save(file_name,results)

# Relire les données

Parallel(n_jobs = 24)(delayed(launch_optim)(n_couches) for n_couches in [20,40,60,80,100,120,140])
