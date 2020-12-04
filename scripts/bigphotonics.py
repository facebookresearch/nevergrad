# If Nevergrad is in .. we have to do this.
# It's better to run this script in a clone rather than with Nevergrad as a system import because
# we might to update the leaderboard.

print('run with << python -c "import bigphotonics ; bigphotonics.main()" >>')
import os
import time
os.chdir('..')

import numpy as np
import nevergrad as ng
print(ng.__file__)
from multiprocessing import Pool
from nevergrad.functions.photonics import Photonics
import submitit

def square(x):
    time.sleep(1)
    return sum((x-0.5)**2)

def photonics(problem='morpho', budget=50*300, num_workers=50):
    mode = "arraysubmitit"  # other: submitit, multiprocessing

    assert problem in ["bragg", "chirped", "morpho"]
    target = Photonics(problem, 60 if problem == "morpho" else 100)
    #target = Photonics(problem, 60 if problem == "morpho" else 100)
    opt = ng.optimizers.OnePlusOne(target.parametrization, num_workers=num_workers, budget=budget)
    num_inoculations = num_workers // 2
    for k in range(num_inoculations):
        suggestion = target.auto_suggest()
        epsilon = 1000000**(-1 + k/num_inoculations)
        sigma = np.random.uniform(1-epsilon, 1., size=suggestion.shape)
        opt.suggest(sigma*suggestion if k > 0 else suggestion)
    best_value = float("inf")
    new_best = None
    for i in range(budget // num_workers):
        population = []
        starttime = time.time()
        for j in range(num_workers):
            population += [opt.ask()] 
        #print('iteration ', i, ' population=', [p.value for p in population])

        if mode == "multiprocessing":
            pool = Pool()
            result = pool.map(target.__call__, [p.value for p in population])
            pool.close()
            if new_best is not None:
                target.evaluation_function(new_best.value)
                new_best = None
        elif mode == "submitit":
            executor = submitit.AutoExecutor(folder="log_bigphotonics")
            executor.update_parameters(timeout_min=60, cpus_per_task=60, slurm_partition="learnfair",
                slurm_array_parallelism=20)
            jobs = [executor.submit(target.__call__, p.value) for p in population]
            if new_best is not None:
                target.evaluation_function(new_best.value)
                new_best = None
            result = [job.result() for job in jobs]
        elif mode == "arraysubmitit":
            executor = submitit.AutoExecutor(folder="log_bigphotonics")
            executor.update_parameters(timeout_min=60, cpus_per_task=60, slurm_partition="learnfair",
                slurm_array_parallelism=num_workers)
            jobs = executor.map_array(target.__call__, [p.value for p in population])
            if new_best is not None:
                target.evaluation_function(new_best.value)
                new_best = None
            result = [job.result() for job in jobs]
        else:
            assert False, mode

        endtime = time.time()
        for j in range(len(population)):
            # Possibly update the leaderboard!
            if result[j] < best_value:
                best_value = result[j]
                new_best = population[j]
            opt.tell(population[j], result[j])
        print(opt.recommend())       
        print(f"Iteration {i}, Time taken {endtime-starttime} seconds, best={best_value}, mean={sum(result)/num_workers}, min={min(result)}.")
    if new_best is not None:
        target.evaluation_function(new_best.value)

def main():
    photonics()
