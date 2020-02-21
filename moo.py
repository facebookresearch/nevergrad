import sys
sys.path.append('/Users/oteytaud/moonevergrad')
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import numpy as np
import random

for dim in [2, 10, 100]:
  for N in [2, 3, 10]:
    for size in [2, 3, 5, 10]:

        myshifts = []
        for u in range(N):
            myshifts += [np.random.normal(size=dim)]
        allfuncs = [lambda x: ng.functions.corefuncs.sphere(x-x0) for x0 in myshifts]
        funcs = []
        for u in range(N):
            funcs += random.sample(allfuncs, 1)
        f = MultiobjectiveFunction(multiobjective_function=lambda x: [func(x) for func in funcs], upper_bounds=[func(np.array([0.] * dim)) for func in funcs])
        print(f(np.array([1.,2.])))
        
        optimizer = ng.optimizers.CMA(parametrization=dim, budget=10000)  # 3 is the dimension, 100 is the budget.
        recommendation = optimizer.minimize(f)
        
        # The function embeds its Pareto-front:
        _, scoresloss = f.pareto_front(size, "loss-covering")
        _, scoresdomain = f.pareto_front(size, "domain-covering")
        _, scoreshyper= f.pareto_front(size, "hypervolume")
        matrix = (list(map(list, zip(*[scoresloss,scoresdomain,scoreshyper]))))
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        
        import matplotlib
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(scoresloss, scoresdomain)
        ax[1].plot(scoresloss, scoreshyper)
        
        ax[0].set(xlabel='Loss-covering', ylabel='Domain-covering', title='')
        ax[1].set(xlabel='Loss-covering', ylabel='Hypervolume', title='')
        ax[0].grid()
        ax[1].grid()
        
        fig.savefig(f"correl{dim}_{N}_{size}.png")
