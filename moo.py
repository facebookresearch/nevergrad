import sys
sys.path.append('/Users/oteytaud/moonevergrad')
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import numpy as np
import random

for dim in [2, 10, 100]:
  for N in [2]:
    for size in [2, 3, 5, 10]:

        myshifts = []
        for u in range(N):
            myshifts += [np.random.normal(size=dim)]
        print(myshifts)
        funcs = lambda x: [np.sum((x - shift) ** 2) for shift in myshifts]
        upper_bounds = funcs(np.array([3.] * dim))
        print("UB=", upper_bounds)
        print(np.array([funcs(np.array([0.] * dim))]))
        f = MultiobjectiveFunction(multiobjective_function=funcs, upper_bounds=upper_bounds)
        print("pouet")
        print(dim, N, size)
        print(f(np.array([0.] * dim)))
        print(f(np.array([1.] * dim)))
        print(f(np.array([-1.] * dim)))
        print(f(np.array([0.5] * dim)))
        optimizer = ng.optimizers.OnePlusOne(parametrization=dim, budget=10000)
        recommendation = optimizer.minimize(f)
        
        # The function embeds its Pareto-front:
        _, scoresloss = f.pareto_front(size, "loss-covering")
        _, scoresdomain = f.pareto_front(size, "domain-covering")
        _, scoreshyper= f.pareto_front(size, "hypervolume")
        print("we get")
        print(scoresloss)
        print(scoresdomain)
        print(scoreshyper)
        matrix = (list(map(list, zip(*[scoresloss,scoresdomain,scoreshyper]))))
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        
        import matplotlib
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(scoresdomain, scoresloss, '*')
        ax[1].plot(scoreshyper, scoresloss, '*')
        
        ax[0].set(ylabel='Loss-covering', xlabel='Domain-covering', title=f'correlation{np.corrcoef(scoresdomain,scoresloss)[0][1]}')
        ax[1].set(xlabel='Hypervolume', title=f'correlation{np.corrcoef(scoreshyper,scoresloss)[0][1]}')
        ax[0].grid()
        ax[1].grid()
        
        fig.savefig(f"correl_dim{dim}_obj{N}_size{size}.png")
