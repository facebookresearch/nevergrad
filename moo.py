import sys
sys.path.append('/Users/oteytaud/moonevergrad')
import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
import numpy as np
import random

for dim in [20, 2, 10]:
  for N in [2, 3]:
    for size in [2, 5, 10]:

        myshifts = []
        for u in range(N):
            #myshifts += [np.array([np.cos(u)] * dim, copy=True)] #[1 * np.random.normal(size=dim)]
            myshifts += [1 * np.array(np.random.normal(size=dim), copy=True)]
        print(myshifts)
        funcs = lambda x: [np.sum((x - np.array(shift, copy=True)) ** 2) for shift in myshifts]
        #funcs = lambda x: [np.sum((x - i) ** 2) for i in range(N)]
        upper_bounds = None #funcs(np.array([N+2] * dim))
        print("UB=", upper_bounds)
        print("f(0)=", np.array([funcs(np.array([0.] * dim))]))
        f = MultiobjectiveFunction(multiobjective_function=funcs, upper_bounds=upper_bounds)
        print("pouet")
        print("dim, N, size = ", dim, N, size)
        print(f(np.array([0.] * dim)))
        print(f(np.array([1.] * dim)))
        print(f(np.array([-1.] * dim)))
        print(f(np.array([0.5] * dim)))
        for i in range(N):
            f(np.array(myshifts[i]))
        pf, _ = f.pareto_front()
        assert len(pf) >= size, f"{len(pf)} instead of {size}"
        print("we prefound ", len(pf), " points, whereas N=", N)
        optimizer = ng.optimizers.RandomSearch(parametrization=dim, budget=100000)
        recommendation = optimizer.minimize(f)
        optimizer = ng.optimizers.chainCMAwithRsqrt(parametrization=dim, budget=100000)
        recommendation = optimizer.minimize(f)
        optimizer = ng.optimizers.PSO(parametrization=dim, budget=100000)
        recommendation = optimizer.minimize(f)
        pf, _ = f.pareto_front()
        print("we found ", len(pf), " points, whereas N=", N)
        #assert len(pf) >= N
        print("*")
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
        
        ax[0].set(ylabel='Loss-covering', xlabel='Domain-covering', title=f'correlation{np.corrcoef(scoresdomain,scoresloss)[0][1]:.2f}')
        ax[1].set(xlabel='Hypervolume', title=f'corr{np.corrcoef(scoreshyper,scoresloss)[0][1]:.2f}, dim{dim}, {N}obj, size={size}')
        ax[0].grid()
        ax[1].grid()
        
        fig.savefig(f"correl_domaindim{dim}_numberofobjectivefunctions{N}_sizeofsampledparetosubset{size}.png")
