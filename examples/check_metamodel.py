import nevergrad as ng
import numpy as np


print(ng.__file__)


for kbudget in [1, 5, 10]:
 print("kbudget =", kbudget)
 for dim in [1,2]:
    print("Experiment in dimension ", dim)
    Nk=1   # we might play with greater values later
    N = 15
    #budget = 10*((N**dim)**2)
    budget = int(1*((N**1)**2)*kbudget)
    print(dim, N, budget)
    
    domain = ng.p.Array(shape=[N]*dim, lower=0., upper=1.)
    
    Y = []
    for k in range(Nk):
        y = np.random.rand(N**dim).reshape([N]*dim)
        Y += [y]
    
    def f(x):
       def subf(x, y):
           v = np.sum(np.abs(x.flatten())) if dim == 2 else 0
           v += np.sum(np.abs(x.transpose().flatten())) if dim == 2 else 0
           return v + np.sum((y-x)**2)  #np.sum(2. + ((x-.5)*(y-.5)))

       return np.min([subf(x,y) for y in Y])
    
    num_manips = 34
    #for optim in ["LognormalDiscreteOnePlusOne", "RFMetaModelLogNormal", "NeuralMetaModelLogNormal"]:
    for optim in ["OnePlusOne", "MetaModelOnePlusOne", "ImageMetaModelOnePlusOne", "CMASL", "CLengler"]:
       loss = [] 
       for k in range(num_manips):
         opt = ng.optimizers.registry[optim](domain, budget).minimize(f).value
         loss += [np.log(f(opt))]
       print(optim, "==>",  np.average(loss), "+-", np.std(loss)/np.sqrt(num_manips-1))
