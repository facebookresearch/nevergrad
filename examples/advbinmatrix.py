import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from collections import defaultdict

score = defaultdict(lambda: defaultdict(list))

nmax=50
numxps=2

list_of_pbs = ["simple", "complex", "flatparts", "combined"]
print(ng.__file__)
def doall(pb):
    def translate(mat, shift):
        mat2 = mat.copy()
        for i in range(len(mat)):
           mat2[i] = mat[(i+shift) % (len(mat))]
        return mat2
    
    algos = list(ng.optimizers.registry.keys())
    #algos = [a for a in algos if "iscre" in a and "Noisy" not in a and "Optimi" not in a and a[-1] != "T"] + ["DiscreteDE", "cGA", "NGOpt", "NgIoh4", "NgIoh5", "NgIoh6", "NGOptRW"]
    #algos = ["DiscreteLenglerHalfOnePlusOne", "DiscreteLenglerFourthOnePlusOne", "DiscreteLenglerOnePlusOne", "RecombiningDiscreteLanglerOnePlusOne"] + [a for a in algos if "Smoot" in a]
    algos = [a for a in algos if "Smooth" in a and "Portfolio" not in a] + ["DiscreteLenglerOnePlusOne"]
    #np.random.shuffle(algos)
    #algos = algos[:5]
    print("algos=", algos)
    for n in range(5, nmax,5):
     for xpi_ in range(numxps):
      target_matrix = np.zeros((n,n))
      for i in range(n):
         for j in range(int(n*(1+np.cos(i*2*3.14/n))/2)):
             target_matrix[i][j] = 1.
      def testloss(binary_matrix):
         #return np.sum(binary_matrix) - 5 * np.sum(np.diag(binary_matrix))
       if pb == "simple":
         return np.sum((binary_matrix - target_matrix)**2)
       elif pb == "combined":
         return (np.sum(binary_matrix) - n*n/4)**2 + np.sum((binary_matrix - target_matrix)**2) + min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n // 2)]) ** 2
       elif pb == "complex":
         val = np.sum((binary_matrix - target_matrix)**2)
         val += min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n // 2)])
         return val
       elif pb == "flatparts":
         return min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n)])
       else:
         assert False, "unknown pb"
      for k in algos:
       #try:
        optimizer = ng.optimizers.registry[k](parametrization=ng.p.Array(shape=(n,n),lower=0.,upper=1.).set_integer_casting(), budget=nmax*10)
        recommendation = optimizer.minimize(testloss)
        score[n][k] += [testloss(recommendation.value)]
        print(f"{k} gets {np.average(score[n][k])}")
       #except Exception as e:
       # print(f"{k} does not work ({e}), maybe package missing ?")
     for u in range(9):
      for f in [4]: #,5,6]:
       for nbest in [5]: #, 10, 20, len(score[n].keys())]:
        for r in [0., 3., 7.]:
           plt.clf()
           for i, k in enumerate(sorted(score[n].keys(), key=lambda k:-np.average(score[n][k]))[:nbest]):
              #print(f" for size {n}, {k} gets {np.average(score[n][k])}")
              x = [ni for ni in score.keys()] # if k in score[ni]]
              y = [np.average(score[x_][k]) for x_ in x]
              plt.plot(x, y, label=k+f" {y[-1]:.2f}")
              plt.text(x[-1], y[-1], k, {'rotation': -min(r * i, 60)})
           plt.legend(loc=u,fontsize=f)
           plt.grid()
           plt.title(f"Comparison between algorithms\nfor binary matrices optimization:\nthe lower the better: {pb}")
           plt.ylabel("average loss")
           plt.xlabel("edge size of the binary matrix")
           #plt.tight_layout()
           plt.savefig(f"binmatrix{n}_loclegend{u}_fontsize{f}_problem{pb}_plotting{nbest}_labelrotation{r}.png")
    #print(recommendation.value)
 
def go():   
  Parallel(n_jobs=4)(delayed(doall)(pb) for pb in list_of_pbs)
