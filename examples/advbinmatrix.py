import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from concurrent import futures

from collections import defaultdict

cmap = plt.get_cmap('jet_r')
score = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

nmax=250
numxps = 7
list_of_pbs = ["simple", "complex", "flatparts", "combined", "smooth", "bands", "shift"]
list_of_pbs = ["simple", "complex", "combined", "hardcore"]
list_ratios = [0.2, 1, 5, 10, 20, 40, 80]


# REMOVE THIS !
#list_of_pbs = ["hardcore"]
#numxps = 2

print(ng.__file__)

def doall(list_of_pbs):
    def translate(mat, shift):
        mat2 = mat.copy()
        for i in range(len(mat)):
           mat2[i] = mat[(i+shift) % (len(mat))]
        return mat2
    
    algos = list(ng.optimizers.registry.keys())
    algos = [a for a in algos if "iscre" in a and "Noisy" not in a and "Optimi" not in a and a[-1] != "T"] + ["DiscreteDE", "cGA", "NGOpt", "NgIoh4", "NgIoh5", "NgIoh6", "NGOptRW", "NgIoh7"]
    #algos = ["DiscreteLenglerHalfOnePlusOne", "DiscreteLenglerFourthOnePlusOne", "DiscreteLenglerOnePlusOne", "RecombiningDiscreteLanglerOnePlusOne"] + [a for a in algos if "Smoot" in a]
    #algos = [a for a in algos if "Smooth" in a and "Portfolio" not in a] + ["DiscreteLenglerOnePlusOne"]
    #np.random.shuffle(algos)
    #algos = algos[:5]
    #print("algos=", algos)
    #assert "SmoothDiscreteOnePlusOne" in algos
    for n in range(10, nmax,5):

     for pb in list_of_pbs:
      print(f"For pb {pb} and n={n} we use {len(algos)} algorithms, namely {algos}")
      for xpi_ in range(numxps):
          target_matrix = np.zeros((n,n))
          for i in range(n):
             for j in range(int(n*(1+np.cos((i+xpi_)*2*3.14/n))/2)):
                 target_matrix[i][j] = 1.
          def testloss(binary_matrix):
             #return np.sum(binary_matrix) - 5 * np.sum(np.diag(binary_matrix))
           if pb == "simple":
             return np.sum((binary_matrix - target_matrix)**2)
           elif pb == "shift":
             avg = translate(binary_matrix, n // 4)
             return np.sum((binary_matrix - target_matrix)**2) + np.sum((binary_matrix - avg) ** 2)
           elif pb == "minishift":
             avg = translate(binary_matrix, 1)
             return np.sum((binary_matrix - target_matrix)**2) + np.sum((binary_matrix - avg) ** 2)
           elif pb == "smooth":
             return np.sum((binary_matrix - np.average(binary_matrix))**2) + np.sum((binary_matrix - target_matrix)**2)
           elif pb == "bands":
             return np.sum((np.average(1.*(binary_matrix>np.average(binary_matrix, axis=0)), axis=0)-0.5)**2)
           elif pb == "combined":
             return (np.sum(binary_matrix) - n*n/4)**2 + np.sum((binary_matrix - target_matrix)**2) + min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n // 2)]) ** 2
           elif pb == "complex":
             val = np.sum((binary_matrix - target_matrix)**2)
             val += min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n // 2)])
             return val
           elif pb == "hardcore":
             val = 0.
             nn = n*n // 5
             for k in range(int(nn)):
                 x = int(n * (np.cos(5*k*(6.28/nn)+xpi_*7) + 1) / 2.)
                 y = int(n * (np.sin(3*k*(6.28/nn)+xpi_*11) + 1) / 2.)
                 x = max(min(x, n-1), 0)
                 y = max(min(y, n-1), 0)
                 xm = min(max(0, int(x - 1 - (k/8)*nn)), n-1)
                 ym = min(max(0, int(y - 1 - (k/8)*nn)), n-1)
                 xM = min(max(0, int(x + 1 + (k/8)*nn)), n-1)
                 yM = min(max(0, int(y + 1 + (k/8)*nn)), n-1)
                 try:
                   #print(x,y,n, np.std(binary_matrix[xm:xM,ym:yM].flatten()))
                   val += np.std(binary_matrix[xm:xM,ym:yM].flatten()) + (((-1) ** k) * (binary_matrix[x,y]-0.5))
                   #val += 0.00 * np.std(binary_matrix[xm:xM,ym:yM].flatten()) + 1.0 * (0.5 + (1. + (-1) ** k) * (binary_matrix[x,y]-0.5))
                 except:
                   assert False, f"{xm} {xM} {ym} {yM} {n}"
             #print(val)
             #assert False
             return val
           elif pb == "flatparts":
             return min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n)])
           else:
             assert False, "unknown pb"
          #for k in algos:
           #try:
          def run_alg(k, ratio):
            print(f"{k} works on {pb} with edge {n}")
            optimizer = ng.optimizers.registry[k](parametrization=ng.p.Array(shape=(n,n),lower=0.,upper=1.).set_integer_casting(), budget=int(n*ratio))# num_workers=20)
            #with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(testloss)#, executor=executor, batch_mode=False)
            return k, ratio, testloss(recommendation.value)
          all_res = Parallel(n_jobs=80)(delayed(run_alg)(k, ratio) for k in algos for ratio in list_ratios for _ in range(3))
          for r in all_res:
              k = r[0]
              ratio = r[1]
              score[pb + str(ratio)][n][k] += [r[2]]
              print(f"=====> {k} gets {np.average(score[pb + str(ratio)][n][k])} on {pb} with edge {n}")
           #except Exception as e:
           # print(f"{k} does not work ({e}), maybe package missing ?")
         #if n < 10:
         #  continue
     
      for f in [6]: #,5,6]:
       for nbest in [7, 12, 19]: #, 10, 20, len(score[n].keys())]:
        for r in [7.]:
         for ratio in list_ratios:
          for u in range(3):
           plt.clf()
           sorted_algos = sorted(score[pb + str(ratio)][n].keys(), key=lambda k: np.average(score[pb + str(ratio)][n][k]))
           #if n > 10:
           #    algos = [a for a in algos if a in sorted_algos[:max(31, int(.7 * len(algos)))] ]
           for a in range(min(nbest, len(sorted_algos))):
               print(f"{a}/{nbest}: {sorted_algos[a]} for ratio {ratio} and pb {pb}")
           sorted_algos = sorted_algos[:nbest][::-1]
           #sorted_algos = [sorted_algos[i] for i in range(len(sorted_algos)) if i <= 18 or i >= len(sorted_algos) - 2]
           for i, k in enumerate(sorted_algos):
              #print(f" for size {n}, {k} gets {np.average(score[n][k])}")
              color = cmap(i/len(sorted_algos))

              x = sorted([ni for ni in score[pb + str(ratio)].keys()]) # if k in score[ni]]
              assert max(x) == n
              assert max(x) == x[-1]
              y = [np.average(score[pb + str(ratio)][x_][k]) for idx, x_ in enumerate(x)]
              #y = [np.average(score[pb + str(ratio)][x_][k]) * np.std(score[pb + str(ratio)][x_][k]) for idx, x_ in enumerate(x)]
              plt.plot(x, y, label=k+f" {y[-1]:.2f}", c=color)
              assert y[-1] == np.average(score[pb + str(ratio)][n][k]), f" {y[-1]} vs {np.average(score[pb + str(ratio)][n][k])}"
              y = [np.average(score[pb + str(ratio)][x_][k]) + ((-1)**idx) * np.std(score[pb + str(ratio)][x_][k]) for idx, x_ in enumerate(x)]
              plt.plot(x, y, c=color, linestyle='dashed')
              #print("plot",x,y,k)
              plt.text(x[-1], y[-1], k, {'rotation': min(r * (len(sorted_algos) - i - 1), 60), 'rotation_mode': 'anchor', 'horizontalalignment': 'left', 'verticalalignment': 'center',})
           plt.legend(loc=u,fontsize=f)
           plt.grid()
           plt.title(f"Comparison between algorithms\nfor binary matrices optimization:\nlower = better, {pb}, budget={ratio}.edge")
           plt.ylabel("average loss")
           plt.xlabel("edge size of the binary matrix")
           #plt.tight_layout()
           plt.savefig(f"binmatrix{n}_loclegend{u}_fontsize{f}_problem{pb}ratio{ratio}_plotting{nbest}_labelrotation{r}.png")
    #print(recommendation.value)
 
def go():   
  #Parallel(n_jobs=80)(delayed(doall)(pb) for pb in list_of_pbs)
  doall(list_of_pbs)
