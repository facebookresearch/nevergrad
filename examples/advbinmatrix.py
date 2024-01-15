import nevergrad as ng
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from concurrent import futures
import scipy
import scipy.signal
import scipy.stats
from collections import defaultdict

cmap = plt.get_cmap('jet_r')

score = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
def d3d():
    return os.environ.get("d3d", "False") in ["true", "True", "1", "T", "t"]

nmax=250
numxps = 7
list_of_pbs = ["simple", "complex", "flatparts", "combined", "smooth", "bands", "shift"]
list_of_pbs = ["simple", "complex", "combined", "hardcore"] if not d3d() else ["simple", "complex", "combined"]
list_of_pbs += ["temp"]
maxalgos = 5000

# TODO !
#list_of_pbs = ["simple"]
#numxps = 2
#maxalgos = 3

list_ratios = [0.2, 1, 5, 10]
list_ratios = [0.05, 0.1, 0.2, 0.3]

list_of_pbs = [np.random.choice(list_of_pbs)]

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
    algos = [a for a in algos if "iscre" in a and "Noisy" not in a and "Optimi" not in a and a[-1] != "T"]# + ["DiscreteDE", "cGA", "NGOpt", "NgIoh4", "NgIoh5", "NgIoh6", "NGOptRW", "NgIoh7"]
    algos += ["BFGS", "LBFGSB"] #"SQP", "CMA", "PSO", "DE"]
    algos = algos[:maxalgos]
    #algos = ["DiscreteLenglerHalfOnePlusOne", "DiscreteLenglerFourthOnePlusOne", "DiscreteLenglerOnePlusOne", "RecombiningDiscreteLanglerOnePlusOne"] + [a for a in algos if "Smoot" in a]
    #algos = [a for a in algos if "Smooth" in a and "Portfolio" not in a] + ["DiscreteLenglerOnePlusOne"]
    #np.random.shuffle(algos)
    #algos = algos[:5]
    #print("algos=", algos)
    #assert "SmoothDiscreteOnePlusOne" in algos
    for n in (range(20, nmax,5) if d3d() else range(24, nmax,4)):

     for pb in list_of_pbs:
      print(f"For pb {pb} and n={n} we use {len(algos)} algorithms, namely {algos}")
      for xpi_ in range(numxps):
          target_matrix = np.zeros((n,n)) if not d3d() else np.zeros((n,n,n))
          for i in range(n):
             for j in range(int(n*(1+np.cos((i+xpi_)*2*3.14/n))/2)):
                 target_matrix[i][j] = 1.
          def testloss(binary_matrix):
             #return np.sum(binary_matrix) - 5 * np.sum(np.diag(binary_matrix))
           if pb == "simple":
             return np.sum((binary_matrix - target_matrix)**2)
           elif pb == "temp":
             walls = binary_matrix.copy()
             walls[0] = 0.
             walls[-1] = 0.
             nowalls = 1. - walls
             t  = 0 * walls
             for _ in range(77):
                 t = scipy.ndimage.gaussian_filter(t * nowalls, [1] * len(t.shape)) / (1e-3 + scipy.ndimage.gaussian_filter(nowalls, [1] * len(t.shape)))
                 t[0] = 0.
                 t[-1] = 1.
             r = t.flatten()
             score = 0.
             num_measures = 7
             for k in range(num_measures):
                 score += (r[np.random.RandomState(k).randint(len(r))] - np.random.RandomState(k+777).rand()) ** 2.
             assert score >= 0., f"score={score}"
             assert score <= num_measures, f"score={score},num_measures={num_measures}"
             return score
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
                   assert False, f"{xm} {xM} {ym} {yM} {n} : {pb}"
             #print(val)
             #assert False
             return val
           elif pb == "flatparts":
             return min([np.sum((binary_matrix - translate(target_matrix, i))**2) for i in range(n)])
           else:
             assert False, "unknown pb"
          #for k in algos:
           #try:
          def run_alg(k, ratio, c0):
            print(f"{k} works on {pb} with edge {n}")
            if d3d():
              parametrization = ng.p.Array(shape=(n,n,n),lower=0.,upper=1.)
              dparametrization = ng.p.Array(shape=(n,n,n),lower=0.,upper=1.).set_integer_casting()
            else:
              parametrization = ng.p.Array(shape=(n,n),lower=0.,upper=1.)
              dparametrization = ng.p.Array(shape=(n,n),lower=0.,upper=1.).set_integer_casting()
            #with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            def cv(x):
              return abs(c0) * np.sum((x - np.round(x))**2)  #).value#, executor=executor, batch_mode=False)
            if c0 > 0:  # Continuous
              optimizer = ng.optimizers.registry[k](parametrization=parametrization, budget=int(n*ratio*parametrization.dimension / 2))# num_workers=20)
              recommendationvalue = optimizer.minimize(testloss, constraint_violation=[cv]).value #, executor=executor, batch_mode=False)
              recommendationvalue = (recommendationvalue > 0.5).astype(recommendationvalue.dtype)
            elif c0 < 0:  # Continuous and discrete
              n1 = max(5, int(n*ratio/2)) * parametrization.dimension / 2
              n2 = max(5, int(n*ratio) - n1)

              optimizer2 = ng.optimizers.registry[k](parametrization=dparametrization, budget=n2)# num_workers=20)
              try:
               try:
                optimizer = ng.optimizers.registry["RBFGS"](parametrization=parametrization, budget=n1)# num_workers=20)
               except:
                optimizer = ng.optimizers.registry["BFGS"](parametrization=parametrization, budget=n1)# num_workers=20)
               recommendationvalue = optimizer.minimize(testloss, constraint_violation=[cv]).value #, executor=executor, batch_mode=False)
               recommendationvalue = (recommendationvalue > 0.5).astype(recommendationvalue.dtype)
               optimizer2.suggest(recommendationvalue)
              except:
               print(f"Relaxation failed for budget {n1} and algorithm {k} and c0 {c0}")
              recommendationvalue = optimizer2.minimize(testloss, constraint_violation=[cv]).value #, executor=executor, batch_mode=False)
            else:  # Purely discrete
              optimizer = ng.optimizers.registry[k](parametrization=dparametrization, budget=int(n*ratio))# num_workers=20)
              recommendationvalue = optimizer.minimize(testloss, constraint_violation=[cv]).value #, executor=executor, batch_mode=False)
            finalloss = testloss(recommendationvalue)
            #if not d3d():  # for graphical exports of solutions
            #    plt.clf()
            #    sol = recommendationvalue.copy()
            #    sol = sol - np.min(sol)
            #    sol = sol / np.max(sol)
            #    plt.imshow(sol, cmap='hot', interpolation='nearest')
            #    plt.savefig(f"sol_{pb}_{n}_{k}_{'3d' if d3d() else ''}_{finalloss:2.2}.png")
            return k + (f"C0({c0})" if c0 != 0. else ""), ratio, finalloss
          all_res = Parallel(n_jobs=80)(delayed(run_alg)(k, ratio, c0) for k in algos for ratio in list_ratios for c0 in [0., 0.01, 0.1, 1.0, 10.0, -10.0, 100.0] for _ in range(3) if not (("Discrete" in k and c0 > 0) or (c0 == 0.  and "BFGS" in k)))
          #all_res = Parallel(n_jobs=80)(delayed(run_alg)(k, ratio, c0) for k in algos for ratio in list_ratios for c0 in [0., 0.1, 1.0, 10.0, -0.1, -1.0, -10.0, 100.0, -100.0] for _ in range(3))
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
          sorted_algos = sorted(score[pb + str(ratio)][n].keys(), key=lambda k: np.average(score[pb + str(ratio)][n][k]))
          #if n > 10:
          #    algos = [a for a in algos if a in sorted_algos[:max(31, int(.7 * len(algos)))] ]
          print("Printing algorithms in stdout and output-file")
          for a in range(len(sorted_algos)):  #range(min(nbest, len(sorted_algos))):
              print(f"{a}/{nbest}: {sorted_algos[a]} for ratio {ratio} and pb {pb}")
          with open(f"output{'3d' if d3d() else ''}_{pb}_{n}_{ratio}_{np.random.randint(50000)}.txt", 'w') as output_file:
            for a in range(len(sorted_algos)):  #range(min(nbest, len(sorted_algos))):
              output_file.write(f"{a}/{len(sorted_algos)}: {sorted_algos[a]} gets {np.average(score[pb+ str(ratio)][n][sorted_algos[a]])}   ==== {score[pb+ str(ratio)][n][sorted_algos[a]]} for ratio {ratio} and pb {pb} and n={n}\n")
          
          sorted_algos = sorted_algos[:nbest] + ([sorted_algos[-1]] if nbest < len(sorted_algos) else [])
          sorted_algos = sorted_algos[::-1]
          allx = []
          ally = []
          print("Computing min/max")
          for i, k in enumerate(sorted_algos):
             print("loop", i, k, len(sorted_algos))
             x = [ni for ni in score[pb + str(ratio)].keys()]
             allx += x
             x_ = x[-1]
             for _ in range(37):
                ally += [np.average(score[pb + str(ratio)][x_][k]) + np.random.randn() * np.std(score[pb + str(ratio)][x_][k])]
          print("end of loop")
          xmax = np.max(allx)
          ymin = np.min(ally)
          ymax = np.max(ally)
          #sorted_algos = [sorted_algos[i] for i in range(len(sorted_algos)) if i <= 18 or i >= len(sorted_algos) - 2]
          for u in range(6):
           print(f"Graphic export {u}")
           plt.clf()
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
              kk = k.replace("Discrete", "D").replace("OnePlusOne", "opo").replace("Lengler", "Lng").replace("Langler", "Lng").replace("LogNormal", "LN").replace("Recombining", "Rec").replace("Elitist", "El").replace("Super", "Sup").replace("Smooth", "Sm").replace("Ultra", "Ult")
              if u % 3 == 0:
                plt.text(xmax, (1 - i/len(sorted_algos))*(ymax-ymin) + ymin, kk) #, {'rotation_mode': 'anchor', 'horizontalalignment': 'left', 'verticalalignment': 'center',})
              elif u % 3 == 1:
                plt.text(xmax, (i/len(sorted_algos))*(ymax-ymin) + ymin, kk) #, {'rotation_mode': 'anchor', 'horizontalalignment': 'left', 'verticalalignment': 'center',})
              else:
                plt.text(x[-1], y[-1], kk, {'rotation': min(r * (len(sorted_algos) - i - 1), 60), 'rotation_mode': 'anchor', 'horizontalalignment': 'left', 'verticalalignment': 'center',})
           plt.legend(loc=u,fontsize=f)
           plt.grid()
           if u > 3:
              plt.title(f"Lower = better, {pb}, budget={ratio}.edge" if d3d() else f"Lower = better, {pb}, budget={ratio}.edge")
           plt.ylabel("average loss")
           plt.xlabel("edge size of the binary matrix")
           plt.tight_layout()
           print("Saving!")
           if d3d():
              #plt.savefig(f"binmatrix{n}_loclegend{u}_fontsize{f}_problem3d{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.png")
              plt.savefig(f"binmatrixlast_loclegend{u}_fontsize{f}_problem3d{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.png")
              plt.savefig(f"binmatrixlast_loclegend{u}_fontsize{f}_problem3d{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.svg")
           else:
              #plt.savefig(f"binmatrix{n}_loclegend{u}_fontsize{f}_problem{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.png")
              plt.savefig(f"binmatrixlast_loclegend{u}_fontsize{f}_problem{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.png")
              plt.savefig(f"binmatrixlast_loclegend{u}_fontsize{f}_problem{pb}ratio{ratio}_plottingfc0{nbest}_labelrotation{r}.svg")
    #print(recommendation.value)
 
def go():   
  #Parallel(n_jobs=80)(delayed(doall)(pb) for pb in list_of_pbs)
  doall(list_of_pbs)
