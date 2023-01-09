import numpy as np
import scipy.io as scio
import scipy.optimize as optim
import algos
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '6'

import numpy.matlib
from labellines import labelLine, labelLines

from joblib import Parallel, delayed

gen = 1500  # Number of samples for estimating generalization
# Test de DE - sur un miroir de Bragg

def do_stuff(depop, n_couches, bud, broad=250.):
  import photonics
  photonics.broad = broad
  photonics.budget = bud
  X_min=np.hstack(20*np.ones(n_couches))
  X_max=np.hstack(180*np.ones(n_couches))
  pas=np.hstack(10*np.ones(n_couches))
  for meta_budget in [bud]: #[10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]:
   full_list = []
   def preplot(full_list, y, label, c, marker):
       full_list += [[y, label, c, marker]]
   def postplot(full_list):
    for bound in [7, 17, 27, 37, 47]:
     plt.clf()
     plt.yscale("log")
     def med(x):
         return x[len(x) // 2]
     #print(sorted([med(l[0]) for l in full_list]))
     medx = np.quantile([med(l[0]) for l in full_list], bound / len(full_list))
     full_list = sorted(full_list, key=lambda l: -l[0][len(l[0]) // 2])
     for l in full_list:
           if med(l[0]) < medx:
               #print(f"Go for {l}")
               plt.plot(l[0], label=l[1], c=l[2], marker=l[3])
           else:
               pass #print(f"Skips {l}")

     for l in plt.gca().get_lines():
       try:
         if len(l[0]) > 3:
             labelLines([l], align=False) #, zorder=2.5)
       except:
         pass#print(f"Fail with line {l}")
     try:
       labelLines(plt.gca().get_lines(), align=False) #, zorder=2.5)
     except:
       print("oh fuck")
     plt.legend(ncol=2) #bbox_to_anchor=(.1,1.1))
     plt.title(f'budget={meta_budget}')
     plt.tight_layout()  
     basename = f"broad{broad}reoutput_pop{depop}_{n_couches}layers_budget{meta_budget}_{bound}.png"
     plt.savefig(basename)
     with open(basename + ".txt", "w") as f:
      f.write(f"num algos: {len(full_list)}\n")
      for l in full_list:
          f.write(f"{l[1]}: {np.min(l[0])} {len(l[0])}\n")
   discs = [i for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if i < meta_budget // 2]
   np.random.shuffle(discs)
   for disc in discs:
    randos = np.linspace(-1., 1., 6)
    np.random.shuffle(randos)
    for rando in randos:
     ngopt_results = {}
     ngopt_values={}
     list_algs = ["CMA", "DiagonalCMA", "DE", "Cobyla", "NGOptRW", "NoisyRL1", "NoisyRL2", "NoisyRL3"]
     np.random.shuffle(list_algs)
     for alg in list_algs:  # LaMCTS, SMAC, SMAC2  ;   AX too weak for anything meaningful  
      if alg not in ngopt_results:
           ngopt_results[alg] = []
           ngopt_values[alg] = []
     de_results = []
     descent_results = []
     bfgs_results = []
     chain_results = []
     ngchain_results = []
     de_values=[]
     steep_values=[]
     bfgs_values=[]
     chain_values=[]
     ngchain_values=[]
     budget = meta_budget // disc
     photonics.default_num = disc
     photonics.default_rando = rando
      #for alg in ["AX", "NGOptRW", "DiagonalCMA", "DE", "Cobyla"]:  # LaMCTS, SMAC, SMAC2
     for k in range(29):
       for alg in list_algs:
        photonics.counter = 0
        if alg in ["AX"] and (budget > 100 or n_couches > 50):
          continue
        [ngopt_best,ngopt_convergence,recom] =  algos.ngopt(photonics.bragg_fixed,budget,X_min,X_max,depop,alg)
        ngopt_results[alg].append([ngopt_best,ngopt_convergence])
        #ngopt_values[alg].append(ngopt_convergence[-1])
        ngopt_values[alg].append(photonics.bragg_fixed(recom, gen, True, False))
   
       photonics.counter = 0
       [de_best,de_convergence,recom] =  algos.DEvol(photonics.bragg_fixed,budget,X_min,X_max,depop)
       de_results.append([de_best,de_convergence])
       de_values.append(photonics.bragg_fixed(recom, gen, True,False)) #de_convergence[-1])
       xmin=np.hstack(20*np.ones(n_couches))
       xmax=np.hstack(180*np.ones(n_couches))
       start=xmin+np.random.random_sample(n_couches)*(xmax-xmin)
   
       photonics.counter = 0
       [x_desc,convergence,recom]=algos.descente(photonics.bragg_fixed,pas,start,budget,xmin,xmax)
       descent_results.append([x_desc,convergence])
       steep_values.append(photonics.bragg_fixed(recom, gen, True, False))  #convergence[-1])
   
       photonics.counter = 0
       [x_desc,convergence,recom]=algos.descente2(photonics.bragg_fixed,pas,start,budget,xmin,xmax)
       bfgs_results.append([x_desc,convergence])
       bfgs_values.append(photonics.bragg_fixed(recom, gen, True, False))  #convergence[-1])
   
       photonics.counter = 0
       [de_best,de_convergence,recom] =  algos.DEvol(photonics.bragg_fixed,budget // 2,X_min,X_max,depop)
       [x_desc,convergence,recom]=algos.descente2(photonics.bragg_fixed,pas,recom,budget - (budget // 2),xmin,xmax)
       chain_results.append([x_desc, de_convergence + convergence])
       chain_values.append(photonics.bragg_fixed(recom, gen, True, False)) #convergence[-1])
   
       photonics.counter = 0
       alg = "NGOptRW"
       [ngopt_best,ngopt_convergence,second_start] =  algos.ngopt(photonics.bragg_fixed,budget // 2,X_min,X_max,depop,alg)
       #[de_best,de_convergence] =  algos.DEvol(photonics.bragg_fixed,budget // 2,X_min,X_max,depop)
       [x_desc,convergence,recom]=algos.descente2(photonics.bragg_fixed,pas,second_start,budget - (budget // 2),xmin,xmax)
       ngchain_results.append([x_desc, ngopt_convergence + convergence])
       ngchain_values.append(photonics.bragg_fixed(recom, gen, True, False)) #convergence[-1])
   
     de_sorted = np.sort(de_values)
     steep_sorted = np.sort(steep_values)
     bfgs_sorted = np.sort(bfgs_values)
     chain_sorted = np.sort(chain_values)
     ngchain_sorted = np.sort(chain_values)
     def random_color():
         #return (np.random.rand(), np.random.rand(), np.random.rand())
         return np.random.choice(['b', 'g', 'r', 'c', 'm', 'k'])
     def random_marker():
         return np.random.choice(list(range(12)))
     preplot(full_list,de_sorted, label='DE'+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
     for a in ngopt_values:
       ngopt_sorted = np.sort(ngopt_values[a])
     preplot(full_list,ngopt_sorted, label='Ng'+a+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
     preplot(full_list,steep_sorted, label='GD'+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
     preplot(full_list,bfgs_sorted, label='BFGS'+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
     preplot(full_list,chain_sorted, label='Chain'+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
     preplot(full_list,ngchain_sorted, label='NgChain'+str(disc)+(f"_{round(rando*100)/100}" if abs(rando) > 1e-3 else ""), c=random_color(),marker=random_marker())
   postplot(full_list)
    
#for depop in [10, 20, 30, 40]:
# for n_couches in [10, 20, 40, 80]:
#  do_stuff(depop, n_couches)
#Parallel(n_jobs=60)(delayed(do_stuff)(depop, n_couches, bud) for n_couches in [60, 80] for depop in [30] for bud in [10, 30, 100, 300, 1000, 3000, 10000])
#Parallel(n_jobs=60)(delayed(do_stuff)(depop, n_couches, bud) for n_couches in [10, 20, 30, 40, 60, 80, 100, 120, 140] for depop in [30] for bud in [100, 300, 1000, 3000, 10000])
import os
broad=250.
if "BROAD" in os.environ:
   broad=float(os.environ["BROAD"])
Parallel(n_jobs=60)(delayed(do_stuff)(depop, n_couches, bud, broad=broad) for n_couches in [10, 20, 30] for depop in [30] for bud in [100, 300, 1000, 3000, 10000, 30000])#, 100000, 300000, 1000000, 3000000])


#reference = np.array([3.,2.]*10 + [600/(4*np.sqrt(3)),600/(4*np.sqrt(2))]*10)
#reference = np.hstack((np.matlib.repmat([3,2],1,int(n/2)),np.matlib.repmat([600/(4*np.sqrt(3)),600/(4*np.sqrt(2))],1,int(n/2))))
#photonics.bragg(reference)
