import numpy as np
import nevergrad as ng
from joblib import Parallel, delayed
from glob import glob


def deal_with_method(method):
 for idx in range(10000):
  if idx > 3:
      method = np.random.choice(list(ng.optimizers.registry.keys()))
  files = glob("examples/francois_data/*.txt")
  this_file = np.random.choice(files)
  print("We work on ", this_file)
  with open(this_file) as f:
      d, n, discrepancy_lower_bound  = [float(x) for x in next(f).split()] # read first line
      array = []
      for line in f: # read rest of lines
          array.append([float(x) for x in line.split()])
  #d = np.random.choice([2, 5, 12, 24, 48, 96])
  #n = np.random.choice([10, 100, 1000])
  d = int(d)
  n = int(n)
  assert len(array) == n
  assert len(array[0]) == d
  b = np.random.choice([1000, 10000, 100000, 1000000, 10000000])
  #L = np.random.RandomState(d*37+n*101+b*12).rand(n,d)
  L = np.array(array)

  def localdisc(x):
      points_in_interior=0
      points_in=0
      volume = 1
      for i in range (0,n):
          if all(L[i] < x):
              points_in_interior += 1
          if all(L[i] <= x):
              points_in += 1
      for i in range (0,len(x)):
          volume=volume*x[i]
      disc=max(volume-points_in_interior/n, points_in/n-volume)
      return(1. - disc)
    
  instrum = ng.p.Array(shape=(int(d),), lower=0., upper=1.)
  try:
      optim = ng.optimizers.registry[method](instrum, int(b*d))
      recom = optim.minimize(localdisc)
      val = localdisc(recom.value)
      assert len(recom.value) == d
      print(f"We get {1-val} for n={n} and d={d} and b/d={b}, file={this_file} and disclowerbound={discrepancy_lower_bound}, our result is {(1-val) / discrepancy_lower_bound}")
      print(f"Recom = {recom.value}")
      print(f"RESLD_{d}_{n}_{b}_{method}  gets {val}")
  except Exception as e:
      print(f"Failure {e} for {method}")


num=70
#list_of_methods = [np.random.choice(list(ng.optimizers.registry.keys()))] * num
list_of_methods = list(np.random.choice(list(ng.optimizers.registry.keys()), num))
for i in range(4):
    list_of_methods[i] = "NGOptF"
for i in range(4, 8):
    list_of_methods[i] = "NGOptF2"
for i in range(8, 14):
    list_of_methods[i] = "NGOptF3"
for i in range(15, 21):
    list_of_methods[i] = "NGOptF5"
list_of_methods[22] = "FSQPCMA"
list_of_methods[23] = "F2SQPCMA"
list_of_methods[24] = "F3SQPCMA"
print("    RESULT", list_of_methods)
results = Parallel(n_jobs=num)(delayed(deal_with_method)(method) for method in list_of_methods)


