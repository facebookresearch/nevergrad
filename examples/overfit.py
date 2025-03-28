from pathlib import Path
import sys
#path_root = Path(__file__).parents[2]
#sys.path.append(str(path_root))
sys.path = ["."] + sys.path

import numpy as np
import nevergrad as ng
print(ng.__file__)
import time
import signal
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib

resampling = 27

matplotlib.rcParams.update({'figure.autolayout': True})


label_type = 2

# no more than max_algos algorithms.
max_algos = int(len(list(ng.optimizers.registry.keys())))

def handler(signum, frame):
    return Exception("too slow")


def perf(algo, d=5, n=50, resample=resampling, nt=1000, budget=20):
  try:
    score = []
    d = int(d)
    n = int(n)
    nt = int(nt)
    budget = int(budget)
    overfits = []
    d2 = d*(3 if label_type == 2 else 1)
    for idx in range(resample):
        np.random.seed(idx)
        x = np.random.randn(n,d)  # training set
        #xt = np.random.randn(nt,d)  # test set
        t = np.random.randn(d2)
        y = np.random.randn(n)
        #yt = np.random.randn(nt)
        def label(x, t):
            assert len(x) == d
            if label_type == 2:
                for k in range(3):
                    #print(k, x.shape, t[(k*d):(k*d+d)].shape, len(t), flush=True)
                    v = np.tanh(x * t[(k*d):(k*d+d)])
                return np.linalg.norm(v**2)
            if label_type == 1:
                return np.linalg.norm(x-t)**2 + np.random.randn()
            assert False
        for i in range(len(x)):
            y[i] = label(x[i], t)
        #for i in range(len(xt)):
        #    yt[i] = label(xt[i], t)
        def loss(p):
            pred = np.random.randn(n)
            for i in range(len(x)):
                pred[i] = label(x[i], p)
            return np.sum((y - pred) ** 2) / len(y)
        def testloss(p):
            pred = np.random.randn(nt)
            sco = 0
            for i in range(nt):
                inp = np.random.randn(d)
                sco += (label(inp, p) + label(inp, t))**2
            return sco / nt
        reco = ng.optimizers.registry[algo](d2, budget).minimize(loss).value
        tl = testloss(reco)
        l = loss(reco)
        score = score + [tl]
        overfits = overfits + [tl - l]
    return np.sum(score) / len(score), np.sum(overfits) / len(overfits)
  except Exception as e:
    print(e, flush=True)
    return float("inf"), float("inf")

score = {}
overfit = {}
explain = {}
score_per_budget = {}
overfit_per_budget = {}


def display_losses(data, explain):
    result = ""
    for i, a in enumerate(sorted(list(data.keys()), key = lambda a: data[a])):
        #if i == 10 or i == len(data) - 10:
        #    result += "... ... ... \n"
        #if i > 10 and i < len(data) - 10 and (i % 20 > 0):
        #    continue
        result += f"{a}, {data[a]}, {i} / {len(data)} ===== {explain[a]}\n"
    return result




def subgo(a,d,n,budget,stop_on_error=False):
    a_full = a +  "__" + str(budget)
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(600)
        assert not "AX" in a  # too slow
        print("Working with ", a, flush=True)
        s, o = perf(a, d=d, n=n, budget=budget)
        print("Finished working with ", a, flush=True)
        e = ""
        signal.alarm(0)
        return a_full,s,o, str(e), a, budget
    except Exception as e:
        print("CRASH", a,d,n,budget,stop_on_error,str(e),flush=True)
        assert not stop_on_error, str(e)
        s = float("inf")
        o = float("inf")
        return a_full,s,o, str(e), a, budget


def draw(data, filename, context):
  for n in [3, 7, 15]:
    plt.clf()
    algorithms = list(data.keys())
    num = len(algorithms) // n
    sortedalgorithms = sorted(algorithms, key=lambda a: -min(data[a].values()))
    mean_algorithm = sortedalgorithms[len(algorithms) // 40]
    roof = 1.3 * min(data[mean_algorithm].values())
    for i, a in enumerate(sortedalgorithms):
        if i < len(algorithms) - n and algorithms[i] != "Zero" and algorithms[i] != "StupidRandom":  #and i < len(algorithms) - n and (i % num != 0):
        #if i > n and i < len(algorithms) - n and (i % num != 0):
            continue
        x = sorted(list(data[a].keys()))
        y = [data[a][x_] for x_ in x]
        yroof = [min(data[a][x_], roof) for x_ in x]
        plt.plot(x, yroof, label=f"{a} {min(y):.2f} ({len(algorithms)-i}/{len(algorithms)})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', prop={'size': 2 + (50 // n)})
    plt.title(context)
    plt.grid()
    plt.tight_layout()
    plt.savefig(str(n) +"___" + filename, bbox_inches='tight')



def go(d,n,budget,max_algos=max_algos, stop_on_error=False):
    context = f"Dim {d}, Dataset size {n}, Budget {budget}"
    # optims = list(ng.optimizers.registry.keys())[:max_algos]
    optims = [x for x in list(ng.optimizers.registry.keys()) if len(x.replace("OnePlusOne","1+1")) < 10 and x not in ["NgIoh" + str(i) for i in range(50)]]
    #optims = [o for o in optims if "PRAXIS" not in o]
    np.random.shuffle(optims)
    nj = 15 #80 
    for realbudget in [budget // 4, budget // 2, budget, 2*budget, 4*budget]:
        res = Parallel(n_jobs=nj)(delayed(subgo)(a,d, n, realbudget,stop_on_error=stop_on_error) for a in optims)
        for i in range(len(res)):
            a_full, s, o, e, a, b = res[i]
            score[a_full] = s
            overfit[a_full] = o
            explain[a_full] = e
            if a not in score_per_budget:
                score_per_budget[a] = {}
                overfit_per_budget[a] = {}
            score_per_budget[a][b] = s
            overfit_per_budget[a][b] = o
    filename = f"perf{label_type}_d{d}_n{n}_budget{budget}______rs{resampling}_{np.random.randint(200)}_{nj}.txt"
    with open(filename, "w") as f:
        res = display_losses(score,explain)
        f.write(res)
    draw(score_per_budget, filename + ".png", context)
    filename = f"overfit{label_type}_d{d}_n{n}_budget{budget}______rs{resampling}_{np.random.randint(200)}_{nj}.txt"
    with open(filename, "w") as f:
        res2 = display_losses(overfit,explain)
        f.write(res2)
    draw(overfit_per_budget, filename + ".png", context)
    return (res, res2) #"perf ===========================================================\n" + res + "overfit =================================================================\n" + res2
    #    print("===============================================================")
    #    print("============= Performances ======================")
    #    display_losses(score)
    #    print("============= Overfittings ======================")
    #    display_losses(overfit)
    
                
    
#go(10,10,10,max_algos=3,stop_on_error=True)
#for d_ in [10, 100, 10000, 1000000]:
#    for n_ in [10, 100, 10000]:
#        for b_ in [10, 100, 1000, 100000]:
            #d += [d_]
            #n += [n_]
            #budget += [b_]
d_ = np.random.choice([3, 10, 100, 1000, 10000]) #, 1000000])
n_ = np.random.choice([3, 10, 100, 1000])
b_ = np.random.choice([3, 10, 100, 1000])  #, 100000])
#d_ = 3
#b_ = 3
#n_ = 3
go(d_, n_, b_)
#Parallel(n_jobs=70)(delayed(go)(d[i], n[i], budget[i]) for i in range(len(n)))   
