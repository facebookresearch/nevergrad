import numpy as np
import nevergrad as ng
import time
import signal
from joblib import Parallel, delayed

def handler(signum, frame):
    return Exception("too slow")


def perf(algo, d=5, n=50, resample=10, nt=10000, budget=20):
    score = []
    overfits = []
    for _ in range(resample):
        x = np.random.randn(n,d)  # training set
        xt = np.random.randn(nt,d)  # test set
        t = np.random.randn(d)  # target
        y = np.random.randn(n)
        yt = np.random.randn(nt)
        def label(x, t):
            return np.linalg.norm(x-t)**2 + np.random.randn()
        for i in range(len(x)):
            y[i] = label(x[i], t)
        for i in range(len(xt)):
            yt[i] = label(xt[i], t)
        def loss(p):
            pred = np.random.randn(n)
            for i in range(len(x)):
                pred[i] = label(x[i], p)
            return np.sum((y - pred) ** 2)
        def testloss(p):
            pred = np.random.randn(nt)
            for i in range(len(xt)):
                pred[i] = label(xt[i], p)
            return np.sum((yt - pred) ** 2)
        reco = ng.optimizers.registry[algo](d, budget).minimize(loss).value
        tl = testloss(reco)
        l = loss(reco)
        score = score + [tl]
        overfits = overfits + [tl - l]
    return np.sum(score) / len(score), np.sum(overfits) / len(overfits)

score = {}
overfit = {}
explain = {}


def display_losses(data):
    result = ""
    for i, a in enumerate(sorted(list(data.keys()), key = lambda a: data[a])):
        result += f"{a}, {data[a]}, {i} / {len(data)}\n"
        if i > 10 and i < len(data) - 10:
            continue
    return result

def go(d,n,budget):
    for a in list(ng.optimizers.registry.keys())[:10]:
        if "AX" in a:
            continue
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(7)
            score[a], overfit[a] = perf(a, d=d, n=n, budget=budget)
            signal.alarm(0)
        except Exception as e:
            score[a + "__" + str(budget)] = float("inf")
            explain[a + "__" + str(budget)] = e
    filename = f"perf_d{d}_n{n}_budget{budget}.txt"
    with open(filename, "w") as f:
        f.write(display_losses(score))
    filename = f"overfit_d{d}_n{n}_budget{budget}.txt"
    with open(filename, "w") as f:
        f.write(display_losses(overfit))
    #    print("===============================================================")
    #    print("============= Performances ======================")
    #    display_losses(score)
    #    print("============= Overfittings ======================")
    #    display_losses(overfit)
    
                
    
d = []
n = []
budget = []
for d_ in [10, 100, 10000, 1000000]:
    for n_ in [10, 100, 10000, 1000000]:
        for b_ in [10, 100, 1000, 1000000]:
            d += [d_]
            n += [n_]
            budget += [b_]
Parallel(n_jobs=10)(delayed(go)(d[i], n[i], budget[i]) for i in range(len(n)))   
