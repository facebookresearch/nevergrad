import sys
sys.path.append("./")
import numpy as np
import nevergrad as ng
import copy


# Solving Nash equilibbria in the antagonist case when the number of pure policies is moderate relatively to the
# dimension.
#
# PKL approach.
# Exploitation optimization approach
# Fictitious play 
# Randomized fictitious play

s = np.random.randint(32)

# Creating a problem
N = int(np.random.RandomState(4*s).choice([6, 15, 27, 48]) )
lambd = int(50 * np.log(N))  # Size of the Nash approximation; lambd is the number of pure policies in our mixed policies
A = np.random.RandomState(4*s + 1).rand(*(N,N))
xopt = np.random.RandomState(4*s+2).rand(N) > .5
yopt = np.random.RandomState(4*s+3).rand(N) > .5



print(f"Specifications: seed{s}, dim={N}, pop={lambd}")
b = np.matmul(A, yopt)

c = np.matmul(xopt, np.transpose(A))

num_calls = 0


def f(x, y):
    global num_calls
    num_calls += 1
    assert len(x) == N and len(x.shape) == 1
    return np.matmul(np.matmul(x, A), y) - np.matmul(b, x) - np.matmul(c, y)

bestval = 0
def exploitation(P, Q, exploitation_budget = 500, algo="DiscreteLenglerOnePlusOne", needargmax=False):
    assert len(P) == lambd
    assert len(Q) == lambd
    assert len(P[0]) == N
    assert len(Q[0]) == N
    optim = ng.optimizers.registry[algo](ng.p.Array(shape=(N,), lower=0., upper=1.).set_integer_casting(), exploitation_budget)   
    f0 = 0
    for i in range(lambd):
        for j in range(lambd):
            f0 += f(P[i], Q[j])
    f0 = f0 / (lambd * lambd)
    global bestval
    bestval = float("inf")
    def maximize(x):
        global bestval
        val = np.sum( [f0 - f(x, Q[i]) for i in range(lambd)] )
        if val < bestval:
            bestval = val
        return val
    optim.minimize(maximize)
    bestx = optim.recommend().value
    loss = bestval  #maximize(optim.minimize(maximize).value)
    optim = ng.optimizers.registry[algo](ng.p.Array(shape=(N,), lower=0., upper=1.).set_integer_casting(), exploitation_budget)   

    bestval = float("inf")
    def minimize(y):
        global bestval
        val = np.sum( [f(P[i], y) - f0 for i in range(lambd)] )
        if val < bestval:
            bestval = val
        return val
    optim.minimize(minimize)
    besty = optim.recommend().value
    loss += bestval #minimize(optim.minimize(minimize).value)
    if needargmax:
        return -loss, bestx, besty
    else:
        return -loss
    

ng_orig_optims = list(ng.optimizers.registry.keys())

ng_optims = (["Pkl"] * 8) +["RS"]

ng_optims += ["explo" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation
ng_optims +=  ["explo" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["explo" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation

ng_optims += ["explO" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["explO" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["explO" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation

ng_optims += ["fip__" + o for o in ng_orig_optims if ("ois" not in o) and "iscr" in o]  # Fictitious play
ng_optims += ["fip__" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["fip__" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation

ng_optims += ["fipl_" + o for o in ng_orig_optims if ("ois" not in o) and "iscr" in o]  # Fictitious play
ng_optims += ["fipl_" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["fipl_" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation

ng_optims += ["fpll_" + o for o in ng_orig_optims if ("ois" not in o) and "iscr" in o]  # Fictitious play
ng_optims += ["fpll_" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["fpll_" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation

ng_optims += ["ficpl" + o for o in ng_orig_optims if "SQP" in o  or "MetaModel" in o]  # Direct optimization of exploitation
ng_optims += ["ficpl" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation
ng_optims += ["ficpl" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["ficpl" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["ficpl" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["ficpl" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration

ng_optims += ["exlog" + o for o in ng_orig_optims if "SQP" in o  or "MetaModel" in o]  # Direct optimization of exploitation
ng_optims += ["exlog" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation
ng_optims += ["exlog" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["exlog" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exlog" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exlog" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration

ng_optims += ["Pkl"] * 80

ng_optims += ["exLOG" + o for o in ng_orig_optims if "SQP" in o  or "MetaModel" in o]  # Direct optimization of exploitation
ng_optims += ["exLOG" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation
ng_optims += ["exLOG" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["exLOG" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exLOG" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exLOG" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration

ng_optims += ["exLOS" + o for o in ng_orig_optims if "SQP" in o  or "MetaModel" in o]  # Direct optimization of exploitation
ng_optims += ["exLOS" + o for o in ng_orig_optims if "XLog" in o ]  # Direct optimization of exploitation
ng_optims += ["exLOS" + o for o in ng_orig_optims if "ptimis" in o ]  # Direct optimization of exploitation
ng_optims += ["exLOS" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exLOS" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
ng_optims += ["exLOS" + o for o in ng_orig_optims if "ois" in o and "iscr" in o]  # Direct optimization of exploitation, with more evals per iteration
#print(ng_optims)


#print(ng_optims)
#print([o for o in ng_optims if "engler" in o])
ng_optims = [o for o in ng_optims if ("Lengler" in o and "ecomb" not in o and "Smooth" not in o) or "Pkl" in o or "Portfolio" in o or "Pkl" in o]
algo = np.random.choice(ng_optims)

print(algo)
#algo = "RS"
#print("We work with ", algo)

factor = 1

if algo[:4] in ["expl", "exlo", "exLO"]:
    budget = 50 * np.random.choice([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240])
    budget *= factor
    optim = ng.optimizers.registry[algo[5:]](ng.p.Array(shape=(2, lambd, N), lower=0., upper=1.).set_integer_casting(), budget)
    num = 0 if algo[:4] == "expl"  else 100
    if algo[:5] == "exLOS":
        num = 0
    def expl(r):
        p = r[0]
        q = r[1]
        global num
        num = num + 1
        if algo[:4] == "exLO":
            return exploitation(p, q, 1+int(np.log(num)))
        return exploitation(p, q, num)
    r = optim.minimize(expl).value
    P = r[0]
    Q = r[1]
elif algo == "RS":
    P = np.random.rand(lambd, N) > .5
    Q = np.random.rand(lambd, N) > .5
    
elif algo[:5] == "fip__":
    P = np.random.rand(lambd, N) > .5
    Q = np.random.rand(lambd, N) > .5
    num = 0
    budget = np.random.choice([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680])
    budget *= factor
    iteration = 0
    for i in range(budget):
        iteration = iteration + 1
        num = num + 1
        l, bestx, besty = exploitation(P, Q, num, algo[5:], needargmax=True)

        P[iteration % lambd] = bestx
        Q[iteration % lambd] = besty
        
elif algo[:5] == "ficpl":
    P = np.random.rand(lambd, N) > .5
    Q = np.random.rand(lambd, N) > .5
    P2 = np.random.rand(lambd, N) > .5
    Q2 = np.random.rand(lambd, N) > .5
    num = 100
    budget = np.random.choice([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680])
    budget *= factor
    iteration = 0
    for i in range(budget // lambd):
      for j in range(lambd):
        iteration = iteration + 1
        num = num + 1
        l, bestx, besty = exploitation(P, Q, num, needargmax=True)  # if algo[:5] == "fipl_" else (1+int(np.log(num))), algo[5:], needargmax=True)

        P2[j] = bestx
        Q2[j] = besty
      P = copy.deepcopy(P2)
      Q = copy.deepcopy(Q2)
        
elif algo[:5] in ["fipl_", "fpll_"]:
    P = np.random.rand(lambd, N) > .5
    Q = np.random.rand(lambd, N) > .5
    num = 100
    budget = np.random.choice([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680])
    budget *= factor
    iteration = 0
    for i in range(budget):
        iteration = iteration + 1
        num = num + 1
        l, bestx, besty = exploitation(P, Q, num if algo[:5] == "fipl_" else (1+int(np.log(num))), algo[5:], needargmax=True)

        P[iteration % lambd] = bestx
        Q[iteration % lambd] = besty
        
elif algo == "Pkl":
    def dom(x1, y1, x2, y2):
        left = f(x1, y2)
        middle = f(x1, y1)
        right = f(x2, y1)
        return (left >= middle) and (middle >= right)
    kappa = 0.3/N   # < ln(2) / N
    
    P = np.random.rand(lambd, N) > .5
    Q = np.random.rand(lambd, N) > .5
    P2 = np.random.rand(lambd, N) > .5
    Q2 = np.random.rand(lambd, N) > .5
    
    nbiter = np.random.choice([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1320000])
    nbiter*= 10
    for t in range(nbiter):
    
        for i in range(lambd):
            x1 = P[np.random.randint(lambd)]
            x2 = P[np.random.randint(lambd)]
            y1 = Q[np.random.randint(lambd)]
            y2 = Q[np.random.randint(lambd)]
            if dom(x1,y1,x2,y2):
                x = x1
                y = y1
            else:
                x = x2
                y = y2
            P2[i] = x
            Q2[i] = y
            for k in range(N):
                if np.random.rand() < kappa / N:
                    P2[i][k] = 1 - P2[i][k]
                if np.random.rand() < kappa / N:
                    Q2[i][k] = 1 - Q2[i][k]
        P = copy.deepcopy(P2)
        Q = copy.deepcopy(Q2)




# P, Q should be an approximate Nash

my_num_calls = num_calls

ex = exploitation(P,Q)

print(f"Algo{algo}_budget{my_num_calls}_loss{ex}_seed{s}__result")
#randex = [exploitation(np.random.rand(lambd, N) > .5, np.random.rand(lambd, N) > .5) for i in range(100)]
#print(randex)
#print("Frequency of random better than optimized:", np.average([ex > rex for rex in randex]))
