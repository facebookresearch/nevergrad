import sys
sys.path.append("./")
import numpy as np
import nevergrad as ng

N = 6

A = np.random.rand(*(N,N))
xopt = np.random.rand(N) > .5
yopt = np.random.rand(N) > .5

b = np.matmul(A, yopt)

c = np.matmul(xopt, np.transpose(A))

def f(x, y):
    assert len(x) == N and len(x.shape) == 1
    return np.matmul(np.matmul(x, A), y) - np.matmul(b, x) - np.matmul(c, y)

def dom(x1, y1, x2, y2):
    left = f(x1, y2)
    middle = f(x1, y1)
    right = f(x2, y1)
    return (left >= middle) and (middle >= right)

lambd = int(50 * np.log(N))
kappa = 0.3/N   # < ln(2) / N

P = np.random.rand(lambd, N) > .5
Q = np.random.rand(lambd, N) > .5

nbiter = 50

for t in range(nbiter):

    for i in range(lambd):
        x1 = P[np.random.randint(N)]
        x2 = P[np.random.randint(N)]
        y1 = Q[np.random.randint(N)]
        y2 = Q[np.random.randint(N)]
        if dom(x1,y1,x2,y2):
            x = x1
            y = y1
        else:
            x = x2
            y = y2
        P[i] = x
        Q[i] = y
        for k in range(N):
            if np.random.rand() < kappa / N:
                P[i][k] = 1 - P[i][k]
            if np.random.rand() < kappa / N:
                Q[i][k] = 1 - Q[i][k]


exploitation_budget = 500
def exploitation(P, Q):
    optim = ng.optimizers.DiscreteLenglerOnePlusOne(ng.p.Array(shape=(N,), lower=0., upper=1.).set_integer_casting(), exploitation_budget)   
    f0 = 0
    for i in range(lambd):
        for j in range(lambd):
            f0 += f(P[i], Q[i])
    f0 = f0 / (lambd * lambd)
    def maximize(x):
        return np.sum( [f0 - f(x, Q[i]) for i in range(lambd)] )
    loss = maximize(optim.minimize(maximize).value)
    optim = ng.optimizers.DiscreteLenglerOnePlusOne(ng.p.Array(shape=(N,), lower=0., upper=1.).set_integer_casting(), exploitation_budget)   
    def minimize(x):
        return np.sum( [f(P[i], y) - f0 for i in range(lambd)] )
    loss += minimize(optim.minimize(minimize).value)
    return -loss
    


# P, Q should be an approximate Nash
randex = [exploitation(np.random.rand(lambd, N) > .5, np.random.rand(lambd, N) > .5) for i in range(100)]

ex = exploitation(P,Q)

print(ex)
print(randex)
print("Frequency of random better than optimized:", np.average([ex > rex for rex in randex]))
