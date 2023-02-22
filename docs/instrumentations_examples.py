import numpy as np
import nevergrad as ng

# We work in NxN:
N = 20

# Let us create a target of shape NxN.
target = np.zeros(shape=(N, N))
for i in range(N):
    for j in range(N):
        if abs(i-j) < N // 3:
            target[i][j] = 1.


# Case "continuous multidimensional array": I want to optimize a NxN array of real numbers.
print("First, we optimize a continuous function on a continuous domain")
domain = ng.p.Array(shape=(N,N))
def loss(x):
    return np.sum(np.abs(x - target))
print(ng.optimizers.registry["NGOpt"](domain, 10*N*N).minimize(loss).value)

# Case "discrete multidimensional array": I want to optimize a NxN array of bool.
print("Now, we optimize on a discrete domain.")
domain = ng.p.Array(shape=(N,N), upper=1., lower=0.).set_integer_casting()
print(ng.optimizers.registry["DiscreteLenglerOnePlusOne"](domain, 10*N*N).minimize(loss).value)

# Case "combination":
print("Now, let us work on a mixed domain continuous/discrete.")
domain = ng.p.Instrumentation(
     x = ng.p.Array(shape=(N,N), upper=1., lower=0.),
     y = ng.p.Array(shape=(N,N), upper=1., lower=0.).set_integer_casting(),
  )
def complex_loss(x, y):
    return loss(x) + loss(np.transpose(y))
print(ng.optimizers.registry["NGOpt"](domain, 10*N*N).minimize(complex_loss).value)

for _ in range(30):
    # Let us test several algorithms.
    optim_name = np.random.choice(list(ng.optimizers.registry.keys()))
    recommendation = ng.optimizers.registry[optim_name](domain, 10*N*N).minimize(complex_loss)
    print(f"Algorithm {optim_name} got {complex_loss(*recommendation.args, **recommendation.kwargs)}")





