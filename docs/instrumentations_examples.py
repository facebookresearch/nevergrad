import numpy as np
from collections import defaultdict
import nevergrad as ng

# Function for creating timeouts.
import signal


def timeout_handler(signum, frame):
    raise Exception("end of time")


# We work in NxN:
for N in [5, 10, 20, 40, 80]:
    # Helper function.
    def testing_domain_and_loss_and_budget(d, l, b):
        score = defaultdict(list)
        for _ in range(560):
            # Let us test several algorithms.
            optim_name = sorted(
                list(ng.optimizers.registry.keys()),
                key=lambda o: np.sum(score[o]) / (len(score[o]) + 0.01)
                + (1 + np.sqrt(1 + np.log(1 + len(score[o])))) / np.sqrt(len(score[0]) + 0.001),
            )[0]
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # seconds at most
            try:
                recommendation = ng.optimizers.registry[optim_name](domain, b).minimize(l)
                try:
                    loss_value = l(recommendation.value)
                except:  # For complex domains!
                    loss_value = l(*recommendation.args, **recommendation.kwargs)
                print(f"Algorithm {optim_name} got {loss_value}")
            except Exception as e:
                loss_value = float(1e10)
                print(f"Algorithm {optim_name} crashed: {e}!")
            signal.alarm(0)
            score[optim_name] += [loss_value]

        print(f"List of best for N={N} and budget={b}:")
        for u in sorted(score, key=lambda x: np.sum(score[x]) / len(score[x])):
            print(u, np.sum(score[u]) / len(score[u]), len(score[u]))

    # Let us create a target of shape NxN.
    target = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            if abs(i - j) < N // 3:
                target[i][j] = 1.0

    # Case "continuous multidimensional array": I want to optimize a NxN array of real numbers.
    print("First, we optimize a continuous function on a continuous domain")
    domain = ng.p.Array(shape=(N, N))

    def loss(x):
        return np.sum(np.abs(x - target))

    print(ng.optimizers.registry["NGOpt"](domain, 10 * N * N).minimize(loss).value)
    testing_domain_and_loss_and_budget(domain, loss, 5 * N * N)

    # Case "discrete multidimensional array": I want to optimize a NxN array of bool.
    print("Now, we optimize on a discrete domain.")
    domain = ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting()
    print(ng.optimizers.registry["DiscreteLenglerOnePlusOne"](domain, 10 * N * N).minimize(loss).value)
    testing_domain_and_loss_and_budget(domain, loss, 5 * N * N)

    # Case "combination":
    print("Now, let us work on a mixed domain continuous/discrete.")
    domain = ng.p.Instrumentation(
        x=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        y=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
    )

    def complex_loss(x, y):
        return loss(x) + loss(np.transpose(y))

    print(ng.optimizers.registry["NGOpt"](domain, 10 * N * N).minimize(complex_loss).value)
    testing_domain_and_loss_and_budget(domain, complex_loss, 5 * N * N)
