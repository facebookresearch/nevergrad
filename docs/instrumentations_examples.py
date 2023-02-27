import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
import nevergrad as ng

# Function for creating timeouts.
import signal


def mean(v):
    s = np.sum(v)
    m = s / len(v)
    if not s == s or len(v) == 0:
        m = float(0.0)
    return m


def ucb_score(v, i):
    m = mean(v)
    stat = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m - stat


def lcb_score(v, i=1.0):
    m = mean(v)
    stat = (1 + np.sqrt(np.log(3.0 + i))) / np.sqrt(len(v))
    return m + 0.1 * stat


def timeout_handler(signum, frame):
    raise Exception("end of time")


# We work in NxN:
num_noise = 90
for N in [5, 10, 20, 40, 80]:
    # Helper function.
    def testing_domain_and_loss_and_budget(d, l, b, name):
        score = defaultdict(list)

        def get_score(optim_name):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # seconds at most
            try:
                recommendation = ng.optimizers.registry[optim_name](domain, b).minimize(l)
                try:
                    loss_value = (
                        l(recommendation.value)
                        if not "noisy" in name
                        else np.sum([l(recommendation.value) for _ in range(num_noise)]) / num_noise
                    )
                except:  # For complex domains!
                    loss_value = (
                        l(*recommendation.args, **recommendation.kwargs)
                        if not "noisy" in name
                        else np.sum(
                            [l(*recommendation.args, **recommendation.kwargs) for _ in range(num_noise)]
                        )
                        / num_noise
                    )
                print(f"Algorithm {optim_name} got {loss_value}")
            except Exception as e:
                loss_value = float(1e10)
                print(f"Algorithm {optim_name} crashed: {e}!")
            signal.alarm(0)
            if not loss_value == loss_value:
                loss_value = float(1e10)
            return loss_value

        for idx in range(560):
            # Let us test several algorithms.
            optim_names = sorted(list(ng.optimizers.registry.keys()), key=lambda o: ucb_score(score[o], idx))[
                :50
            ]
            results = Parallel(n_jobs=len(optim_names))(delayed(get_score)(o) for o in optim_names)
        assert len(results) == len(optim_names)
        for optim_name, loss_values in zip(optim_names, results):
            score[optim_name] += [loss_value]

        print(f"List of best for N={N} and budget={b} and name={name}:")
        for i, u in enumerate(sorted(score, key=lambda x: lcb_score(score[x]))):
            print(
                "Rank ", i, ":", u, ",", np.sum(score[u]) / len(score[u]), "(", len(score[u]), ")", score[u]
            )

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

    # print(ng.optimizers.registry["NGOpt"](domain, 10 * N * N).minimize(loss).value)
    testing_domain_and_loss_and_budget(domain, loss, 5 * N * N, "c0,2d")

    # Case "discrete multidimensional array": I want to optimize a NxN array of bool.
    print("Now, we optimize on a discrete domain.")
    domain = ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting()
    # print(ng.optimizers.registry["DiscreteLenglerOnePlusOne"](domain, 10 * N * N).minimize(loss).value)
    testing_domain_and_loss_and_budget(domain, loss, 5 * N * N, "int,2d")

    # Case "combination":
    print("Now, let us work on a mixed domain continuous/discrete.")
    domain = ng.p.Instrumentation(
        x=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        y=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
    )

    def complex_loss(x, y):
        return loss(x) + loss(np.transpose(y))

    def correlated_loss(x, y):
        return loss(x) + loss(np.transpose(y))

    # print(ng.optimizers.registry["NGOpt"](domain, 10 * N * N).minimize(complex_loss).value)
    testing_domain_and_loss_and_budget(domain, complex_loss, 5 * N * N, "c0+int, 2d")
    testing_domain_and_loss_and_budget(domain, correlated_loss, 5 * N * N, "c0+int, 2d, corr")

    # Case "multiple blocks"
    domain = ng.p.Instrumentation(
        x1=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        x2=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        x3=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        x4=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        x5=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
        x6=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0),
    )

    def six_losses(x1, x2, x3, x4, x5, x6):
        return loss(x1) + loss(x2) + loss(x3) + loss(x4) + loss(x5) + loss(x6)

    def noisy_six_losses(x1, x2, x3, x4, x5, x6):
        return six_losses(x1, x2, x3, x4, x5, x6) + np.random.randn() * 17.0

    testing_domain_and_loss_and_budget(domain, six_losses, 5 * N * N, "sixlosses, 2d, c0")
    testing_domain_and_loss_and_budget(domain, noisy_six_losses, 5 * N * N, "noisysixlosses, 2d, c0")

    # Case "multiple blocks"
    domain = ng.p.Instrumentation(
        x1=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
        x2=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
        x3=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
        x4=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
        x5=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
        x6=ng.p.Array(shape=(N, N), upper=1.0, lower=0.0).set_integer_casting(),
    )

    def six_losses(x1, x2, x3, x4, x5, x6):
        return loss(x1) + loss(x2) + loss(x3) + loss(x4) + loss(x5) + loss(x6)

    def noisy_six_losses(x1, x2, x3, x4, x5, x6):
        return six_losses(x1, x2, x3, x4, x5, x6) + np.random.randn() * 17.0

    testing_domain_and_loss_and_budget(domain, six_losses, 5 * N * N, "sixlosses, 2d, int")
    testing_domain_and_loss_and_budget(domain, noisy_six_losses, 5 * N * N, "noisysixlosses, 2d, int")
