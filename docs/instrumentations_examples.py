import numpy as np
import nevergrad as ng

# Function for creating timeouts.
import signal
def timeout_handler(signum, frame):
    raise Exception("end of time")


# We work in NxN:
for N in [5, 10, 20, 40, 80]:
    # Helper function.
    def testing_domain_and_loss_and_budget(d, l, b):
        score = {}
        for _ in range(260):
            # Let us test several algorithms.
            optim_name = np.random.choice(list(ng.optimizers.registry.keys()))
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # seconds at most
            try:
                recommendation = ng.optimizers.registry[optim_name](domain, b).minimize(l)
                try:
                    loss_value = l(recommendation.value)
                except:  # For complex domains!
                    loss_value = l(*recommendation.args, **recommendation.kwargs)
                print(f"Algorithm {optim_name} got {loss_value}")
                if optim_name in score:
                    score[optim_name] += [loss_value]
                else:
                    score[optim_name] = [loss_value]
            except Exception as e:
                loss_value = float(1e10)
                if optim_name in score:
                    score[optim_name] += [loss_value]
                else:
                    score[optim_name] = [loss_value]
                print(f"Algorithm {optim_name} crashed: {e}!")
            signal.alarm(0) 
    
        print(f"List of best for N={N} and budget={b}:")
        print(sorted(score, key=lambda x: np.sum(score[x]) / len(score[x])))
    
    
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
    testing_domain_and_loss_and_budget(domain, loss, 5*N*N)
    
    # Case "discrete multidimensional array": I want to optimize a NxN array of bool.
    print("Now, we optimize on a discrete domain.")
    domain = ng.p.Array(shape=(N,N), upper=1., lower=0.).set_integer_casting()
    print(ng.optimizers.registry["DiscreteLenglerOnePlusOne"](domain, 10*N*N).minimize(loss).value)
    testing_domain_and_loss_and_budget(domain, loss, 5*N*N)
    
    # Case "combination":
    print("Now, let us work on a mixed domain continuous/discrete.")
    domain = ng.p.Instrumentation(
         x = ng.p.Array(shape=(N,N), upper=1., lower=0.),
         y = ng.p.Array(shape=(N,N), upper=1., lower=0.).set_integer_casting(),
      )
    def complex_loss(x, y):
        return loss(x) + loss(np.transpose(y))
    print(ng.optimizers.registry["NGOpt"](domain, 10*N*N).minimize(complex_loss).value)
    testing_domain_and_loss_and_budget(domain, complex_loss, 5*N*N)
