# Nevergrad applied to Machine Learning: 3 examples.

The first example is simply the optimization of continuous hyperparameters.

The second example is the optimization of mixed (continuous and discrete) hyperparameters. A second, more complicated, objective function is proposed (just uncomment).

The third example is the optimization of parameters in a noisy setting, typically as in reinforcement learning.

```python

import nevergrad.optimization as optimization
import numpy as np


# Optimization of continuous hyperparameters.

print(" ")
print(" ")
print("Optimization of continuous hyperparameters =========")


def train_and_return_test_error(x):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x])


budget = 1200  # How many trainings we will do before concluding.


# We compare several algorithms.
# "RandomSearch" is well known, "ScrHammersleySearch" is a quasirandom; these two methods
# are fully parallel, i.e. we can perform the 1200 trainings in parallel.
# "CMA" and "PSO" are classical optimization algorithms, and "TwoPointsDE"
# is Differential Evolution equipped with a 2-points crossover.
# A complete list is available in optimization.registry.
for tool in ["RandomSearch", "TwoPointsDE", "CMA", "PSO", "ScrHammersleySearch"]:

    optim = optimization.registry[tool](dimension=300, budget=budget)
    
    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        x1 = optim.ask()
        x2 = optim.ask()
        x3 = optim.ask()
        # The three folowing lines could be parallelized.
        # We could also do things asynchronously, i.e. do one more ask
        # as soon as a training is over.
        y1 = train_and_return_test_error(x1)
        y2 = train_and_return_test_error(x2)
        y3 = train_and_return_test_error(x3)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
        optim.tell(x3, y3)
    
    recommendation = optim.provide_recommendation()
    print("* ", tool, " provides a vector of parameters with test error ",
          train_and_return_test_error(recommendation))



# Optimization of mixed (continuous and discrete) hyperparameters.
# We apply a softmax for converting real numbers to discrete values.
print(" ")
print(" ")
print("Optimization of mixed (continuous and discrete) hyperparameters ======")


def softmax(x, possible_values=None):
    expx = [np.exp(x_ - max(x)) for x_ in x]
    probas = [e / sum(expx) for e in expx]
    return np.random.choice(len(x) if possible_values is None
            else possible_values, size=1, p=probas)


def train_and_return_test_error_mixed(x):
    cx = [x_ - 0.1 for x_ in x[3:]]
    activation = softmax(x[:3], ["tanh", "sigmoid", "relu"])
    return np.linalg.norm(cx) + (1. if activation != "tanh" else 0.)

#This version is possibly bigger.
#def train_and_return_test_error_mixed(x):
#    cx = x[:(len(x) // 2)]  # continuous part.
#    presoftmax_values = x[(len(x) // 2):]  # discrete part.
#    values_for_this_softmax = []
#    dx = []
#    for g in presoftmax:
#        values_for_this_softmax += [g]
#        if len(values_for_this_softmax) > 4:
#            dx += softmax(values_for_this_softmax)
#            values_for_this_softmax = []
#    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in cx]) + [
#            1 if d != 1 else 0 for d in dx]



budget = 1200  # How many episode we will do before concluding.


for tool in ["RandomSearch", "TwoPointsDE", "CMA", "PSO"]:

    optim = optimization.registry[tool](dimension=300, budget=budget)
    
    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        x1 = optim.ask()
        x2 = optim.ask()
        x3 = optim.ask()
        # The three folowing lines could be parallelized.
        # We could also do things asynchronously, i.e. do one more ask
        # as soon as a training is over.
        y1 = train_and_return_test_error_mixed(x1)
        y2 = train_and_return_test_error_mixed(x2)
        y3 = train_and_return_test_error_mixed(x3)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
        optim.tell(x3, y3)
    
    recommendation = optim.provide_recommendation()
    print("* ", tool, " provides a vector of parameters with test error ",
          train_and_return_test_error_mixed(recommendation))



# Similar, but with a noisy case: typically a case in which we train in reinforcement learning.
# This is about parameters rather than hyperparameters. TBPSA is a strong candidate in this case.
# We do *not* manually average over multiple evaluations; the algorithm will take care of averaging or reevaluate
# whatever it wants to reevaluate.

print(" ")
print(" ")
print("Optimization of parameters in reinforcement learning ===============")


def simulate_and_return_test_error_with_rl(x):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x]) + len(x) * np.random.normal()


def simulate_and_return_test_error_with_rl_without_noise(x):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x])


budget = 1200  # How many trainings we will do before concluding.


for tool in ["TwoPointsDE", "RandomSearch", "TBPSA", "CMA", "NaiveTBPSA",
        "PortfolioNoisyDiscreteOnePlusOne"]:

    optim = optimization.registry[tool](dimension=300, budget=budget)
    
    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        x1 = optim.ask()
        x2 = optim.ask()
        x3 = optim.ask()
        # The three folowing lines could be parallelized.
        # We could also do things asynchronously, i.e. do one more ask
        # as soon as a training is over.
        y1 = simulate_and_return_test_error_with_rl(x1)
        y2 = simulate_and_return_test_error_with_rl(x2)
        y3 = simulate_and_return_test_error_with_rl(x3)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
        optim.tell(x3, y3)
    
    recommendation = optim.provide_recommendation()
    print("* ", tool, " provides a vector of parameters with test error ",
          simulate_and_return_test_error_with_rl_without_noise(recommendation))




```



