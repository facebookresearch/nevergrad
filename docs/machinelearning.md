# Nevergrad for machine learning: which optimizer should I use ?

Let us assume that you have defined an objective function as in
```python
train_and_return_test_error_mixed = instru.InstrumentedFunction(myfunction, arg1, arg2,
                                                                "blublu", value=value)
```
(as in an example below)

If you have both continuous and discrete parameters, you have a good initial guess, maybe just use `OrderedDiscrete` for all discrete variables (yes, even if they are not ordered), `Gaussian` for all your continuous variables, and use `PortfolioDiscreteOnePlusOne`. Just take care that the default value (your initial guess) is at the middle in the list of possible values. You can check that things are correct by checking that for zero you get the default:
```python
dim = train_and_return_test_error_mixed.dimension
args, kwarg = train_and_return_test_error_mixed.convert_to_arguments([0] * dim)
print(args)
print(kwarg)
```

The fact that you use ordered discrete variables is not a big deal because by nature `PortfolioDiscreteOnePlusOne` will ignore the order. This algorithm is quite stable.

If you have more budget, a cool possibility is to use `CategoricalSoftmax` for all discrete variables and then apply `TwoPointsDE`. You might also compare this to DE (classical differential evolution). This might need a budget in the hundreds.

If you want to double-check that you are not worse than random search, you might use `RandomSearch`.

If you want something fully parallel (the number of workers can be equal to the budget), then you might use `ScrHammersleySearch`. Yes, this includes the discrete case. Then, you should use `OrderedDiscrete` rather than `CategoricalSoftmax`. This does not have the traditional drawback of grid search and should still be more uniform than random. By nature `ScrHammersleySearch` will deal correctly with `OrderedDiscrete` type for `CategoricalSoftmax`.

If you are optimizing weights in reinforcement learning, you might use `TBPSA` (high noise) or `CMA` (low noise).





# Nevergrad applied to Machine Learning: 3 examples.

The first example is simply the optimization of continuous hyperparameters.
It is also presented in an asynchronous setting. All other examples are based on the ask and tell interface, which can be synchronous or not but relies on the user for setting up asynchronicity.

The second example is the optimization of mixed (continuous and discrete) hyperparameters. A second, more complicated, objective function is proposed (just uncomment).

The third example is the optimization of parameters in a noisy setting, typically as in reinforcement learning.

## First example: optimization of continuous hyperparameters with CMA, PSO, DE, Random and QuasiRandom. Synchronous version.

```python
import nevergrad.optimization as optimization
import numpy as np


# Optimization of continuous hyperparameters.
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
        # Here we ask 3 times and tell 3 times in order to fake asynchronicity
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
```

## First example: optimization of continuous hyperparameters with CMA, PSO, DE, Random and QuasiRandom. Asynchronous version.

```python
from concurrent import futures
import nevergrad.optimization as optimization
import numpy as np


# Optimization of continuous hyperparameters.
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

    with futures.ThreadPoolExecutor(max_workers=optim.num_workers) as executor:
        recommendation = optim.optimize(train_and_return_test_error, executor=executor)
    print("* ", tool, " provides a vector of parameters with test error ",
          train_and_return_test_error(recommendation))

```
## Second example: optimization of mixed (continuous and discrete) hyperparameters.

```python
import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
import numpy as np
# Optimization of mixed (continuous and discrete) hyperparameters.
# We apply a softmax for converting real numbers to discrete values.


print("Optimization of mixed (continuous and discrete) hyperparameters ======")


# Let us define a function.
def myfunction(arg1, arg2, arg3, value=3):
    print(arg1, arg2, arg3, value, 0 * np.abs(value) + (1 if arg1 != "a" else 0) + (1 if arg2 != "e" else 0))
    return 0 * np.abs(value) + (1 if arg1 != "a" else 0) + (1 if arg2 != "e" else 0)


# argument transformation
arg1 = instru.variables.OrderedDiscrete(["a", "b"])  # 1st arg. = positional discrete argument
arg2 = instru.variables.SoftmaxCategorical(["a", "c", "e"])  # 2nd arg. = positional discrete argument
value = instru.variables.Gaussian(mean=1, std=2)  # the 4th arg. is a keyword argument with Gaussian prior

# create the instrumented function
train_and_return_test_error_mixed = instru.InstrumentedFunction(myfunction, arg1, arg2, "blublu", value=value)
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print(train_and_return_test_error_mixed.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete var. has 1 possible values, represented by a hard thresholding in a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete var. has 3 possible values, represented by softmax,
#   i.e. we add 3 coordinates to the continuous problem
# - the 3rd var. has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th var. is a real number, represented by single coordinate.

train_and_return_test_error_mixed([1, -80, -80, 80, 3])  # will print "b e blublu" and return 49 = (mean + std * arg)**2 = (1 + 2 * 3)**2
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))
dimension = train_and_return_test_error_mixed.dimension

#Below two examples in case you prefer to do the instrumentation manually.
#def softmax(x, possible_values=None):
#    expx = [np.exp(x_ - max(x)) for x_ in x]
#    probas = [e / sum(expx) for e in expx]
#    return np.random.choice(len(x) if possible_values is None
#            else possible_values, size=1, p=probas)
#
#
#def train_and_return_test_error_mixed(x):
#    cx = [x_ - 0.1 for x_ in x[3:]]
#    activation = softmax(x[:3], ["tanh", "sigmoid", "relu"])
#    return np.linalg.norm(cx) + (1. if activation != "tanh" else 0.)
#dimension = 10
#
#This version is bigger.
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
#dimension = 300


budget = 1200  # How many episode we will do before concluding.

# PortfolioDiscreteOnePlusOne is quite a natural choice when you have a good initial guess and a mix of discrete and continuous variables;
# in this case, it might be better to use OrderedDiscrete rather than CategoricalSoftmax.
# TwoPointsDE is often excellent in the large scale case (budget in the hundreds).
for tool in ["RandomSearch", "ScrHammersleySearch", "TwoPointsDE", "PortfolioDiscreteOnePlusOne", "CMA", "PSO"]:

    optim = optimization.registry[tool](dimension=dimension, budget=budget)

    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        # Here we ask 3 times and tell 3 times in order to fake asynchronicity
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

# you can then recover the arguments in the initial function domain with:
args, kwargs = train_and_return_test_error_mixed.convert_to_arguments(recommendation)
# or print a summary
print(train_and_return_test_error_mixed.get_summary(recommendation))
```

## Third example: optimization of parameters for reinforcement learning.

We do not average evaluations over multiple episodes - the algorithm is in charge of averaging, if need be.
`TBPSA`, based on population-control mechasnisms, performs quite well in this case.

```python
import nevergrad.optimization as optimization
import numpy as np

# Similar, but with a noisy case: typically a case in which we train in reinforcement learning.
# This is about parameters rather than hyperparameters. TBPSA is a strong candidate in this case.
# We do *not* manually average over multiple evaluations; the algorithm will take care
# of averaging or reevaluate whatever it wants to reevaluate.


print("Optimization of parameters in reinforcement learning ===============")


def simulate_and_return_test_error_with_rl(x, noisy=True):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x]) + noisy * len(x) * np.random.normal()


budget = 1200  # How many trainings we will do before concluding.


for tool in ["TwoPointsDE", "RandomSearch", "TBPSA", "CMA", "NaiveTBPSA",
        "PortfolioNoisyDiscreteOnePlusOne"]:

    optim = optimization.registry[tool](dimension=300, budget=budget)

    for u in range(budget // 3):
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        # Here we ask 3 times and tell 3 times in order to fake asynchronicity
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
          simulate_and_return_test_error_with_rl(recommendation, noisy=False))
```
