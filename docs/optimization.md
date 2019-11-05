# Optimization

**All optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.**

## Basic example

Optimizing (minimizing!) a function using an optimizer (here `OnePlusOne`) can be easily run with:

```python
import nevergrad as ng

def square(x, y=12):
    return sum((x - .5)**2) + abs(y)

optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
# alternatively, you could use ng.optimizers.registry["OnePlusOne"]
# (registry is a dict containing all optimizer classes)
recommendation = optimizer.minimize(square)
print(recommendation)
>>> Candidate(args=(array([0.500, 0.499]),), kwargs={})
```
`recommendation` holds the optimal attributes `args` and `kwargs` found by the optimizer for the provided function.
In this example, the optimal value will be found in `recommendation.args[0]` and will be a `np.ndarray` of size 2.

`instrumentation=n` is a shortcut to state that the function has only one variable, of dimension `n`,
Defining the following instrumentation instead will optimize on both `x` and `y`.
```python
instrum = ng.Instrumentation(ng.var.Array(2), y=ng.var.Array(1).asscalar())
optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation)
>>> Candidate(args=(array([0.490, 0.546]),), kwargs={'y': 0.0})
```
See the [instrumentation tutorial](instrumentation.md) for more complex instrumentations.


## Using several workers

Running the function evaluation in parallel with several workers is as easy as providing an executor:
```python
from concurrent import futures
optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100, num_workers=5)
with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(square, executor=executor, batch_mode=False)
```
With `batch_mode=True` it will ask the optimizer for `num_workers` points to evaluate, run the evaluations, then update the optimizer with the `num_workers` function outputs, and repeat until the budget is all spent. Since no executor is provided, the evaluations will be sequential. `num_workers > 1` with no executor is therefore suboptimal but nonetheless useful for evaluation purpose (i.e. we simulate parallelism but have no actual parallelism). `batch_mode=False` (steady state mode) will ask for a new evaluation whenever a worker is ready.

## Ask and tell interface

An *ask and tell* interface is also available. The 3 key methods for this interface are respectively:
- `ask`: suggest a candidate on which to evaluate the function to optimize.
- `tell`: for updated the optimizer with the value of the function for a candidate.
- `provide_recommendation`: returns the candidate the algorithms considers the best.
For most optimization algorithms in the platform, they can be called in arbitrary order - asynchronous optimization is OK. Some algorithms (with class attribute `no_parallelization=True` however do not support this.

The `Candidate` class holds attributes `args` and `kwargs` corresponding to the `args` and `kwargs` of the function you optimize,
given its [instrumentation](instrumentation.md). It also holds a `data` attribute corresponding to the data point in the optimization space.

Here is a simpler example in the sequential case (this is what happens in the `optimize` method for `num_workers=1`):
```python
for _ in range(optimizer.budget):
    x = optimizer.ask()
    value = square(*x.args, **x.kwargs)
    optimizer.tell(x, value)
recommendation = optimizer.provide_recommendation()
```

Please make sure that your function returns a float, and that you indeed want to perform minimization and not maximization ;)


## Optimization with constraints

Nevergrad has a mechanism for cheap constraints.
"Cheap" means that we do not try to reduce the number of calls to such constraints.
We basically repeat mutations until we get a satisfiable point.
Let us say that we want to minimize `(x[0]-.5)**2 + (x[1]-.5)**2` under the constraint `x[0] >= 1`.
```python
import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
# define a constraint on first variable of x:
optimizer.instrumentation.set_cheap_constraint_checker(lambda x: x[0] >= 1)

recommendation = optimizer.minimize(square)
print(recommendation)  # optimal args and kwargs
>>> Candidate(args=(array([1.00037625, 0.50683314]),), kwargs={})
```

## Choosing an optimizer

**You can print the full list of optimizers** with:
```python
import nevergrad as ng
print(sorted(ng.optimizers.registry.keys()))
```

All algorithms have strengths and weaknesses. Questionable rules of thumb could be:
- `TwoPointsDE` is excellent in many cases, including very high `num_workers`.
- `PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant; it's possibly a good choice for hyperparameter choice.
- `OnePlusOne` is a simple robust method for continuous parameters with `num_workers` < 8.
- `CMA` is excellent for control (e.g. neurocontrol) when the environment is not very noisy (num_workers ~50 ok) and when the budget is large (e.g. 1000 x the dimension).
- `TBPSA` is excellent for problems corrupted by noise, in particular overparameterized (neural) ones; very high `num_workers` ok).
- `PSO` is excellent in terms of robustness, high `num_workers` ok.
- `ScrHammersleySearchPlusMiddlePoint` is excellent for super parallel cases (fully one-shot, i.e. `num_workers` = budget included) or for very multimodal cases (such as some of our MLDA problems); don't use softmax with this optimizer.
- `RandomSearch` is the classical random search baseline; don't use softmax with this optimizer.

## Optimizing machine learning hyperparameters

When optimizing hyperparameters as e.g. in machine learning. If you don't know what variables (see [instrumentation](instrumentation.md)) to use:
- use `SoftmaxCategorical` for discrete variables
- use `TwoPointsDE` with `num_workers` equal to the number of workers available to you.
See the [machine learning example](machinelearning.md) for more.

Or if you want something more aimed at robustly outperforming random search in highly parallel settings (one-shot):
- use `OrderedDiscrete` for discrete variables, taking care that the default value is in the middle.
- Use `ScrHammersleySearchPlusMiddlePoint` (`PlusMiddlePoint` only if you have continuous parameters or good default values for discrete parameters).


## Example of chaining, or inoculation, or initialization of an evolutionary algorithm

Chaining consists in running several algorithms in turn, information being forwarded from the first to the second and so on.
More precisely, the budget is distributed over several algorithms, and when an objective function value is computed, all algorithms are informed.

Here is how to create such optimizers:
```python
# Running LHSSearch with budget num_workers and then DE:
DEwithLHS = Chaining([LHSSearch, DE], ["num_workers"])

# Runninng LHSSearch with budget the dimension and then DE:
DEwithLHSdim = Chaining([LHSSearch, DE], ["dimension"])

# Runnning LHSSearch with budget 30 and then DE:
DEwithLHS30 = Chaining([LHSSearch, DE], [30])

# Running LHS for 100 iterations, then DE for 60, then CMA:
LHSthenDEthenCMA = Chaining([LHSSearch, DE, CMA], [100, 60])
```

We can then minimize as usual:
```python
import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = DEwithLHS30(instrumentation=2, budget=300)
recommendation = optimizer.minimize(square)
print(recommendation)  # optimal args and kwargs
>>> Candidate(args=(array([0.50843113, 0.5104554 ]),), kwargs={})
```


## Reproducibility

Each instrumentation has its own `random_state` for generating random numbers. All optimizers pull from it when they require stochastic behaviors.
For reproducibility, this random state can be seeded in two ways:
- by setting `numpy`'s global random state seed (`np.random.seed(32)`) before the instrumentation's first use. Indeed, when first used,
  the instrumentation's random state is seeded with a seed drawn from the global random state.
- by manually seeding the instrumentation random state (E.g.: `instrumentation.random_state.seed(12)` or `optimizer.instrumentation.random_state = np.random.RandomState(12)`)
