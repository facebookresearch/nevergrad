# Optimization

**All optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.**

## Basic example

Optimizing (minimizing!) a function using an optimizer (here `OnePlusOne`) can be easily run with:

```python
from nevergrad.optimization import optimizerlib

def square(x):
    return sum((x - .5)**2)

optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100)
# alternatively, you can use optimizerlib.registry which is a dict containing all optimizer classes
recommendation = optimizer.optimize(square)
```


## Using several workers

Running the funciton evaluation in parallel with several workers is as easy as provided an executor:
```python
from concurrent import futures
with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.optimize(square, executor=executor, batch_mode=False, num_workers=5)
```
`num_workers=5` with `batch_mode=True` will ask the optimizer for 5 points to evaluate, run the evaluations, then update the optimizer with the 5 function outputs, and repeat until the budget is all spent. Since no executor is provided, the evaluations will be sequential. `num_workers > 1` with no executor is therefore suboptimal but nonetheless useful for evaluation purpose (i.e. we simulate parallelism but have no actual parallelism). `batch_mode=False` (steady state mode) will ask for a new evaluation whenever a worker is ready.

## Ask and tell interface

An *ask and tell* interface is also available. The 3 key methods for this interface are respectively:
- `ask`: suggest a point on which to evaluate the function to optimize.
- `tell`: for updated the optimizer with the value of the function at a given point.
- `provide_recommendation`: returns the point the algorithms considers the best.
For most optimization algorithms in the platform, they can be called in arbitrary order - asynchronous optimization is OK. Some algorithms (with class attribute `no_parallelization=True` however do not support this.

Here is a simpler example in the sequential case (this is what happens in the `optimize` method for `num_workers=1`):
```python
for _ in range(optimizer.budget):
    x = optimizer.ask()
    value = square(x)
    optimizer.tell(x, value)
recommendation = optimizer.provide_recommendation()
```

Please make sure that your function returns a float, and that you indeed want to perform minimization and not maximization ;)

## Choosing an optimizer

**You can print the full list of optimizers** with:
```
from nevergrad.optimization import registry
print(sorted(registry.keys()))
```

All algorithms have strenghts and weaknesses. Questionable rules of thumb could be:
- `TwoPointsDE` is excellent in many cases, including very high `num_workers`.
- `PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant; it's possibly a good choice for hyperparameter choice.
- `OnePlusOne` is a simple robust method for continuous parameters with `num_workers` < 8.
- `CMA` is excellent for control (e.g. neurocontrol) when the environment is not very noisy (num_workers ~50 ok) and when the budget is large (e.g. 1000 x the dimension).
- `TBPSA` is excellent for problems corrupted by noise, in particular overparametrized (neural) ones; very high `num_workers` ok).
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
