[![CircleCI](https://circleci.com/gh/facebookresearch/nevergrad/tree/master.svg?style=svg)](https://circleci.com/gh/facebookresearch/nevergrad/tree/master)

# Nevergrad - A gradient-free optimization platform

`nevergrad` is a Python 3.6+ library. It can be installed with:

```
pip install nevergrad
```

You can also install the master branch instead of the latest release with:

```
pip install git+https://github.com/facebookresearch/nevergrad@master#egg=nevergrad
```

Alternatively, you can clone the repository and run `pip install -e .` from inside the repository folder.

By default, this only installs requirements for the optimization and instrumentation subpackages. If you are also interested in the benchmarking part,
you should install with the `[benchmark]` flag (example: `pip install 'nevergrad[benchmark]'`), and if you also want the test tools, use
the `[all]` flag (example: `pip install -e '.[all]'`)


## Goals and structure

The goals of this package are to provide:
- **gradient/derivative-free optimization algorithms**, including algorithms able to handle noise.
- **tools to instrument any code**, making it painless to optimize your parameters/hyperparameters, whether they are continuous, discrete or a mixture of continuous and discrete variables.
- **functions** on which to test the optimization algorithms.
- **benchmark routines** in order to compare algorithms easily.

The structure of the package follows its goal, you will therefore find subpackages:
- `optimization`: implementing optimization algorithms
- `instrumentation`: tooling to convert code into a well-defined function to optimize.
- `functions`: implementing both simple and complex benchmark functions
- `benchmark`: for running experiments comparing the algorithms on benchmark functions
- `common`: a set of tools used throughout the package

![Example of optimization](TwoPointsDE.gif)

*Convergence of a population of points to the minima with two-points DE.*


## Documentation

The following README is very general, here are links to find more details on:
- [how to perform optimization](docs/optimization.md) using `nevergrad`, including using parallelization and a few recommendation on which algorithm should be used depending on the settings
- [how to instrument](docs/instrumentation.md) functions with any kind of parameters in order to convert them into a function defined on a continuous vectorial space where optimization can be performed. It also provides a tool to instantiate a script or non-python code in order into a Python function and be able to tune some of its parameters.
- [how to benchmark](docs/benchmarking.md) all optimizers on various test functions.
- [benchmark results](docs/benchmarks.md) of some standard optimizers an simple test cases.
- examples of [optimization for machine learning](docs/machinelearning.md).
- how to [contribute](.github/CONTRIBUTING.md) through issues and pull requests and how to setup your dev environment.
- guidelines of how to contribute by [adding a new algorithm](docs/adding_an_algorithm.md).


## Basic optimization example

**All optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.**


Optimizing (minimizing!) a function using an optimizer (here `OnePlusOne`) can be easily run with:

```python
import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
recommendation = optimizer.optimize(square)
print(recommendation)  # optimal args and kwargs
>>> Candidate(args=(array([0.500, 0.499]),), kwargs={})
```

`recommendation` holds the optimal attributes `args` and `kwargs` found by the optimizer for the provided function.
In this example, the optimal value will be found in `recommendation.args[0]` and will be a `np.ndarray` of size 2.

`instrumentation=n` is a shortcut to state that the function has only one variable, of dimension `n`,
See the [instrumentation tutorial](docs/instrumentation.md) for more complex instrumentations.


You can print the full list of optimizers with:
```python
import nevergrad as ng
print(list(sorted(ng.optimizers.registry.keys())))
```

The [optimization documentation](docs/optimization.md) contains more information on how to use several workers, take full control of the optimization through the `ask` and `tell` interface and some pieces of advice on how to choose the proper optimizer for your problem.


## Example of chaining, or inoculation, or initialization of an evolutionary algorithm

Chaining consists in running several algorithms in turn, information being forwarded from the first to the second and so on.
More precisely, the budget is distributed over several algorithms, and when an objective function value is computed, all algorithms are informed.
In optimizerlib.py, you can read the following:
```
DEwithLHS = chaining([LHSSearch, DE], [-1])  # Runs LHSSearch with budget num_workers and then DE.
DEwithLHSdim = chaining([LHSSearch, DE], [-2])  # Runs LHSSearch with budget the dimension and then DE.
DEwithLHS30 = chaining([LHSSearch, DE], [30])  # Runs LHSSearch with budget the dimension and then DE.
```

We could also do 
```
my_chain = chaining([LHSSearch, DE, CMA], [100, 60, 1000])
```

Using `chaining([LHSSearch, DE]` means that we ``chain'' LHS and DE (in the first case LHS is run with as budget the number of workers, and in the second case with budget equal to the dimension).
We can then do:
```python
import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.DEwithLHS30(instrumentation=2, budget=100)
recommendation = optimizer.optimize(square)
print(recommendation)  # optimal args and kwargs
>>> Candidate(args=(array([0.500, 0.499]),), kwargs={})
```

## Citing

```bibtex
@misc{nevergrad,
    author = {J. Rapin and O. Teytaud},
    title = {{Nevergrad - A gradient-free optimization platform}},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://GitHub.com/FacebookResearch/Nevergrad}},
}
```

## License

`nevergrad` is released under the MIT license. See [LICENSE](LICENSE) for additional details.
