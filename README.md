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

By default, this only installs requirements for the optimization and instrumentation subpackages. If you are also interesting in the benchmarking part,
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

![Example of benchmark result](TwoPointsDE.gif)

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
from nevergrad.optimization import optimizerlib

def square(x):
    return sum((x - .5)**2)

optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100)
# alternatively, you can use optimizerlib.registry which is a dict containing all optimizer classes
recommendation = optimizer.optimize(square)
```

You can print the full list of optimizers with:
```
from nevergrad.optimization import registry
print(sorted(registry.keys()))
```

The [optimization documentation](docs/optimization.md) contains more information on how to use several workers, take full control of the optimization through the `ask` and `tell` interface and some pieces of advice on how to choose the proper optimizer for your problem.

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
