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

By default, this only installs requirements for the optimization and parametrization subpackages. If you are also interested in the benchmarking part,
you should install with the `[benchmark]` flag (example: `pip install nevergrad[benchmark]`), and if you also want the test tools, use
the `[all]` flag (example: `pip install -e .[all]`).

**Notes**:
- with `zhs` you will need to run `pip install 'nevergrad[all]'` instead of `pip install nevergrad[all]`
- under Windows, you may need to preinstall torch (for `benchmark` or `all` installations) using instructions [here](https://pytorch.org/get-started/locally/).


You can join Nevergrad users Facebook group [here](https://www.facebook.com/groups/nevergradusers/).


## Goals and structure

The goals of this package are to provide:
- **gradient/derivative-free optimization algorithms**, including algorithms able to handle noise.
- **tools to parametrize any code**, making it painless to optimize your parameters/hyperparameters, whether they are continuous, discrete or a mixture of continuous and discrete parameters.
- **functions** on which to test the optimization algorithms.
- **benchmark routines** in order to compare algorithms easily.

The structure of the package follows its goal, you will therefore find subpackages:
- `optimization`: implementing optimization algorithms
- `parametrization`: specifying what are the parameters you want to optimize
- `functions`: implementing both simple and complex benchmark functions
- `benchmark`: for running experiments comparing the algorithms on benchmark functions
- `common`: a set of tools used throughout the package

![Example of optimization](TwoPointsDE.gif)

*Convergence of a population of points to the minima with two-points DE.*


## Documentation

The following README is very general, here are links to find more details on:
- [how to perform optimization](docs/optimization.md) using `nevergrad`, including using parallelization and a few recommendation on which algorithm should be used depending on the settings
- [how to parametrize](docs/parametrization.md) your problem so that the optimizers are informed of the problem to solve. This also provides a tool to instantiate a script or non-python code into a Python function and be able to tune some of its parameters.
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

optimizer = ng.optimizers.OnePlusOne(parametrization=2, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation)  # optimal args and kwargs
>>> Array{(2,)}[recombination=average,sigma=1.0]:[0.49971112 0.5002944 ]
```

`parametrization=n` is a shortcut to state that the function has only one variable, of dimension `n`,
See the [parametrization tutorial](docs/parametrization.md) for more complex parametrizations.

`recommendation` holds the optimal value(s) found by the for the provided function. It can be
directly accessed through `recommendation.value` which is here a `np.ndarray` of size 2.

You can print the full list of optimizers with:
```python
import nevergrad as ng
print(list(sorted(ng.optimizers.registry.keys())))
```

The [optimization documentation](docs/optimization.md) contains more information on how to use several workers,
take full control of the optimization through the `ask` and `tell` interface, perform multiobjective optimization,
as well as pieces of advice on how to choose the proper optimizer for your problem.


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

`nevergrad` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
LGPL code is however also included in the multiobjective subpackage.
