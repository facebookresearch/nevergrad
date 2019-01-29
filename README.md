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

Alternatively, you can clone the repository and run `python3 setup.py develop` from inside the repository folder.

The following README is very general, you can find:
- machine learning examples in [machinelearning.md](docs/machinelearning.md).
- optimization of classical objective functions in [benchmarks.md](docs/benchmarks.md).
- guidelines of how to contribute by adding a new algorithm in [adding_an_algorithm.md](docs/adding_an_algorithm.md)

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

## Optimization

**All optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.**

Optimizing (minimizing!) a function using an optimizer (here OnePlusOne) can be easily run with:

```python
from nevergrad.optimization import optimizerlib

def square(x):
    return sum((x - .5)**2)

optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100, num_workers=5)
# alternatively, you can use optimizerlib.registry which is a dict containing all optimizer classes
recommendation = optimizer.optimize(square, executor=None, batch_mode=True)
```

`num_workers=5` with `batch_mode=True` will ask the optimizer for 5 points to evaluate, run the evaluations, then update the optimizer with the 5 function outputs, and repeat until the budget is all spent. Since no executor is provided, the evaluations will be sequential. `num_workers > 1` with no executor is therefore suboptimal but nonetheless useful for evaluation purpose (i.e. we simulate parallelism but have no actual parallelism).

Providing an executor is easy:
```python
from concurrent import futures
with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.optimize(square, executor=executor, batch_mode=True)
```

`batch_mode=False` (steady state mode) will ask for a new evaluation whenever a worker is ready. The current implementation is efficient in this sense; but keep in mind that this steady state mode has drawbacks in terms of reproducibility: the order in which evaluations are launched is not controlled.

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

### Choosing an optimizer

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

**Optimizing machine learning hyperparameters**:

When optimizing hyperparameters as e.g. in machine learning. If you don't know what to do,
- use `SoftmaxCategorical` for discrete variables (see below).
- use `TwoPointsDE` with `num_workers` equal to the number of workers available to you.
See https://github.com/facebookresearch/nevergrad/blob/master/docs/machinelearning.md for more.

Or if you want something more aimed at robustly outperforming random search:
- use `OrderedDiscrete` for discrete variables, taking care that the default value is in the middle.
- Use `ScrHammersleySearchPlusMiddlePoint` (`PlusMiddlePoint` only if you have continuous parameters or good default values for discrete parameters).

## Benchmarks

The benchmark tools aim at providing a way to evaluate optimizers on a large range of settings. They provide a way to run the optimizers on all settings and record the results, as well as ways to plot the results of the experiments.

### Creating data from experiments

Experiment plans are described in `nevergrad.benchmark.experiments`. Each experiment plan is a generator yielding different `Experiment` instances (defining function and optimizer settings).
To run an experiment plan, use:
```
python -m nevergrad.benchmark <experiment_name>
```
You may try with the experiment name `repeated_basic` for instance.
Check out `python -m nevergrad.benchmark -h` for all options. In particular, using `--num_workers=4` can speed up the computation by using 4 processes, and
for some experiments, you may provide a seed for reproducibility.

This benchmark command creates a file `<experiment_name>.csv` holding all information. The output file can be changed with the `--output` parameter. Each line of the csv file corresponds to one experiment from the experiment plan. The columns correspond to the function and optimizer settings for this experiment as well as its result (loss/regret value, elapsed time...).


### Plotting results

In order to plot the data in the csv file, just run:
```
python -m nevergrad.benchmark.plotting <csv_file>
```
Check out `python -m nevergrad.benchmark.plotting -h` for more options.

The plotting tool will create 2 types of plot:
- *experiment plots*: regret with respect to budget for each optimizer after averaging on all experiments with identical settings.
- *fight plots*: creates a matrix of size `num_algo x num_algo` where element at position `ij` is the mean winning rate of `algo_i` over `algo_j` in similar settings. The matrix is ordered to show best algorithms first, and only the first 5 rows (first 5 best algorithms) are displayed.


### Full pipeline

For simplicity's sake, you can directly plot from the `nevergrad.benchmark` command by adding a `--plot` argument. If a path is specified
afterwards, it will be used to save all figures, otherwise it will be saved a folder `<csvfilename>_plots` alongside the experiment csv file.


### Adding your own experiments and/or optimizers and/or functions
The `nevergrad.benchmark` command has an "import" system which can import additional modules where you defined your own experiments, possibly with your own functions and/or optimizers.

Example (please note that `nevergrad` needs to be cloned in your working directory for this example to work):
```
python -m nevergrad.benchmark additional_experiment --imports=nevergrad/benchmark/additional/example.py
```
See the [example file](nevergrad/benchmark/additional/example.py) to understand more precisely how functions/optimizers/experiments are specified. You can also submit a pull request to add your code directly in `nevergrad`. In this case, please refer to these [guidelines](docs/adding_an_algorithm.md).

Functions used for the experiments must derive from `nevergrad.functions.BaseFunction`. This abstract class helps you set up a description of your function settings through the `_descriptors` attribute,  which is used to create the columns of the data file produced by the experiments. See the docstrings for more information, and [functionlib.py](nevergrad/functions/functionlib.py) and [example.py](nevergrad/benchmark/additional/example.py) for examples.

If you want your experiment plan to be seedable, be extra careful as to how you handle randomness in the experiment generator, since each individual experiment may be run in any order. See [experiments.py](nevergrad/benchmark/experiments.py) for examples of seedable experiment plans. If you do not care for it. For simplicity's sake, the experiment plan generator is however not required to have a seed parameter (but will not be reproducible in this case).

## Instrumentation

**Please note that instrumentation is still a work in progress. We will try to update it to make it simpler and simpler to use (all feedbacks are welcome ;) ), with the side effect that their will be breaking changes (see Issues #44 to #47).**

**More specifically, the current description applies to the master branch but is not implemented in the current release (v0.1.3).**

The aim of instrumentation is to turn a piece of code with parameters you want to optimize into a function defined on an n-dimensional continuous data space in which the optimization can easily be performed. For this, discrete/categorial arguments must be transformed to continuous variables, and all variables concatenated. The instrumentation subpackage will help you do thanks to:
- the `variables` modules providing priors that can be used to define each argument.
- the `Instrumentation`, and `InstrumentedFunction` classes which provide an interface for converting any arguments into the data space used for optimization, and convert from data space back to the arguments space.
- the `FolderFunction` which helps transform any code into a Python function in a few lines. This can be especially helpful to optimize parameters in non-Python 3.6+ code (C++, Octave, etc...) or parameters in scripts.


### Variables

3 types of variables are currently provided:
- `SoftmaxCategorical`: converts a list of `n` (unordered) categorial variables into an `n`-dimensional space. The returned element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic and makes the function evaluation noisy.
- `OrderedDiscrete`: converts a list of (ordered) discrete variables into a 1-dimensional variable. The returned value will depend on the value on this dimension: low values corresponding to first elements of the list, and high values to the last.
- `Gaussian`: normalizes a `n`-dimensional variable with independent Gaussian priors (1-dimension per value).


### Instrumentation

Instrumentation helps you convert a set of arguments into variables in the data space which can be optimized. The core class performing this conversion is called `Instrumentation`. It provides arguments conversion through the `arguments_to_data` and `data_to_arguments` methods.


```python
from nevergrad import instrumentation as inst

# argument transformation
arg1 = inst.var.OrderedDiscrete(["a", "b"])  # 1st arg. = positional discrete argument
arg2 = inst.var.SoftmaxCategorical(["a", "c", "e"])  # 2nd arg. = positional discrete argument
value = inst.var.Gaussian(mean=1, std=2)  # the 4th arg. is a keyword argument with Gaussian prior

# create the instrumented function
instrum = inst.Instrumentation(arg1, arg2, "blublu", value=value)
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print(instrum.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete variable has 1 possible values, represented by a hard thresholding in
#   a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete variable has 3 possible values, represented by softmax, i.e. we add 3 coordinates to the continuous problem
# - the 3rd variable has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th variable is a real number, represented by single coordinate.


print(instrum.data_to_arguments([1, -80, -80, 80, 3]))
# prints (args, kwargs): (('b', 'e', 'blublu'), {'value': 7})
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))
# value=7 because 3 * std + mean = 7
```


For convenience and until a better way is implemented (see future notice), we provide an `InstrumentedFunction` class converting a function of any parameter space into the data space. Here is a basic example of its use:

**Future notice**: `InstrumentedFunction` may come to disappear (or at least we will discourage its use) when a new API for instrumenting on the optimizer side is ready.

```python

def myfunction(arg1, arg2, arg3, value=3):
    print(arg1, arg2, arg3)
    return value**2

# create the instrumented function using the "Instrumentation" instance above
ifunc = instrum.instrument(myfunction)
print(ifunc.dimension)  # 5 dimensional space as above
# you can still access the instrumentation instance will ifunc.instrumentation

ifunc([1, -80, -80, 80, 3])  # will print "b e blublu" and return 49 = 7**2
# check the instrumentation output explanation above if this is not clear
```

You can then directly perform optimization on this object:
```python
from nevergrad.optimization import optimizerlib
optimizer = optimizerlib.OnePlusOne(dimension=ifunc.dimension, budget=100)
recommendation = optimizer.optimize(ifunc)
```

When you have performed optimization on this function and want to trace back to what should your values be, use:
```python
recommendation = [1, -80, -80, 80, -.5]  # example of recommendation
# recover the arguments this way (don't forget deteriministic=True)
args, kwargs = ifunc.data_to_arguments(recommendation, deterministic=True)
print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 0} because -.5 * std + mean = 0

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), setting deterministic=False may yield different results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))
```



### External code instantiation

Sometimes it is completely impractical or impossible to have a simple Python3.6+ function to optimize. This may happen when the code you want to optimize is a script. Even more so if the code you want to optimize is not Python3.6+.

We provide tooling for this situation. Go through this steps to instrument your code:
 - **identify the variables** (parameters, constants...) you want to optimize.
 - **add placeholders** to your code. Placeholders are just tokens of the form `NG_ARG{name|comment}` where you can modify the name and comment. The name you set will be the one you will need to use as your function argument. In order to avoid breaking your code, the line containing the placeholders can be commented. To notify that the line should be uncommented for instrumentation, you'll need to add "@nevergrad@" at the start of the comment. Here is an example in C which will notify that we want to obtain a function with a `step` argument which will inject values into the `step_size` variable of the code:
```c
int step_size = 0.1
// @nevergrad@ step_size = NG_ARG{step|any comment}
```
- **prepare the command to execute** that will run your code. Make sure that the last printed line is just a float, which is the value to base the optimization upon. We will be doing minimization here, so this value must decrease for better results.
- **instantiate** your code into a function using the `FolderFunction` class:
```python
from nevergrad.instrumentation import FolderFunction
folder = "nevergrad/instrumentation/examples" # folder containing the code
command = ["python", "examples/script.py"]  # command to run from right outside the provided folder
func = FolderFunction(folder, command, clean_copy=True)
print(func.placeholders)  # will print the number of variables of the function
# prints: [Placeholder('value1', 'this is a comment'), Placeholder('value2', None), Placeholder('string', None)]
print(func(value1=2, value2=3, string="blublu"))
# prints: 12.0
```
- **instrument** the function, (see Instrumentation section just above).


Some important things to note:
 - using `FolderFunction` argument `clean_copy=True` will copy your folder so that tempering with it during optimization will run different versions of your code.
 - under the hood, with or without `clean_copy=True`, when calling the function, `FolderFunction` will create symlink copy of the initial folder, remove the files that have tokens, and create new ones with appropriate values. Symlinks are used in order to avoid duplicating large projects, but they have some drawbacks, see next point ;)
 - one can add a compilation step to `FolderFunction` (the compilation just has to be included in the script). However, be extra careful that if the initial folder contains some build files, they could be modified by the compilation step, because of the symlinks. Make sure that during compilation, you remove the build symlinks first! **This feature has not been fool proofed yet!!!**
 - the following external file types are registered by default: `[".c", ".h", ".cpp", ".hpp", ".py", ".m"]`. Custom file types can be registered using `instrumentation.register_file_type` by providing the relevant file suffix as well as the characters that indicate a comment. However, for now, variables which can provide a vector or values (`Gaussian` when providing a `shape`) will inject code with a Python format (list) by default, which may not be suitable.




## Dev setup

To install `nevergrad` in development mode (if you wish to contribute to it), clone the repository and run `python setup.py develop` from inside the repository folder.

Run tests with:
```
nosetests nevergrad --with-coverage --cover-package=nevergrad
```

You can also run type checking with:
```
mypy --ignore-missing-imports --strict nevergrad
```

Please use pre-commit hooks when commiting. This will run `pylint` and `autopep8` on the files you modified. You can install them with:
```
pre-commit install
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
