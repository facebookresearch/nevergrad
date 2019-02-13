# Benchmarks

The benchmark tools aim at providing a way to evaluate optimizers on a large range of settings. They provide a way to run the optimizers on all settings and record the results, as well as ways to plot the results of the experiments.

## Creating data from experiments

Experiment plans are described in `nevergrad.benchmark.experiments`. Each experiment plan is a generator yielding different `Experiment` instances (defining function and optimizer settings).
To run an experiment plan, use:
```
python -m nevergrad.benchmark <experiment_name>
```
You may try with the experiment name `repeated_basic` for instance.
Check out `python -m nevergrad.benchmark -h` for all options. In particular, using `--num_workers=4` can speed up the computation by using 4 processes, and
for some experiments, you may provide a seed for reproducibility.

This benchmark command creates a file `<experiment_name>.csv` holding all information. The output file can be changed with the `--output` parameter. Each line of the csv file corresponds to one experiment from the experiment plan. The columns correspond to the function and optimizer settings for this experiment as well as its result (loss/regret value, elapsed time...).


## Plotting results

In order to plot the data in the csv file, just run:
```
python -m nevergrad.benchmark.plotting <csv_file>
```
Check out `python -m nevergrad.benchmark.plotting -h` for more options.

The plotting tool will create 2 types of plot:
- *experiment plots*: regret with respect to budget for each optimizer after averaging on all experiments with identical settings.
- *fight plots*: creates a matrix of size `num_algo x num_algo` where element at position `ij` is the mean winning rate of `algo_i` over `algo_j` in similar settings. The matrix is ordered to show best algorithms first, and only the first 5 rows (first 5 best algorithms) are displayed.


## Full pipeline

For simplicity's sake, you can directly plot from the `nevergrad.benchmark` command by adding a `--plot` argument. If a path is specified
afterwards, it will be used to save all figures, otherwise it will be saved a folder `<csvfilename>_plots` alongside the experiment csv file.


## Adding your own experiments and/or optimizers and/or functions
The `nevergrad.benchmark` command has an "import" system which can import additional modules where you defined your own experiments, possibly with your own functions and/or optimizers.

Example (please note that `nevergrad` needs to be cloned in your working directory for this example to work):
```
python -m nevergrad.benchmark additional_experiment --imports=nevergrad/benchmark/additional/example.py
```
See the [example file](../nevergrad/benchmark/additional/example.py) to understand more precisely how functions/optimizers/experiments are specified. You can also submit a pull request to add your code directly in `nevergrad`. In this case, please refer to these [guidelines](adding_an_algorithm.md).

Functions used for the experiments must derive from `nevergrad.functions.BaseFunction`. This abstract class helps you set up a description of your function settings through the `_descriptors` attribute,  which is used to create the columns of the data file produced by the experiments. See the docstrings for more information, and [functionlib.py](../nevergrad/functions/functionlib.py) and [example.py](../nevergrad/benchmark/additional/example.py) for examples.

If you want your experiment plan to be seedable, be extra careful as to how you handle randomness in the experiment generator, since each individual experiment may be run in any order. See [experiments.py](../nevergrad/benchmark/experiments.py) for examples of seedable experiment plans. If you do not care for it. For simplicity's sake, the experiment plan generator is however not required to have a seed parameter (but will not be reproducible in this case).
