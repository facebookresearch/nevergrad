.. _benchmarking:

Running algorithm benchmarks
============================

The benchmark tools aim at providing derivative-free optimization researchers with a way to evaluate optimizers on a large range of settings. They provide ways to run the optimizers on all settings and record the results, as well as ways to plot the results of the experiments.

By default, :code:`nevergrad` does not come with all the requirements for the benchmarks to run. Make sure you have installed/updated it with either the :code:`benchmark` or the :code:`all` flag (example: :code:`pip install 'nevergrad[benchmark]'`) or you will miss some packages.

Creating data from experiments
------------------------------

Experiment plans are described in :code:`nevergrad.benchmark.experiments`. Each experiment plan is a generator yielding different :code:`Experiment` instances (defining function and optimizer settings).
To run an experiment plan, use:

.. code-block:: bash

    python -m nevergrad.benchmark <experiment_name>

You may try with the experiment name :code:`repeated_basic` for instance.
Check out :code:`python -m nevergrad.benchmark -h` for all options. In particular, using :code:`--num_workers=4` can speed up the computation by using 4 processes, and
for some experiments, you may provide a seed for reproducibility.

This benchmark command creates a file :code:`<experiment_name>.csv` holding all information. The output file can be changed with the :code:`--output` parameter. Each line of the csv file corresponds to one experiment from the experiment plan. The columns correspond to the function and optimizer settings for this experiment as well as its result (loss/regret value, elapsed time...).


Plotting results
----------------

In order to plot the data in the csv file, just run:

.. code-block:: bash

    python -m nevergrad.benchmark.plotting <csv_file>

Check out :code:`python -m nevergrad.benchmark.plotting -h` for more options.

The plotting tool will create 2 types of plot:

- *experiment plots*: regret with respect to budget for each optimizer after averaging on all experiments with identical settings.
- *fight plots*: creates a matrix of size :code:`num_algo x num_algo` where element at position :code:`ij` is the mean winning rate of :code:`algo_i` over :code:`algo_j` in similar settings. The matrix is ordered to show best algorithms first, and only the first 5 rows (first 5 best algorithms) are displayed.


Full pipeline
-------------

For simplicity's sake, you can directly plot from the :code:`nevergrad.benchmark` command by adding a :code:`--plot` argument. If a path is specified
afterwards, it will be used to save all figures, otherwise it will be saved a folder :code:`<csvfilename>_plots` alongside the experiment csv file.


Adding your own experiments and/or optimizers and/or test functions
-------------------------------------------------------------------

The :code:`nevergrad.benchmark` command has an "import" system which can import additional modules where you defined your own experiments, possibly with your own functions and/or optimizers.
This system however does not work on Windows (yet? feel free to help us!)

Example (please note that :code:`nevergrad` needs to be cloned in your working directory for this example to work):

.. code-block:: bash

    python -m nevergrad.benchmark additional_experiment --imports=nevergrad/benchmark/additional/example.py

See the `example file <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/additional/example.py>`_ to understand more precisely how functions/optimizers/experiments are specified. You can also submit a pull request to add your code directly in :code:`nevergrad`.
In this case, please refer to these [guidelines](adding_an_algorithm.md).

Functions used for the experiments must derive from :code:`nevergrad.functions.ExperimentFunction`. This class implements features necessary for the benchmarks:

- keeps the parametrization of the function, used for instantiating the optimizers.
- keeping a dictionary of descriptors of your function settings through the :code:`descriptors` attribute,  which is used to create the columns of the data file produced by the experiments.
  Descriptors are automatically filled up based on int/bool/float/str variables you used to initialize the function, but you can add more used :code:`add_descriptors`
- let's you override methods allowing custom behaviors such as :code:`evaluation_function` called at evaluation time to possibly avoid noise when possible, and :code:`compute_pseudotime` to mock computation time during benchkmarks.
- implements a :code:`copy` method for creating a new instance, using the same parameters you provided to initialize the function. The initilization of functions may draw random variables
  so that two copies can differ, providing more robustness to benchmarks.

See the docstrings for more information, and `arcoating/core.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/functions/arcoating/core.py>`_ and `example.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/additional/example.py>`_ for examples.

If you want your experiment plan to be seedable, be extra careful as to how you handle randomness in the experiment generator, since each individual experiment may be run in any order. See `experiments.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py>`_ for examples of seedable experiment plans. If you do not care for it. For simplicity's sake, the experiment plan generator is however not required to have a seed parameter (but will not be reproducible in this case).
