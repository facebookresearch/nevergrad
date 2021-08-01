.. _getting_started:

Getting started
===============

Installing
----------

Nevergrad is a Python 3.6+ library. It can be installed with:

.. code-block:: bash

    pip install nevergrad

You can also install the master branch instead of the latest release with:

.. code-block:: bash

    pip install git+https://github.com/facebookresearch/nevergrad@master#egg=nevergrad


A conda-forge version is also `available <https://github.com/conda-forge/nevergrad-feedstock>`_ thanks to @oblute:

.. code-block:: bash

    conda install -c conda-forge nevergrad


Alternatively, you can clone the repository and run :code:`pip install -e .` from inside the repository folder.

By default, this only installs requirements for the optimization and parametrization subpackages. If you are also interested in the benchmarking part,
you should install with the :code:`[benchmark]` flag (example: :code:`pip install nevergrad[benchmark]`), and if you also want the test tools, use
the :code:`[all]` flag (example: :code:`pip install -e .[all]`).

**Notes**:

- with :code:`zsh` you will need to run :code:`pip install 'nevergrad[all]'` instead of :code:`pip install nevergrad[all]`
- under Windows, you may need to preinstall torch (for :code:`benchmark` or :code:`all` installations) using Pytorch `installation instructions <https://pytorch.org/get-started/locally/>`_.

Installing on Windows
---------------------

For Windows installation, please refer to the `Windows documention <windows.html>`_.

Basic optimization example
--------------------------

**By default all optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation).**

Optimizing (minimizing!) a function using an optimizer (here :code:`NGOpt`, our adaptative optimization algorithm) can be easily run with:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_SIMPLEST_0
    :end-before: DOC_SIMPLEST_1


:code:`parametrization=n` is a shortcut to state that the function has only one variable, of dimension :code:`n`,
See the :ref:`parametrization tutorial <parametrizing>` for more complex parametrizations.

:code:`recommendation` holds the optimal value(s) found for the provided function. It can be
directly accessed through :code:`recommendation.value` which is here a :code:`np.ndarray` of size 2.

You can print the full list of optimizers with:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_OPT_REGISTRY_0
    :end-before: DOC_OPT_REGISTRY_1

The [optimization documentation](docs/optimization.md) contains more information on how to use several workers,
take full control of the optimization through the :code:`ask` and :code:`tell` interface, perform multiobjective optimization,
as well as pieces of advice on how to choose the proper optimizer for your problem.


Structure of the package
------------------------

The goals of this package are to provide:

- **gradient/derivative-free optimization algorithms**, including algorithms able to handle noise.
- **tools to parametrize any code**, making it painless to optimize your parameters/hyperparameters, whether they are continuous, discrete or a mixture of continuous and discrete parameters.
- **functions** on which to test the optimization algorithms.
- **benchmark routines** in order to compare algorithms easily.

The structure of the package follows its goal, you will therefore find subpackages:

- :code:`optimization`: implementing optimization algorithms
- :code:`parametrization`: specifying what are the parameters you want to optimize
- :code:`functions`: implementing both simple and complex benchmark functions
- :code:`benchmark`: for running experiments comparing the algorithms on benchmark functions
- :code:`common`: a set of tools used throughout the package
