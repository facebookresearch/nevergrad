Contributing to Nevergrad
#########################

General considerations
======================

We want to make contributing to this project as easy and transparent as possible.

Whether you want to contribute or not, don't hesitate to join `Nevergrad users' Facebook group <https://www.facebook.com/groups/nevergradusers/>`_


Our Development Process
-----------------------

To install :code:`nevergrad` in development mode (if you wish to contribute to it), clone the repository and run :code:`pip install -e .[all]` from inside the repository folder,
or :code:`pip install -e '.[all]'` if you use :code:`zsh`. If the install fails because of :code:`torch`, you can preinstall it with the instructions `on Pytorch website <https://pytorch.org/get-started/locally/>`_
and rerun :code:`nevergrad`'s installation.

You will also need to install :code:`numpy-stubs` with :code:`pip install git+https://github.com/numpy/numpy-stubs@master`


Most of the code is covered by unit tests. You can run them with:

.. code-block:: bash

    pytest nevergrad --cov=nevergrad

You can then run :code:`mypy` on :code:`nevergrad` with:

.. code-block:: bash

    mypy --implicit-reexport nevergrad

If you are not familiar with type checking, we do not want it to be an annoyance and you can can ignore errors by adding :code:`# type: ignore` at the end of lines flagged as incorrect.
If we consider it useful to have correct typing, we will update the code after your pull request is merged.
If you are however familiar with type hints, you can check with the strict mode: :code:`mypy --implicit-reexport --strict nevergrad`, but at any given moment all the code is not guaranteed to pass the test (we try to work on it regularly, see #409).

Unit tests and type checks (in non-strict mode) will be automatically run every time a pull request is submitted/updated.

Finally, we use pre-commit hooks to make sure the code follows the same coding style. We currently use  `black` and :code:`pylint`. To install them, just run :code:`pre-commit install` once, and they will be activated for all your commits on this repository.

:code:`black` compliance is automatically checked on all PRs. If you do not use precommit hooks, you
can install :code:`black` with :code:`pip install black` make your code compliant by running:

.. code-block:: bash

    black nevergrad


Pull Requests
-------------

We actively welcome your pull requests.

1. Fork the repo and create your branch from :code:`master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

Contributor License Agreement ("CLA")
-------------------------------------

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

Issues
------

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a `bounty program <https://www.facebook.com/whitehat/>`_ for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

Coding Style
------------

We use pep8, but allow lines to be as long as 140 characters.
Please use the pre-commit hooks to ensure correctness (see section "Our Development Process").

Documentation
-------------

Documentation can be build with :code:`make html` from the :code:`docs` folder.

License
-------

By contributing to :code:`nevergrad`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.


Adding an algorithm
===================

The following guidelines are for people who want to add an algorithm to :code:`nevergrad`. They may be already outdated, feel free to update them if you find them unclear or think they should evolve.

Where to add the algorithm?
---------------------------

All optimizers are implemented in the :code:`ng.optimization` subpackage, and all optimizer classes are available either in the :code:`ng.optimization.optimizerlib` module (which is aliased to :code:`ng.optimizers`, or through the optimizer registry: :code:`ng.optimizers.registry`.

Implementations are however spread into several files:

- `optimizerlib.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/optimizerlib.py>`_: this is the default file, where most algorithms are implemented. It also imports optimizers from all other files.
- `oneshot.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/oneshot.py>`_: this is where one-shot optimizers are implemented
- `differentialevolution.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/differentialevolution.py>`_: this is where differential evolution algorithms are implemented.
- `recastlib.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/recastlib.py>`_: this is where we implement ask & tell versions of existing Python implementations which do not follow this pattern. The underlying class which helps spawn a subprocess to run the existing implementation into is in `recaster.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/recaster.py>`_. Hopefully, you won't need this.

If you implement one new algorithm and if this algorithm is not one-shot/evolutionary/recast, you should implement it into `optimizerlib.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/optimizerlib.py>`_. If you implement a whole family of algorithms, you are welcome to create a new corresponding file.
Still, this structure is not final, it is bound to evolve and you are welcome to amend it.


How to implement it?
--------------------

Base class features
^^^^^^^^^^^^^^^^^^^

All algorithms derive from a base class named :code:`Optimizer` and are registered through a decorator. The implementation of the base class is `here <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/base.py>`_.
This base class implements the :code:`ask` and :code:`tell` interface.

It records a sample of the best evaluated points through the :code:`archive` attribute of class :code:`Archive`.  The archive can be seen be used as if it was of type
:code:`Dict[np.ndarray, Value]`, but since :code:`np.ndarray` are not hashable, the underlying implementation converts arrays into bytes and
register them into the :code:`archive.bytesdict` dictionary. :code:`Archive` however does not implement :code:`keys` and :code:`items` methods
because converting from bytes to array is not very efficient, one should therefore integrate on :code:`bytesdict` and the keys can then be
transformed back to arrays using :code:`np.frombuffer(key)`. See
`OnePlusOne implementation <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/optimizerlib.py>`_ for an example.


The key tuple if the point location, and :code:`Value` is a class with attributes:

- :code:`count`: number of evaluations at this point.
- :code:`mean`: mean value of the evaluations at this point.
- :code:`variance`: variance of the evaluations at this point.

For more details, see the implementation in `utils.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/utils.py>`_.

Through the archive, you can therefore access most useful information about past evaluations. A pruning mechanism makes sure this archive does
not grow too much. This pruning can be tuned through the :code:`pruning` attribute of the optimizer.
By default it keeps at least the best 100 points and cleans up when reaching 1000 points. It can be straightforwardly deactivated by setting optimizer's archive
attribute to :code:`None`.

The base :code:`Optimizer` class also tracks the best optimistic and pessimistic points through the :code:`current_bests` attribute which is of type:
:code:`Dict[str, Point]`. The key string is either :code:`optimistic` or :code:`pessimistic`, and the :code:`Point` value is a :code:`Value` with an additional :code:`x` attribute, recording the location of the point.

Methods and attributes
^^^^^^^^^^^^^^^^^^^^^^^

4 methods are designed to be overridden:

- :code:`__init__`: for the initialization of your algorithm
- :code:`_internal_ask_candidate`: to fetch the next point to be evaluated. This function is the only one that is absolutely required to be overridden. The default :code:`ask` method calls this method (please do not override the default :code:`ask`).
- :code:`_internal_tell_candidate`: to update your algorithm with the new point. The default :code:`tell` method calls this internal method after updating the archive (see paragraph above), please do not override it.
- :code:`_internal_provide_recommendation`: to provide the final recommendation. By default, the recommendation is the pessimistic best point.
- :code:`_internal_tell_not_asked` (optional): if the optimizer must handle points differently if they were not asked for, this method must be implemented. If you do not want to support this, you can raise :code:`base.TellNotAskedNotSupportedError`. A unit test will make sure that the optimizer either accepts the point or raises this error.

These functions work with :code:`Parameter` instances, which hold the parameter(s) :code:`value` (which can also be accessed through :code:`args` and :code:`kwargs`) depending on the parametrization provided at the initialization of the optimizer.
New instances of :code:`Parameter` can be easily created through the :code:`optimizer.parametrization.spawn_child()`. This way it keeps track of the
filiation between parameters. The value can then be updated either directly through the :code:`parameter.value` attribute, or by setting
the value in the "standardized space" (`parameter.set_standardized_data`).



If the algorithm is not able to handle parallelization (if :code:`ask` cannot be called multiple times consecutively), the :code:`no_parallelization` **class attribute** must be set to :code:`True`.


Seeding
^^^^^^^

Seeding has an important part for the significance and reproducibility of the algorithm benchmarking. We want to ensure the following constraints:

- we expect stochastic algorithms to be actually stochastic, if we set a hard seed inside the implementation this assumption is broken.
- we need the randomness to obtain relevant statistics when benchmarking the algorithms on deterministic functions.
- we should be able to seed from **outside** when we need it: we expect that setting a seed to the global random state should lead to reproducible results.

In order to facilitate these behaviors, each parametrization has a :code:`random_state` attribute (`np.random.RandomState`), which can be seeded by the
user if need be. :code:`optimizer._rng` is a shortcut to access it. All calls to stochastic functions should there be made through it.
By default, it will be seeded randomly by drawing a number from the global numpy random state so
that seeding the global numpy random state will yield reproducible results as well

A unit tests automatically makes sure that all optimizers have repeatable behaviors  on a simple test case when seeded from outside (see below).


About type hints
^^^^^^^^^^^^^^^^

We have used `type hints <https://docs.python.org/3/library/typing.html>`_ throughout :code:`nevergrad` to make it more robust, and the continuous integration will check that everything is correct when pull requests are submitted.
If you need to add base types for your code, please import them through :code:`import nevergrad.common.typing as tp`.
However, **we do not want typing to be an annoyance** for contributors who do not care about it, so please feel entirely free to use :code:`# type: ignore` on each line the continuous integration will flag as incorrect, so that the errors disappear. If we consider it useful to have correct typing, we will update the code after your pull request is merged.


Optimizer families
^^^^^^^^^^^^^^^^^^

If it makes sense to create several variations of your optimizer, using different hyperparameters, you can implement an :code:`OptimizerFamily`. The only aim of this class is to create :code:`Optimizers` and set the parameters before returning it. This is still an experimental API which may evolve soon, and an example can be found in the implementation of `differential evolution algorithms <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/differentialevolution.py>`_.

How to test it
--------------

You are welcome to add tests if you want to make sure your implementation is correct. It is however not required since some tests are run on all registered algorithms. They will test two features:

- that all algorithms are able to find the optimum of a simple 2-variable quadratic fitness function.
- that running the algorithms twice after setting a seed lead to the exact same recommendation. This is useful to make sure we will get repeatability in the benchmarks.

To run these tests, you can use:

.. code-block:: bash

    pytest nevergrad/optimization/test_optimizerlib.py

The repeatability test will however crash the first time you run it, since no value for the recommendation of your algorithm exists. This is automatically added when running the tests, and if everything goes well the second time you run them, it means everything is fine. You will see in you diff that an additional line was added to a file containing all expected recommendations.

If for any reason one of this test is not suitable for your algorithm, we'll discuss this in the pull request and decide of the appropriate workaround.

How to benchmark it
-------------------

Benchmarks are implemented in two files `experiments.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py>`_ and `frozenexperiments.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/frozenexperiments.py>`_.
While the former can be freely modified (benchmarks will be regularly added and removed), the latter file implements experiments which should not be modified when adding an algorithm, because they are used in tests, or for reproducibility of published results.

Providing some benchmark results along your pull requests will highlight the interest of your algorithm. It is however not required. For now, there is no standard approach for benchmarking your algorithm. You can implement your own benchmark, or copy an existing one and add your algorithm. Feel free to propose other solutions.

How benchmarks are implemented
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A benchmark is made of many :code:`Experiment` instances.  An :code:`Experiment` is basically the combination of a test function, and settings for the optimization (optimizer, budget, etc...).

Benchmarks are specified using a generator of :code:`Experiment` instances. See examples in `experiments.py <https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/experiments.py>`_. If you want to make sure your benchmark is perfectly reproducible, you will need to be careful of properly seeding the functions and/or the experiments.
