How to perform optimization
===========================

**By default, all optimizers assume a centered and reduced prior at the beginning of the optimization (i.e. 0 mean and unitary standard deviation). They are however able to find solutions far from this initial prior.**

Basic example
-------------

Minimizing a function using an optimizer (here :code:`OnePlusOne`) can be easily run with:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_BASE_0
    :end-before: DOC_BASE_1

:code:`parametrization=n` is a shortcut to state that the function has only one variable, of dimension :code:`n`,
See the :ref:`Parametrization section <parametrizing>` for more complex parametrizations.

:code:`parametrization=n` is a shortcut to state that the function has only one variable, continuous, of dimension :code:`n`,
Defining the following parametrization instead will optimize on both :code:`x` (continuous, dimension 2) and :code:`y` (continuous, dimension 1).


.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_BASE_1
    :end-before: DOC_BASE_2

We can work in the discrete case as well, e.g. with the one-max function applied on :code:`{0,1,2,3,4,5,6}^10`:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_BASE_4
    :end-before: DOC_BASE_5


Using several workers
---------------------

Running the function evaluation in parallel with several workers is as easy as providing an executor:


.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_BASE_2
    :end-before: DOC_BASE_3

With :code:`batch_mode=True` it will ask the optimizer for :code:`num_workers` points to evaluate, run the evaluations, then update the optimizer with the :code:`num_workers` function outputs, and repeat until the budget is all spent. Since no executor is provided, the evaluations will be sequential. :code:`num_workers > 1` with no executor is therefore suboptimal but nonetheless useful for evaluation purpose (i.e. we simulate parallelism but have no actual parallelism). :code:`batch_mode=False` (steady state mode) will ask for a new evaluation whenever a worker is ready.

Ask and tell interface
----------------------

An *ask and tell* interface is also available. The 3 key methods for this interface are respectively:

- :code:`ask`: suggest a candidate on which to evaluate the function to optimize.
- :code:`tell`: for updated the optimizer with the value of the function for a candidate.
- :code:`provide_recommendation`: returns the candidate the algorithms considers the best.

For most optimization algorithms in the platform, they can be called in arbitrary order - asynchronous optimization is OK. Some algorithms (with class attribute :code:`no_parallelization=True` however do not support this.

The :code:`Parameter` class holds attribute :code:`value` which contain the actual value to evaluate through the function.

Here is a simpler example in the sequential case (this is what happens in the :code:`optimize` method for :code:`num_workers=1`):


.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_BASE_3
    :end-before: DOC_BASE_4

Please make sure that your function returns a float, and that you indeed want to perform minimization and not maximization ;)


Choosing an optimizer
---------------------

:code:`ng.optimizers.registry` is a :code:`dict` of all optimizers, so you :code:`ng.optimizers.OnePlusOne` is equivalent to :code:`ng.optimizers.registry["OnePlusOne"]`.
Also, **you can print the full list of optimizers** with:


.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_OPT_REGISTRY_0
    :end-before: DOC_OPT_REGISTRY_1


All algorithms have strengths and weaknesses. Questionable rules of thumb could be:

- :code:`NGOpt` is "meta"-optimizer which adapts to the provided settings (budget, number of workers, parametrization) and should therefore be a good default.
- :code:`TwoPointsDE` is excellent in many cases, including very high :code:`num_workers`.
- :code:`PortfolioDiscreteOnePlusOne` is excellent in discrete settings of mixed settings when high precision on parameters is not relevant; it's possibly a good choice for hyperparameter choice.
- :code:`OnePlusOne` is a simple robust method for continuous parameters with :code:`num_workers` < 8.
- :code:`CMA` is excellent for control (e.g. neurocontrol) when the environment is not very noisy (num_workers ~50 ok) and when the budget is large (e.g. 1000 x the dimension).
- :code:`TBPSA` is excellent for problems corrupted by noise, in particular overparameterized (neural) ones; very high :code:`num_workers` ok).
- :code:`PSO` is excellent in terms of robustness, high :code:`num_workers` ok.
- :code:`ScrHammersleySearchPlusMiddlePoint` is excellent for super parallel cases (fully one-shot, i.e. :code:`num_workers` = budget included) or for very multimodal cases (such as some of our MLDA problems); don't use softmax with this optimizer.
- :code:`RandomSearch` is the classical random search baseline; don't use softmax with this optimizer.


Telling non-asked points, or suggesting points
----------------------------------------------
There are two ways to inoculate information you already have about some points:

- :code:`optimizer.sugggest(*args, **kwargs)`: after suggesting a point, the next :code:`ask` will be a point
  with the provided :code:`value`.
- :code:`candidate = optimizer.parametrization.spawn_child(new_value=your_value)`  which you can then use to
  :code:`tell` the optimizer with the corresponding loss.

**Examples:**

- parametrized with an :code:`Array`:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_INOCULATION_0
    :end-before: DOC_INOCULATION_1

- parametrized with an :code:`ng.p.Instrumentation`

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_INOCULATION_1
    :end-before: DOC_INOCULATION_2


**Note:** some optimizers do not support such inoculation. Those will raise a :code:`TellNotAskedNotSupportedError`.

Adding callbacks
----------------

You can add callbacks to the :code:`ask` and :code:`tell` methods through the :code:`register_callback` method.
The functions/callbacks registered on :code:`ask` must have signature :code:`callback (optimizer)` and functions registered on :code:`tell` must have signature :code:`function(optimizer, candidate, value)`.

The example below shows a callback which prints :code:`candidate` and :code:`value` on :code:`tell`:

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 4
    :start-after: DOC_CALLBACK_0
    :end-before: DOC_CALLBACK_1


Two callbacks are available through :code:`ng.callbacks`, see the :ref:`callbacks module documentation <callbacks>`. 


Optimization with constraints
-----------------------------

Nevergrad has a mechanism for cheap constraints.
"Cheap" means that we do not try to reduce the number of calls to such constraints.
We basically repeat mutations until we get a satisfiable point.
Let us say that we want to minimize :code:`(x[0]-.5)**2 + (x[1]-.5)**2` under the constraint :code:`x[0] >= 1`.

.. literalinclude:: ../nevergrad/optimization/test_doc.py
    :language: python
    :dedent: 8
    :start-after: DOC_CONSTRAINED_0
    :end-before: DOC_CONSTRAINED_1

Optimizing machine learning hyperparameters
-------------------------------------------

When optimizing hyperparameters as e.g. in machine learning. If you don't know what variables (see :ref:`Parametrization <parametrizing>` to use:

- use :code:`Choice` for discrete variables
- use :code:`TwoPointsDE` with :code:`num_workers` equal to the number of workers available to you. See the :ref:`machine learning examples <machinelearning>` for more.

Or if you want something more aimed at robustly outperforming random search in highly parallel settings (one-shot):

- use :code:`TransitionChoice` for discrete variables, taking care that the default value is in the middle.
- Use :code:`ScrHammersleySearchPlusMiddlePoint` (:code:`PlusMiddlePoint` only if you have continuous parameters or good default values for discrete parameters).


Example of chaining, or inoculation, or initialization of an evolutionary algorithm
-----------------------------------------------------------------------------------

Chaining consists in running several algorithms in turn, information being forwarded from the first to the second and so on.
More precisely, the budget is distributed over several algorithms, and when an objective function value is computed, all algorithms are informed.

Here is how to create such optimizers:

.. code-block:: python

    # Running LHSSearch with budget num_workers and then DE:
    DEwithLHS = Chaining([LHSSearch, DE], ["num_workers"])

    # Runninng LHSSearch with budget the dimension and then DE:
    DEwithLHSdim = Chaining([LHSSearch, DE], ["dimension"])

    # Runnning LHSSearch with budget 30 and then DE:
    DEwithLHS30 = Chaining([LHSSearch, DE], [30])

    # Running LHS for 100 iterations, then DE for 60, then CMA:
    LHSthenDEthenCMA = Chaining([LHSSearch, DE, CMA], [100, 60])

We can then minimize as usual:

.. code-block:: python

    import nevergrad as ng

    def square(x):
        return sum((x - .5)**2)

    optimizer = DEwithLHS30(parametrization=2, budget=300)
    recommendation = optimizer.minimize(square)
    print(recommendation.value)
    >>> [0.50843113, 0.5104554]


Multiobjective minimization with Nevergrad
------------------------------------------

Multiobjective minimization is a **work in progress** in :code:`nevergrad`. It is:

 - **not stable**: the API may be updated at any time, hopefully to make it simpler and more intuitive.
 - **not robust**: there are probably corner cases we have not investigated yet.
 - **not scalable**: it is not yet clear how the current version will work with large number of losses, or large budget. For now the features have been implemented without time complexity considerations.
 - **not optimal**: this currently transforms multiobjective functions into monoobjective functions, hence losing some structure and making the function dynamic, which some optimizers are not designed to work on.

In other words, use it at your own risk ;) and provide feedbacks (both positive and negative) if you have any!


The initial API that was added into :code:`nevergrad` to work with multiobjective functions uses a function wrapper to convert them into monoobjective functions.
Let us minimize :code:`f1` and :code:`f2` (two objective functions) assuming that values above 2.5 are of no interest:

.. literalinclude:: ../nevergrad/functions/multiobjective/test_core.py
    :language: python
    :dedent: 4
    :start-after: DOC_MULTIOBJ_0
    :end-before: DOC_MULTIOBJ_1


We are currently working on an **new experimental API** allowing users to directly :code:`tell` the results as an array or list of floats. When this API is stabilized and proved to work, it will probably replace the older one. Here is an example on how to use it:

.. literalinclude:: ../nevergrad/optimization/multiobjective/test_core.py
    :language: python
    :dedent: 4
    :start-after: DOC_MULTIOBJ_OPT_0
    :end-before: DOC_MULTIOBJ_OPT_1

Note that `DE` and its variants have been updated to make use of the multi-objective losses [#789](https://github.com/facebookresearch/nevergrad/pull/789). This is a **preliminary** fix since the initial `DE` implementaton was ill-suited for this use case.


Reproducibility
---------------

Each parametrization has its own :code:`random_state` for generating random numbers. All optimizers pull from it when they require stochastic behaviors.
For reproducibility, this random state can be seeded in two ways:

- by setting :code:`numpy`'s global random state seed (:code:`np.random.seed(32)`) before the parametrization's first use. Indeed, when first used,
  the parametrization's random state is seeded with a seed drawn from the global random state.
- by manually seeding the parametrization random state (E.g.: :code:`parametrization.random_state.seed(12)` or :code:`optimizer.parametrization.random_state = np.random.RandomState(12)`)
