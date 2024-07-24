.. _machinelearning:

Examples - Nevergrad for machine learning
=========================================

Let us assume that you have defined an objective function as in:

.. code-block:: python

    def myfunction(lr, num_layers, arg3, arg4, other_anything):
        ...
        return -accuracy  # something to minimize

You should define how it must be instrumented, i.e. what are the arguments you want to optimize upon, and on which space they are defined. If you have both continuous and discrete parameters, you have a good initial guess, maybe just use :code:`TransitionChoice` for all discrete variables, :code:`Array` for all your continuous variables, and use :code:`PortfolioDiscreteOnePlusOne` as optimizer.


.. code-block:: python

    import nevergrad as ng
    # instrument learning rate and number of layers, keep arg3 to 3 and arg4 to 4
    lr = ng.p.Log(lower=0.0001, upper=1)  # log distributed between 0.001 and 1
    num_layers = ng.p.TransitionChoice([4, 5, 6])
    parametrization = ng.p.Instrumentation(lr, num_layers, 3., arg4=4)

Make sure :code:`parametrization.value` holds your initial guess. It is automatically populated, but can be updated manually (just set :code:`value` to what you want). For more details on parametrization, see the :ref:`parametrization section <parametrizing>`.

The fact that you use (ordered) discrete variables through :code:`TransitionChoice` is not a big deal because by nature :code:`PortfolioDiscreteOnePlusOne` will ignore the order. This algorithm is quite stable.

If you have more budget, a cool possibility is to use :code:`Choice` for all discrete variables and then apply :code:`TwoPointsDE`. You might also compare this to :code:`DE` (classical differential evolution). This might need a budget in the hundreds.

If you want to double-check that you are not worse than random search, you might use :code:`RandomSearch`.

If you want something fully parallel (the number of workers can be equal to the budget), then you might use :code:`ScrHammersleySearch`, which includes the discrete case. Then, you should use :code:`TransitionChoice` rather than :code:`Choice`. This does not have the traditional drawback of grid search and should still be more uniform than random. By nature :code:`ScrHammersleySearch` will deal correctly with :code:`TransitionChoice` type for discrete variables.

If you are optimizing weights in reinforcement learning, you might use :code:`TBPSA` (high noise) or :code:`CMA` (low noise).


Below are 3 examples :

 1. the optimization of continuous hyperparameters. It is also presented in an asynchronous setting. All other examples are based on the ask and tell interface, which can be synchronous or not but relies on the user for setting up asynchronicity.
 2. the optimization of mixed (continuous and discrete) hyperparameters.
 3. the optimization of parameters in a noisy setting, typically as in reinforcement learning.


Optimization of continuous hyperparameters with CMA, PSO, DE, Random and QuasiRandom
------------------------------------------------------------------------------------

Let's first define our test case:


.. code-block:: python

    import nevergrad as ng
    import numpy as np


    print("Optimization of continuous hyperparameters =========")


    def train_and_return_test_error(x):
        return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x])

    parametrization = ng.p.Array(shape=(300,))  # optimize on R^300

    budget = 1200  # How many trainings we will do before concluding.

    names = ["RandomSearch", "TwoPointsDE", "CMA", "PSO", "ScrHammersleySearch"]


We will compare several algorithms (defined in :code:`names`).
:code:`RandomSearch` is well known, :code:`ScrHammersleySearch` is a quasirandom; these two methods
are fully parallel, i.e. we can perform the 1200 trainings in parallel.
:code:`CMA` and :code:`PSO` are classical optimization algorithms, and :code:`TwoPointsDE`
is Differential Evolution equipped with a 2-points crossover.
A complete list is available in :code:`ng.optimizers.registry`.

Ask and tell version
^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    for name in names:
        optim = ng.optimizers.registry[name](parametrization=parametrization, budget=budget, num_workers=3)
        for u in range(budget // 3):
            x1 = optim.ask()
            # Ask and tell can be asynchronous.
            # Just be careful that you "tell" something that was asked.
            # Here we ask 3 times and tell 3 times in order to fake asynchronicity
            x2 = optim.ask()
            x3 = optim.ask()
            # The three folowing lines could be parallelized.
            # We could also do things asynchronously, i.e. do one more ask
            # as soon as a training is over.
            y1 = train_and_return_test_error(*x1.args, **x1.kwargs)  # here we only defined an arg, so we could omit kwargs
            y2 = train_and_return_test_error(*x2.args, **x2.kwargs)  # (keeping it here for the sake of consistency)
            y3 = train_and_return_test_error(*x3.args, **x3.kwargs)
            optim.tell(x1, y1)
            optim.tell(x2, y2)
            optim.tell(x3, y3)
        recommendation = optim.recommend()
        print("* ", name, " provides a vector of parameters with test error ",
              train_and_return_test_error(*recommendation.args, **recommendation.kwargs))

Asynchronous version with concurrent.futures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    from concurrent import futures

    for name in names:
        optim = ng.optimizers.registry[name](parametrization=instru, budget=budget)

        with futures.ThreadPoolExecutor(max_workers=optim.num_workers) as executor:  # the executor will evaluate the function in multiple threads
            recommendation = optim.minimize(train_and_return_test_error, executor=executor)
        print("* ", name, " provides a vector of parameters with test error ",
              train_and_return_test_error(*recommendation.args, **recommendation.kwargs))


Optimization of mixed (continuous and discrete) hyperparameters
---------------------------------------------------------------


Let's define our function:

.. code-block:: python

    import numpy as np

    # Let us define a function.
    def myfunction(arg1, arg2, arg3, value=3):
        return np.abs(value) + (1 if arg1 != "a" else 0) + (1 if arg2 != "e" else 0)

This function must then be instrumented in order to let the optimizer now what are the arguments:

.. code-block:: python

    import nevergrad as ng
    # argument transformation
    # Optimization of mixed (continuous and discrete) hyperparameters.
    arg1 = ng.p.TransitionChoice(["a", "b"])  # 1st arg. = positional discrete argument
    # We apply a softmax for converting real numbers to discrete values.
    arg2 = ng.p.Choice(["a", "c", "e"])  # 2nd arg. = positional discrete argument
    value = ng.p.Scalar(init=1.0).set_mutation(sigma=2)  # the 4th arg. is a keyword argument with Gaussian prior

    # create the parametrization
    # the 3rd arg. is a positional arg. which will be kept constant to "blublu"
    instru = ng.p.Instrumentation(arg1, arg2, "blublu", value=value)

    print(instru.dimension)  # 5 dimensional space

The dimension is 5 because:

- the 1st discrete var. has 1 possible values, represented by a hard thresholding in a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
- the 2nd discrete var. has 3 possible values, represented by softmax,   i.e. we add 3 coordinates to the continuous problem
- the 3rd var. has no uncertainty, so it does not introduce any coordinate in the continuous problem
- the 4th var. is a real number, represented by single coordinate.


.. code-block:: python

    instru.set_standardized_data([1, -80, -80, 80, 3])
    print(instru.args, instru.kwargs)
    >>> (('b', 'e', 'blublu'), {'value': 7.0})
    myfunction(*instru.args, **instru.kwargs)
    >>> 8.0

In this case:
- :code:`args[0] == "b"` because 1 > 0 (the threshold is 0 here since there are 2 values.
- :code:`args[1] == "e"` is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80)) = 1
- :code:`args[2] == "blublu"` because it is kept constant
- :code:`value == 7` because std * 3 + current_value = 2 * 3 + 1 = 7
The function therefore returns 7 + 1 = 8.


Then you can run the optimization as usual. :code:`PortfolioDiscreteOnePlusOne` is quite a natural choice when you have a good initial guess and a mix of discrete and continuous variables; in this case, it might be better to use :code:`TransitionChoice` rather than :code:`Choice`.  
`TwoPointsDE` is often excellent in the large scale case (budget in the hundreds).


.. code-block:: python

    import nevergrad as ng
    budget = 1200  # How many episode we will do before concluding.
    for name in ["RandomSearch", "ScrHammersleySearch", "TwoPointsDE", "PortfolioDiscreteOnePlusOne", "CMA", "PSO"]:
        optim = ng.optimizers.registry[name](parametrization=instru, budget=budget)
        for u in range(budget // 3):
            x1 = optim.ask()
            # Ask and tell can be asynchronous.
            # Just be careful that you "tell" something that was asked.
            # Here we ask 3 times and tell 3 times in order to fake asynchronicity
            x2 = optim.ask()
            x3 = optim.ask()
            # The three folowing lines could be parallelized.
            # We could also do things asynchronously, i.e. do one more ask
            # as soon as a training is over.
            y1 = myfunction(*x1.args, **x1.kwargs)  # here we only defined an arg, so we could omit kwargs
            y2 = myfunction(*x2.args, **x2.kwargs)  # (keeping it here for the sake of consistency)
            y3 = myfunction(*x3.args, **x3.kwargs)
            optim.tell(x1, y1)
            optim.tell(x2, y2)
            optim.tell(x3, y3)
        recommendation = optim.recommend()
        print("* ", name, " provides a vector of parameters with test error ",
              myfunction(*recommendation.args, **recommendation.kwargs))


Manual parametrization
^^^^^^^^^^^^^^^^^^^^^^

You always have the possibility to define your own parametrization inside your function (not recommended):

.. code-block:: python

    def softmax(x, possible_values=None):
        expx = [np.exp(x_ - max(x)) for x_ in x]
        probas = [e / sum(expx) for e in expx]
        return np.random.choice(len(x) if possible_values is None
                else possible_values, size=1, p=probas)


    def train_and_return_test_error_mixed(x):
        cx = [x_ - 0.1 for x_ in x[3:]]
        activation = softmax(x[:3], ["tanh", "sigmoid", "relu"])
        return np.linalg.norm(cx) + (1. if activation != "tanh" else 0.)

    parametrization = 10  # you can just provide the size of your input in this case

    #This version is bigger.
    def train_and_return_test_error_mixed(x):
        cx = x[:(len(x) // 2)]  # continuous part.
        presoftmax_values = x[(len(x) // 2):]  # discrete part.
        values_for_this_softmax = []
        dx = []
        for g in presoftmax:
            values_for_this_softmax += [g]
            if len(values_for_this_softmax) > 4:
                dx += softmax(values_for_this_softmax)
                values_for_this_softmax = []
        return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in cx]) + [
                1 if d != 1 else 0 for d in dx]

    parametrization = 300


Optimization of parameters for reinforcement learning
-----------------------------------------------------

We do not average evaluations over multiple episodes - the algorithm is in charge of averaging, if need be.
:code:`TBPSA`, based on population-control mechanisms, performs quite well in this case.

If you want to run Open AI Gym, see `One-line for learning state-of-the-art OpenAI Gym controllers with Nevergrad <https://docs.google.com/document/d/1noubQ_ZTZ4PZeQ1St7Asi1Af02q7k0nRoX_Pipu9ZKs/edit?usp=sharing/>`_

.. code-block:: python

    import nevergrad as ng
    import numpy as np

    # Similar, but with a noisy case: typically a case in which we train in reinforcement learning.
    # This is about parameters rather than hyperparameters. TBPSA is a strong candidate in this case.
    # We do *not* manually average over multiple evaluations; the algorithm will take care
    # of averaging or reevaluate whatever it wants to reevaluate.


    print("Optimization of parameters in reinforcement learning ===============")


    def simulate_and_return_test_error_with_rl(x, noisy=True):
        return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x]) + noisy * len(x) * np.random.normal()


    budget = 1200  # How many trainings we will do before concluding.


    for tool in ["TwoPointsDE", "RandomSearch", "TBPSA", "CMA", "NaiveTBPSA",
            "PortfolioNoisyDiscreteOnePlusOne"]:

        optim = ng.optimizers.registry[tool](parametrization=300, budget=budget)

        for u in range(budget // 3):
            # Ask and tell can be asynchronous.
            # Just be careful that you "tell" something that was asked.
            # Here we ask 3 times and tell 3 times in order to fake asynchronicity
            x1 = optim.ask()
            x2 = optim.ask()
            x3 = optim.ask()
            # The three folowing lines could be parallelized.
            # We could also do things asynchronously, i.e. do one more ask
            # as soon as a training is over.
            y1 = simulate_and_return_test_error_with_rl(*x1.args)
            y2 = simulate_and_return_test_error_with_rl(*x2.args)
            y3 = simulate_and_return_test_error_with_rl(*x3.args)
            optim.tell(x1, y1)
            optim.tell(x2, y2)
            optim.tell(x3, y3)

        recommendation = optim.recommend()
        print("* ", tool, " provides a vector of parameters with test error ",
              simulate_and_return_test_error_with_rl(*recommendation.args, noisy=False))


Examples from our external users
--------------------------------

Nevergrad is a plugin in `Hydra <https://hydra.cc/docs/plugins/nevergrad_sweeper/>`_ Facebook's parameter sweeping library.

Nevergrad is interfaced in `IOH Profiler <https://iohprofiler.liacs.nl/>`_, a tool from Univ. Leiden, CNRS, Sorbonne univ and Tel Hai college for profiling optimization algorithms.

Nevergrad is interfaced in `MixSimulator <https://github.com/Foloso/MixSimulator/>`_, a useful tool to get the optimal parameters for an electrical mix.


Nevergrad is also used for `Guiding latent image generation <https://github.com/facebookresearch/nevergrad/blob/main/docs/examples/guiding/Guiding%20image%20generation%20with%20Nevergrad.md/>`_.

And for `Increasing diversity in image generation <https://github.com/facebookresearch/nevergrad/blob/main/docs/examples/diversity/Diversity%20in%20image%20generation%20with%20Nevergrad.md/>`_.

And for the `Detection of fake images <https://github.com/facebookresearch/nevergrad/blob/main/docs/examples/lognormal/Lognormal%20mutations%20in%20Nevergrad.md/>`_.

And for `Retrofitting <https://github.com/facebookresearch/nevergrad/blob/main/docs/examples/retrofitting/Retrofitting%20with%20Nevergrad.md>`_.
