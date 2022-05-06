Optimizers API Reference
========================

Optimizer API
-------------

All the optimizers share the following common API:

.. autoclass:: nevergrad.optimizers.base.Optimizer
    :members:



.. _callbacks:

Callbacks
---------

Callbacks can be registered through the :code:`optimizer.register_callback` for call on either :code:`ask` or :code:`tell` methods. 
Callbacks just need to be functions with parameters depending on what they are registered upon:

- on :code:`ask` they only take the optimizer as input. The callback is called at the beginning of the :code:`ask` method.
- on :code:`tell` they take the optimizer as input, as well as the candidate and the loss as a float. 
  The callback is called at the end of the :code:`tell` method.

Some predefined callbacks are available through the `ng.callbacks` namespace:

.. automodule:: nevergrad.callbacks
    :members: OptimizerDump, ParametersLogger, ProgressBar, EarlyStopping

Configurable optimizers
-----------------------

Configurable optimizers share the following API to create optimizers instances:

.. autoclass:: nevergrad.optimizers.base.ConfiguredOptimizer
    :members:
    :special-members: __call__


Here is a list of the available configurable optimizers:

.. automodule:: nevergrad.families
    :members:

Optimizers
----------

Here are all the other optimizers available in :code:`nevergrad`:

.. Caution::
    Only non-family-based optimizers are listed in the documentation,
    you can get a full list of available optimizers with :code:`sorted(nevergrad.optimizers.registry.keys())`

.. automodule:: nevergrad.optimization.optimizerlib
    :members:
    :undoc-members:
