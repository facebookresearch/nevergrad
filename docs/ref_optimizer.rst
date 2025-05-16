
Optimization API (ng.optimizers)
================================


All the optimizers share the following common API:

.. autoclass:: nevergrad.optimizers.base.Optimizer
    :members:

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
