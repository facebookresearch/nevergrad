Optimizers API Reference
========================


Configurable optimizers
-----------------------

Configurable optimizers share the following API to create optimizers instances:

.. autoclass:: nevergrad.optimizers.base.ParametrizedFamily
    :members:
    :special-members: __call__


Here is a list of the available configurable optimizers:

.. automodule:: nevergrad.families
    :members:

Optimizers
----------

Here are all the other optimizers available in `nevergrad`:

.. Caution::
    Only non-family-based optimizers are listed in the documentation,
    you can get a full list of available optimizers with `sorted(nevergrad.optimizers.registry.keys())`

.. automodule:: nevergrad.optimization.optimizerlib
    :members:
    :undoc-members:


Optimizer API
-------------

All te optimizers share the following common API:

.. autoclass:: nevergrad.optimizers.base.Optimizer
    :members:

