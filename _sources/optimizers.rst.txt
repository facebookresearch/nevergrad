Optimizers
==========

Optimizer API
-------------

.. autoclass:: nevergrad.optimizers.base.Optimizer
    :members:

Current implementations
-----------------------

.. Caution::
    Only non-family-based optimizers are listed in the documentation,
    you can get a full list of available optimizers with `sorted(nevergrad.optimizers.registry.keys())`

.. automodule:: nevergrad.optimization.optimizerlib
    :members:
    :undoc-members:

Configurable optimizers
-----------------------

Configurable optimizers share the following API to create optimizers instances:

.. autoclass:: nevergrad.optimizers.base.ParametrizedFamily
    :members:
    :special-members: __call__


.. automodule:: nevergrad.families
    :members:
