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

.. automodule:: nevergrad.optimizers
    :members:
    :undoc-members:

Families of optimizers
----------------------

Parametrized family API
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: nevergrad.optimizers.base.ParametrizedFamily
    :members:
    :special-members: __call__

Current families
^^^^^^^^^^^^^^^^

.. automodule:: nevergrad.families
    :members:
