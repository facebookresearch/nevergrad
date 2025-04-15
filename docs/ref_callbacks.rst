Callbacks API
=============

.. _callbacks:

Callbacks can be registered through the :code:`optimizer.register_callback` for call on either :code:`ask` or :code:`tell` methods. Two of them are available through the
`ng.callbacks` namespace.

.. automodule:: nevergrad.callbacks
    :members: OptimizerDump, ParametersLogger, ProgressBar, EarlyStopping

