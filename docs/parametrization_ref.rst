.. _parametrization_ref:

Parametrization API reference
=============================

**Please note that parametrization is still a work in progress and changes are on their way (including for this documentation)! We are trying to update it to make it simpler and simpler to use (all feedbacks are welcome ;) ), with the side effect that there will be breaking changes.**

The aim of parametrization is to specify what are the parameters that the optimization should be performed upon.
The parametrization subpackage will help you do thanks to:

- the `parameter` modules (accessed by the shortcut `nevergrad.p`) providing classes that should be used to specify each parameter.
- the `FolderFunction` which helps transform any code into a Python function in a few lines. This can be especially helpful to optimize parameters in non-Python 3.6+ code (C++, Octave, etc...) or parameters in scripts.


Parameters
----------

Here are the current types of parameters currently provided:

.. autosummary::
    nevergrad.p.Array
    nevergrad.p.Scalar
    nevergrad.p.Log
    nevergrad.p.Dict
    nevergrad.p.Tuple
    nevergrad.p.Instrumentation
    nevergrad.p.Choice
    nevergrad.p.TransitionChoice

.. automodule:: nevergrad.p
    :members: Array, Scalar, Log, Dict, Tuple, Instrumentation, Choice, TransitionChoice
    :show-inheritance:
    :exclude-members: freeze, recombine, get_value_hash, mutate, satisfies_constraints, args, kwargs, sample


Parameter API
-------------

.. autoclass:: nevergrad.p.Parameter
    :members:
    :inherited-members:
