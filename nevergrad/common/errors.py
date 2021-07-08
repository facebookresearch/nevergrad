# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest


# base classes


class NevergradError(Exception):
    """Base class for error raised by Nevergrad"""


class NevergradWarning(Warning):
    pass


# errors
# pylint: disable=too-many-ancestors


class NevergradEarlyStopping(StopIteration, NevergradError):
    """Stops the minimization loop if raised"""


class NevergradRuntimeError(RuntimeError, NevergradError):
    """Runtime error raised by Nevergrad"""


class NevergradTypeError(TypeError, NevergradError):
    """Runtime error raised by Nevergrad"""


class NevergradValueError(ValueError, NevergradError):
    """Runtime error raised by Nevergrad"""


class NevergradNotImplementedError(NotImplementedError, NevergradError):
    """Not implemented functionality"""


class TellNotAskedNotSupportedError(NevergradNotImplementedError):
    """To be raised by optimizers which do not support the tell_not_asked interface."""


class ExperimentFunctionCopyError(NevergradNotImplementedError):
    """Raised when the experiment function fails to copy itself (for benchmarks)"""


class UnsupportedExperiment(unittest.SkipTest, NevergradRuntimeError):
    """Raised if the experiment is not compatible with the current settings:
    Eg: missing data, missing import, unsupported OS etc
    This automatically skips tests.
    """


class NevergradDeprecationError(NevergradRuntimeError):
    """Deprecated function/class"""


class UnsupportedParameterOperationError(NevergradRuntimeError):
    """This type of operation is not supported by the parameter"""


# warnings


class NevergradDeprecationWarning(DeprecationWarning, NevergradWarning):
    """Deprecated function/class"""


class NevergradRuntimeWarning(RuntimeWarning, NevergradWarning):
    """Runtime warning raise by nevergrad"""


class InefficientSettingsWarning(NevergradRuntimeWarning):
    """Optimization settings are not optimal for the optimizer"""


class BadLossWarning(NevergradRuntimeWarning):
    """Provided loss is unhelpful"""


class LossTooLargeWarning(BadLossWarning):
    """Sent when Loss is clipped because it is too large"""


class NevergradBehaviorChangesWarning(NevergradRuntimeWarning):
    """Notifies a change of behavior"""


class FinishedUnderlyingOptimizerWarning(NevergradRuntimeWarning):
    """Underlying scipy optimizer finished"""


class FailedConstraintWarning(NevergradRuntimeWarning):
    """Constraint could not be applied"""
