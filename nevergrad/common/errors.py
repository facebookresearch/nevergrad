import unittest


# base classes


class NevergradError(Exception):
    """Base class for error raised by Nevergrad"""


class NevergradWarning(Warning):
    pass


# errors


class TellNotAskedNotSupportedError(NotImplementedError, NevergradError):
    """To be raised by optimizers which do not support the tell_not_asked interface."""


class ExperimentFunctionCopyError(NotImplementedError, NevergradError):
    """Raised when the experiment function fails to copy itself (for benchmarks)"""


class UnsupportedExperiment(RuntimeError, unittest.SkipTest, NevergradError):
    """Raised if the experiment is not compatible with the current settings:
    Eg: missing data, missing import, unsupported OS etc
    This automatically skips tests.
    """


# warnings


class NevergradDeprecationWarning(DeprecationWarning, NevergradWarning):
    """Deprecated function/class"""


class InefficientSettingsWarning(RuntimeWarning, NevergradWarning):
    """Optimization settings are not optimal for the optimizer"""


class BadLossWarning(RuntimeWarning, NevergradWarning):
    """Provided loss is unhelpful"""


class LossTooLargeWarning(BadLossWarning):
    """Sent when Loss is clipped because it is too large"""


class FinishedUnderlyingOptimizerWarning(RuntimeWarning, NevergradWarning):
    """Underlying scipy optimizer finished"""


class FailedConstraintWarning(RuntimeWarning, NevergradWarning):
    """Constraint could not be applied"""
