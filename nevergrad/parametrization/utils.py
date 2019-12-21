import typing as tp


class Descriptors(tp.NamedTuple):
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.
    """
    deterministic: bool = True
    deterministic_function: bool = True
    continuous: bool = True
    metrizable: bool = True


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """
