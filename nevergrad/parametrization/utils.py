# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import copy
import shutil
import bisect
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from nevergrad.common import errors
from nevergrad.common import typing as tp
from nevergrad.common import tools as ngtools


class BoundChecker:
    """Simple object for checking whether an array lies
    between provided bounds.

    Parameter
    ---------
    lower: float or None
        minimum value
    upper: float or None
        maximum value

    Note
    -----
    Not all bounds are necessary (data can be partially bounded, or not at all actually)
    """

    def __init__(self, lower: tp.BoundValue = None, upper: tp.BoundValue = None) -> None:
        self.bounds = (lower, upper)

    def __call__(self, value: np.ndarray) -> bool:
        """Checks whether the array lies within the bounds

        Parameter
        ---------
        value: np.ndarray
            array to check

        Returns
        -------
        bool
            True iff the array lies within the bounds
        """
        for k, bound in enumerate(self.bounds):
            if bound is not None:
                if np.any((value > bound) if k else (value < bound)):
                    return False
        return True


class Descriptors:
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.

    Parameters
    ----------
    deterministic: bool
        whether the function equipped with its instrumentation is deterministic.
        Can be false if the function is not deterministic or if the instrumentation
        contains a softmax.
    deterministic_function: bool
        whether the objective function is deterministic.
    non_proxy_function: bool
        whether the objective function is not a proxy of a more interesting objective function.
    continuous: bool
        whether the domain is entirely continuous.
    metrizable: bool
        whether the domain is naturally equipped with a metric.
    ordered: bool
        whether all domains and subdomains are ordered.
    """  # TODO add repr

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        deterministic: bool = True,
        deterministic_function: bool = True,
        non_proxy_function: bool = True,
        continuous: bool = True,
        metrizable: bool = True,
        ordered: bool = True,
    ) -> None:
        self.deterministic = deterministic
        self.deterministic_function = deterministic_function
        self.non_proxy_function = non_proxy_function
        self.continuous = continuous
        self.metrizable = metrizable
        self.ordered = ordered

    def __and__(self, other: "Descriptors") -> "Descriptors":
        values = {field: getattr(self, field) & getattr(other, field) for field in self.__dict__}
        return Descriptors(**values)

    def __repr__(self) -> str:
        diff = ",".join(
            f"{x}={y}"
            for x, y in sorted(ngtools.different_from_defaults(instance=self, check_mismatches=True).items())
        )
        return f"{self.__class__.__name__}({diff})"


class TemporaryDirectoryCopy(tempfile.TemporaryDirectory):  # type: ignore
    """Creates a full copy of a directory inside a temporary directory
    This class can be used as TemporaryDirectory but:
    - the created copy path is available through the copyname attribute
    - the contextmanager returns the clean copy path
    - the directory where the temporary directory will be created
      can be controlled through the CLEAN_COPY_DIRECTORY environment
      variable
    """

    key = "CLEAN_COPY_DIRECTORY"

    @classmethod
    def set_clean_copy_environment_variable(cls, directory: tp.Union[Path, str]) -> None:
        """Sets the CLEAN_COPY_DIRECTORY environment variable in
        order for subsequent calls to use this directory as base for the
        copies.
        """
        assert Path(directory).exists(), "Directory does not exist"
        os.environ[cls.key] = str(directory)

    # pylint: disable=redefined-builtin
    def __init__(self, source: tp.Union[Path, str], dir: tp.Optional[tp.Union[Path, str]] = None) -> None:
        if dir is None:
            dir = os.environ.get(self.key, None)
        super().__init__(prefix="tmp_clean_copy_", dir=dir)
        self.copyname = Path(self.name) / Path(source).name
        shutil.copytree(str(source), str(self.copyname))

    def __enter__(self) -> Path:
        super().__enter__()
        return self.copyname


class FailedJobError(RuntimeError):
    """Job failed during processing"""


class CommandFunction:
    """Wraps a command as a function in order to make sure it goes through the
    pipeline and notify when it is finished.
    The output is a string containing everything that has been sent to stdout

    Parameters
    ----------
    command: list
        command to run, as a list
    verbose: bool
        prints the command and stdout at runtime
    cwd: Path/str
        path to the location where the command must run from

    Returns
    -------
    str
       Everything that has been sent to stdout
    """

    def __init__(
        self,
        command: tp.List[str],
        verbose: bool = False,
        cwd: tp.Optional[tp.Union[str, Path]] = None,
        env: tp.Optional[tp.Dict[str, str]] = None,
    ) -> None:
        if not isinstance(command, list):
            raise TypeError("The command must be provided as a list")
        self.command = command
        self.verbose = verbose
        self.cwd = None if cwd is None else str(cwd)
        self.env = env

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> str:
        """Call the cammand line with addidional arguments
        The keyword arguments will be sent as --{key}={val}
        The logs are bufferized. They will be printed if the job fails, or sent as output of the function
        Errors are provided with the internal stderr
        """
        # TODO make the following command more robust (probably fails in multiple cases)
        full_command = (
            self.command + [str(x) for x in args] + ["--{}={}".format(x, y) for x, y in kwargs.items()]
        )
        if self.verbose:
            print(f"The following command is sent: {full_command}")
        outlines: tp.List[str] = []
        with subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=self.cwd,
            env=self.env,
        ) as process:
            try:
                assert process.stdout is not None
                for line in iter(process.stdout.readline, b""):
                    if not line:
                        break
                    outlines.append(line.decode().strip())
                    if self.verbose:
                        print(outlines[-1], flush=True)
            except Exception:  # pylint: disable=broad-except
                process.kill()
                process.wait()
                raise FailedJobError("Job got killed for an unknown reason.")
            stderr = process.communicate()[1]  # we already got stdout
            stdout = "\n".join(outlines)
            retcode = process.poll()
            if stderr and (retcode or self.verbose):
                print(stderr.decode(), file=sys.stderr)
            if retcode:
                subprocess_error = subprocess.CalledProcessError(
                    retcode, process.args, output=stdout, stderr=stderr
                )
                raise FailedJobError(stderr.decode()) from subprocess_error
        return stdout


X = tp.TypeVar("X")


class Subobjects(tp.Generic[X]):
    """Identifies suboject of a class and applies
    functions recursively on them.

    Parameters
    ----------
    object: Any
        an object containing other (sub)objects
    base: Type
        the base class of the subobjects (to filter out other items)
    attribute: str
        the attribute containing the subobjects

    Note
    ----
    The current implementation is rather inefficient and could probably be
    improved a lot if this becomes critical
    """

    def __init__(self, obj: X, base: tp.Type[X], attribute: str) -> None:
        self.obj = obj
        self.cls = base
        self.attribute = attribute

    def new(self, obj: X) -> "Subobjects[X]":
        """Creates a new instance with same configuratioon
        but for a new object.
        """
        return Subobjects(obj, base=self.cls, attribute=self.attribute)

    def items(self) -> tp.Iterator[tp.Tuple[tp.Any, X]]:
        """Returns a dict {key: subobject}"""
        container = getattr(self.obj, self.attribute)
        if not isinstance(container, (list, dict)):
            raise TypeError("Subcaller only work on list and dict")
        iterator = enumerate(container) if isinstance(container, list) else container.items()
        for key, val in iterator:
            if isinstance(val, self.cls):
                yield key, val

    def _get_subobject(self, obj: X, key: tp.Any) -> tp.Any:
        """Returns the corresponding subject if obj is from the
        base class, or directly the object otherwise.
        """
        if isinstance(obj, self.cls):
            return getattr(obj, self.attribute)[key]
        return obj

    def apply(self, method: str, *args: tp.Any, **kwargs: tp.Any) -> tp.Dict[tp.Any, tp.Any]:
        """Calls the named method with the provided input parameters (or their subobjects if
        from the base class!) on the subobjects.
        """
        outputs: tp.Dict[tp.Any, tp.Any] = {}
        for key, subobj in self.items():
            subargs = [self._get_subobject(arg, key) for arg in args]
            subkwargs = {k: self._get_subobject(kwarg, key) for k, kwarg in kwargs.items()}
            outputs[key] = getattr(subobj, method)(*subargs, **subkwargs)
        return outputs


def float_penalty(x: tp.Union[bool, float]) -> float:
    """Unifies penalties as float (bool=False becomes 1).
    The value is positive for unsatisfied penality else 0.
    """
    if isinstance(x, (bool, np.bool_)):
        return float(not x)  # False ==> 1.
    elif isinstance(x, (float, np.float)):
        return -min(0, x)  # Negative ==> >0
    raise TypeError(f"Only bools and floats are supported for check constaint, but got: {x} ({type(x)})")


L = tp.TypeVar("L", bound="Layered")


class Layered:
    """Hidden API for overriding/modifying the behavior of a Parameter,
    which is itself a Layered object.

    Layers can be added and will be ordered depending on their level
    0: root
    1-10: bounds
    11-20: casting
    21-30: casting
    """

    _LAYER_LEVEL = 1.0

    def __init__(self) -> None:
        self._layers = [self]
        self._index = 0
        self._name: tp.Optional[str] = None

    def _get_layer_index(self) -> int:
        print("self", self.name)
        print("index", self._index)
        print("layers", [l.name for l in self._layers])
        if self._layers[self._index] is not self:
            layers = [f"{l.name}({l._index})" for l in self._layers]
            raise errors.NevergradRuntimeError(
                "Layer indexing has changed for an unknown reason. Please open an issue:\n"
                f"Caller at index {self._index}: {self.name}"
                f"Layers: {layers}.\n"
            )
        return self._index

    def _get_value(self) -> tp.Any:
        index = self._get_layer_index()
        if not index:  # root must have an implementation
            raise NotImplementedError
        print(f"getting {index - 1} from {index}")
        return self._layers[index - 1]._get_value()

    def _set_value(self, value: tp.Any) -> tp.Any:
        index = self._get_layer_index()
        if not index:  # root must have an implementation
            raise NotImplementedError
        self._layers[index - 1]._set_value(value)

    def _del_value(self) -> tp.Any:
        pass

    def add_layer(self: L, other: "Layered") -> L:
        """Adds a layer which will modify the object behavior"""
        if self is not self._layers[0] or self._LAYER_LEVEL:
            raise errors.NevergradRuntimeError("Layers can only be added from the root.")
        if len(other._layers) > 1:
            raise errors.NevergradRuntimeError("Cannot append multiple layers at once")
        print(f"Inserting {other.name}")
        if other._LAYER_LEVEL >= self._layers[-1]._LAYER_LEVEL:
            print("ordered")
            other._index = len(self._layers)
            self._layers.append(other)
        else:
            print("unordered")
            ind = bisect.bisect_right([x._LAYER_LEVEL for x in self._layers], other._LAYER_LEVEL)
            self._layers.insert(ind, other)
            for k, x in enumerate(self._layers):
                x._index = k
        other._layers = self._layers
        return self

    def copy(self: L) -> L:
        """Creates a new unattached layer with the same behavior"""
        new = copy.copy(self)
        new._layers = [new]
        new._index = 0
        return new

    # naming capacity

    def _get_name(self) -> str:
        """Internal implementation of parameter name. This should be value independant, and should not account
        for internal/model parameters.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        """Name of the parameter
        This is used to keep track of how this Parameter is configured (included through internal/model parameters),
        mostly for reproducibility A default version is always provided, but can be overriden directly
        through the attribute, or through the set_name method (which allows chaining).
        """
        if self._name is not None:
            return self._name
        return self._get_name()

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)  # with_name allows chaining

    def set_name(self: L, name: str) -> L:
        """Sets a name and return the current instrumentation (for chaining)

        Parameters
        ----------
        name: str
            new name to use to represent the Parameter
        """
        self._name = name
        return self


class ValueProperty(tp.Generic[X]):
    """Typed property (descriptor) object so that the value attribute of
    Parameter objects fetches _get_value and _set_value methods
    """

    # This uses the descriptor protocol, like a property:
    # See https://docs.python.org/3/howto/descriptor.html
    #
    # Basically parameter.value calls parameter.value.__get__
    # and then parameter._get_value
    def __init__(self) -> None:
        self.__doc__ = """Value of the Parameter, which should be sent to the function
        to optimize.

        Example
        -------
        >>> ng.p.Array(shape=(2,)).value
        array([0., 0.])
        """

    def __get__(self, obj: Layered, objtype: tp.Optional[tp.Type[object]] = None) -> X:
        return obj._layers[-1]._get_value()  # type: ignore

    def __set__(self, obj: Layered, value: X) -> None:
        obj._layers[-1]._set_value(value)

    def __delete__(self, obj: Layered) -> None:
        for layer in obj._layers:
            layer._del_value()
