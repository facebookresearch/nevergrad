# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import shutil
import tempfile
import warnings
import subprocess
from pathlib import Path
import numpy as np
from nevergrad.common import errors
from nevergrad.common import typing as tp


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


class FunctionInfo:  # Note: eventually, this should be a dataclass (dropping old Python support)
    """Information about the function

    Parameters
    ----------
    deterministic: bool
        whether the function equipped with its instrumentation is deterministic.
        Can be false if the function is not deterministic or if the instrumentation
        contains a softmax.
    proxy: bool
        whether the objective function is a proxy of a more interesting objective function.
    metrizable: bool
        whether the domain is naturally equipped with a metric.
    """

    def __init__(
        self,
        deterministic: bool = True,
        proxy: bool = False,
        metrizable: bool = True,
    ) -> None:
        self.deterministic = deterministic
        self.proxy = proxy
        self.metrizable = metrizable

    def __repr__(self) -> str:
        diff = ",".join(f"{x}={y}" for x, y in sorted(self.__dict__.items()))
        return f"{self.__class__.__name__}({diff})"


_WARNING = "parameter.descriptors is deprecated use {} instead"


class DeprecatedDescriptors:
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.

    Deprecated
    ----------
    This is replaced by ng.p.helpers.analyze(parameter), and parameter.function

    """

    _ANALYSIS_NAMES = ["deterministic", "continuous", "ordered"]

    # pylint: disable=too-many-arguments
    def __init__(self, param: tp.Any) -> None:
        self._param = param
        self._info: tp.Any = None

    def __getattr__(self, name: str) -> tp.Any:
        if name in self._ANALYSIS_NAMES:
            if self._info is None:
                from . import helpers  # pylint: disable=import-outside-toplevel

                self._info = helpers.analyze(self._param)
            warnings.warn(
                _WARNING.format(f"'ng.p.helpers.analyze(parameter).{name}'"),
                errors.NevergradDeprecationWarning,
            )
            return getattr(self._info, name)
        if name == "non_proxy_function":
            warnings.warn(
                _WARNING.format(f"'not parameter.function.{name}'"), errors.NevergradDeprecationWarning
            )
            return not self._param.function.proxy
        translation = dict(deterministic_function="deterministic", metrizable="metrizable")
        if name not in translation:
            return super().__getattr__(name)  # type: ignore
        warnings.warn(
            _WARNING.format(f"'parameter.function.{translation[name]}'"), errors.NevergradDeprecationWarning
        )
        return getattr(self._param.function, translation[name])

    def __setattr__(self, name: str, value: bool) -> None:
        if name in self._ANALYSIS_NAMES:
            raise RuntimeError(
                f"Setting {name} descriptor value is no longer supported, as "
                "this is now included in ng.p.helpers.analyze(parameter)"
            )
        if name == "non_proxy_function":
            self._param.function.proxy = not value
            warnings.warn(
                _WARNING.format(f"'not parameter.function.{name}'"), errors.NevergradDeprecationWarning
            )
            return
        translation = dict(deterministic_function="deterministic", metrizable="metrizable")
        if name in translation:
            setattr(self._param.function, translation[name], value)
            warnings.warn(
                _WARNING.format(f"'parameter.function.{translation[name]}'"),
                errors.NevergradDeprecationWarning,
            )
            return
        super().__setattr__(name, value)


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


class _ConstraintCompatibilityFunction:
    """temporary hack for "register_cheap_constraint", to be removed"""

    def __init__(self, func: tp.Callable[[tp.Any], tp.Loss]) -> None:
        self.func = func

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss:
        out = self.func((args, kwargs))
        print("calling", args, kwargs, "out =", out)
        return out
