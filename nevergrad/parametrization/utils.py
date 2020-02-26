# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import shutil
import tempfile
import subprocess
import typing as tp
from pathlib import Path
import numpy as np
from nevergrad.common import tools as ngtools


class Descriptors:
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.
    """  # TODO add repr

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        deterministic: bool = True,
        deterministic_function: bool = True,
        continuous: bool = True,
        metrizable: bool = True,
        ordered: bool = True,
    ) -> None:
        self.deterministic = deterministic
        self.deterministic_function = deterministic_function
        self.continuous = continuous
        self.metrizable = metrizable
        self.ordered = ordered

    def __and__(self, other: "Descriptors") -> "Descriptors":
        values = {field: getattr(self, field) & getattr(other, field) for field in self.__dict__}
        return Descriptors(**values)

    def __repr__(self) -> str:
        diff = ",".join(f"{x}={y}" for x, y in sorted(ngtools.different_from_defaults(instance=self, check_mismatches=True).items()))
        return f"{self.__class__.__name__}({diff})"


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """


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
    """Job failed during processing
    """


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

    def __init__(self, command: tp.List[str], verbose: bool = False, cwd: tp.Optional[tp.Union[str, Path]] = None,
                 env: tp.Optional[tp.Dict[str, str]] = None) -> None:
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
        full_command = self.command + [str(x) for x in args] + ["--{}={}".format(x, y) for x, y in kwargs.items()]
        if self.verbose:
            print(f"The following command is sent: {full_command}")
        outlines: tp.List[str] = []
        with subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=False, cwd=self.cwd, env=self.env) as process:
            try:
                for line in iter(process.stdout.readline, b''):
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
                subprocess_error = subprocess.CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
                raise FailedJobError(stderr.decode()) from subprocess_error
        return stdout


class Mutation:
    """Custom mutation or recombination
    This is an experimental API

    Either implement:
    - `_apply_array`  which provides a new np.ndarray from a list of arrays
    - `apply` which updates the first p.Array instance

    Mutation should take only one p.Array instance as argument, while
    Recombinations should take several
    """

    def __repr__(self) -> str:
        diff = ngtools.different_from_defaults(instance=self, check_mismatches=True)
        params = ", ".join(f"{x}={y!r}" for x, y in sorted(diff.items()))
        return f"{self.__class__.__name__}({params})"

    def apply(self, arrays: tp.Sequence[tp.Any]) -> tp.Any:  # avoiding circular imports... restructuring needed eventually
        new_value = self._apply_array([a._value for a in arrays], arrays[0].random_state)
        arrays[0]._value = new_value

    def _apply_array(self, arrays: tp.Sequence[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
        raise RuntimeError("Mutation._apply_array should either be implementer or bypassed in Mutation.apply")
        return np.array([])  # pylint: disable=unreachable


class Crossover(Mutation):
    """Operator for merging part of an array into another one

    Parameters
    ----------
    axis: None or int or tuple of ints
        the axis (or axes) on which the merge will happen. This axis will be split into 3 parts: the first and last one will take
        value from the first array, the middle one from the second array.
    max_size: None or int
        maximum size of the part taken from the second array. By default, this is at most around half the number of total elements of the
        array to the power of 1/number of axis.


    Notes
    -----
    - this is experimental, the API may evolve
    - when using several axis, the size of the second array part is the same on each axis (aka a square in 2D, a cube in 3D, ...)

    Examples:
    ---------
    - 2-dimensional array, with crossover on dimension 1:
      0 1 0 0
      0 1 0 0
      0 1 0 0
    - 2-dimensional array, with crossover on dimensions 0 and 1:
      0 0 0 0
      0 1 1 0
      0 1 1 0
    """

    def __init__(self, axis: tp.Optional[tp.Union[int, tp.Iterable[int]]] = None, max_size: tp.Optional[int] = None) -> None:
        self.axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None
        self.max_size = max_size

    def _apply_array(self, arrays: tp.Sequence[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
        # checks
        arrays = list(arrays)
        if len(arrays) != 2:
            raise Exception("Crossover can only be applied between 2 individuals")
        shape = arrays[0].shape
        assert shape == arrays[1].shape, "Individuals should have the same shape"
        # settings
        axis = tuple(range(len(shape))) if self.axis is None else self.axis
        max_size = int(((arrays[0].size + 1) / 2)**(1 / len(axis))) if self.max_size is None else self.max_size
        max_size = min(max_size, *(shape[a] - 1 for a in axis))
        size = 1 if max_size == 1 else rng.randint(1, max_size)
        # slices
        slices = []
        for a, s in enumerate(shape):
            if a in axis:
                if s <= 1:
                    raise ValueError("Cannot crossover an shape with size 1")
                start = rng.randint(s - size)
                slices.append(slice(start, start + size))
            else:
                slices.append(slice(0, s))
        result = np.array(arrays[0], copy=True)
        result[tuple(slices)] = arrays[1][tuple(slices)]
        return result


class Rolling(Mutation):

    def __init__(self, axis: tp.Optional[tp.Union[int, tp.Iterable[int]]]):
        self.axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None

    def _apply_array(self, arrays: tp.Sequence[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
        arrays = list(arrays)
        assert len(arrays) == 1
        data = arrays[0]
        if rng is None:
            rng = np.random.RandomState()
        axis = tuple(range(data.dim)) if self.axis is None else self.axis
        shifts = [rng.randint(data.shape[a]) for a in axis]
        return np.roll(data, shifts, axis=axis)  # type: ignore
