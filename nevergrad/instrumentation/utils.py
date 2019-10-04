# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import shutil
import tempfile
import subprocess
from typing import List, Any, Iterable, Tuple, Union, Optional, Dict
from pathlib import Path
import numpy as np
from ..common.typetools import ArrayLike
from .core2 import Variable


def split_data(data: ArrayLike, variables: Iterable[Variable]) -> List[np.ndarray]:
    """Splits data according to the data requirements of the variables
    """
    # this functions should be tested
    data = np.array(data).ravel()
    variables = list(variables)  # make sure it is not an iterator
    total = sum(t.dimension for t in variables)
    assert len(data) == total, f"Expected {total} values but got {len(data)}"
    splitted_data: List[np.ndarray] = []
    variables = list(variables)  # make sure it is not an iterator
    start, end = 0, 0
    for variable in variables:
        end = start + variable.dimension
        splitted_data.append(data[start: end])
        start = end
    assert end == len(data), f"Finished at {end} but expected {len(data)}"
    return splitted_data


def process_variables(variables: Iterable[Variable], data: ArrayLike,
                      deterministic: bool = True) -> Tuple[Any, ...]:
    # this function should be removed (but tests of split_data are currently
    # made through this function)
    variables = list(variables)
    splitted_data = split_data(data, variables)
    return tuple([variable.data_to_arguments(d, deterministic=deterministic)[0][0] for variable, d in zip(variables, splitted_data)])


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
    def set_clean_copy_environment_variable(cls, directory: Union[Path, str]) -> None:
        """Sets the CLEAN_COPY_DIRECTORY environment variable in
        order for subsequent calls to use this directory as base for the
        copies.
        """
        assert Path(directory).exists(), "Directory does not exist"
        os.environ[cls.key] = str(directory)

    # pylint: disable=redefined-builtin
    def __init__(self, source: Union[Path, str], dir: Optional[Union[Path, str]] = None) -> None:
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

    def __init__(self, command: List[str], verbose: bool = False, cwd: Optional[Union[str, Path]] = None,
                 env: Optional[Dict[str, str]] = None) -> None:
        if not isinstance(command, list):
            raise TypeError("The command must be provided as a list")
        self.command = command
        self.verbose = verbose
        self.cwd = None if cwd is None else str(cwd)
        self.env = env

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Call the cammand line with addidional arguments
        The keyword arguments will be sent as --{key}={val}
        The logs are bufferized. They will be printed if the job fails, or sent as output of the function
        Errors are provided with the internal stderr
        """
        # TODO make the following command more robust (probably fails in multiple cases)
        full_command = self.command + [str(x) for x in args] + ["--{}={}".format(x, y) for x, y in kwargs.items()]
        if self.verbose:
            print(f"The following command is sent: {full_command}")
        outlines: List[str] = []
        with subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              shell=False, cwd=self.cwd, env=self.env) as process:
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    outlines.append(line.decode().strip())  # type: ignore
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
