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
        monoobjective: bool = True,
        not_manyobjective: bool = True,
        continuous: bool = True,
        metrizable: bool = True,
        ordered: bool = True,
    ) -> None:
        self.deterministic = deterministic
        self.deterministic_function = deterministic_function
        self.continuous = continuous
        self.metrizable = metrizable
        self.ordered = ordered
        self.monoobjective = monoobjective
        self.not_manyobjective = not_manyobjective

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
                assert process.stdout is not None
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
