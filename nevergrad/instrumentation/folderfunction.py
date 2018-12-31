# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Optional, Any
from pathlib import Path
import numpy as np
from ..instrumentation.utils import CommandFunction
from .instantiate import InstrumentizedFolder


class FolderFunction:
    """Turns a folder into a parametrized function
    (with nevergrad tokens)

    Parameters
    ----------
    folder: Path/str
        path to the folder to instrumentize
    command: list
        command to run from inside the folder. The last line in stdout will
        be the output of the function.
        The command must be performed from just outside the instrumentized
        directory
    verbose: bool
        whether to print the run command and from where it is run.
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case)
    extension: tuple
        list of extensions for files to parametrize (files with dftokens)

    Returns
    -------
    the post-processed output of the called command

    Note
    ----
    By default, the postprocessing attribute holds a function which recovers the last line
    and converts it to float. The sequence of postprocessing can however be tampered
    with directly in order to change it

    Caution
    -------
        The clean copy is generally located in /tmp and may not be accessible for
        computation in a cluster. You may want to create a clean copy yourself
        in the folder of your choice, or set the the TemporaryDirectoryCopy class
        (located in instrumentation.instantiate) CLEAN_COPY_DIRECTORY environment
        variable to a shared directory
    """

    # pylint: disable=too-many-arguments
    def __init__(self, folder: Union[Path, str], command: List[str], verbose: bool = False, clean_copy: bool = False,
                 extensions: Optional[List[str]] = None) -> None:
        if extensions is None:
            extensions = [".py", "m", ".cpp", ".hpp", ".c", ".h"]
        self.command = command
        self.verbose = verbose
        self.postprocessings = [get_last_line_as_float]
        self.instrumentized_folder = InstrumentizedFolder(folder, extensions=extensions, clean_copy=clean_copy)
        self.last_full_output: Optional[str] = None

    @property
    def dimension(self) -> int:
        return self.instrumentized_folder.dimension

    def __call__(self, parameters: np.ndarray) -> Any:
        with self.instrumentized_folder.instantiate(parameters) as folder:
            if self.verbose:
                print(f"Running {self.command} from {folder.parent} which holds {folder}")
            output: Any = CommandFunction(self.command, cwd=folder.parent)()
        if self.verbose:
            print(f"FolderFunction recovered full output:\n{output}")
        self.last_full_output = output.strip()
        if not output:
            raise ValueError("No output")
        for postproc in self.postprocessings:
            output = postproc(output)
        if self.verbose:
            print(f"FolderFunction returns: {output}")
        return output

    def get_summary(self, parameters: np.ndarray) -> str:
        return self.instrumentized_folder.get_summary(parameters)


def get_last_line_as_float(output: str) -> float:
    split_output = output.strip().splitlines()
    return float(split_output[-1])
