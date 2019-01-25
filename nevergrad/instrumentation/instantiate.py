# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import tempfile
import operator
import contextlib
from pathlib import Path
from typing import Union, List, Any, Optional, Generator, Callable, Tuple, Dict
import numpy as np
from ..functions import base
from ..common.typetools import ArrayLike
from . import utils
from . import variables


BIG_NUMBER = 3000
LINETOKEN = "@nevergrad" + "@"  # Do not trigger an error when parsing this file...
COMMENT_CHARS = {".c": "//", ".h": "//", ".cpp": "//", ".hpp": "//", ".py": "#", ".m": "%"}


def register_file_type(suffix: str, comment_chars: str) -> None:
    """Register a new file type to be used for token instrumentation by providing the relevant file suffix as well as
    the characters that indicate a comment."""
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    COMMENT_CHARS[suffix] = comment_chars


def symlink_folder_tree(folder: Union[Path, str], shadow_folder: Union[Path, str]) -> None:
    """Utility for copying the tree structure of a folder and symlinking all files
    This can help creating lightweight copies of a project, for instantiating several
    copies with different parameters.
    """
    folder, shadow_folder = (Path(x).expanduser().absolute() for x in (folder, shadow_folder))
    for fp in folder.glob("**/*"):
        shadow_fp = shadow_folder / fp.relative_to(folder)
        if fp.is_file() and not shadow_fp.exists():
            os.makedirs(str(shadow_fp.parent), exist_ok=True)
            shadow_fp.symlink_to(fp)


def uncomment_line(line: str, extension: str) -> str:
    if extension not in COMMENT_CHARS:
        raise RuntimeError(f'Unknown file type: {extension}\nDid you register it using {register_file_type.__name__}?')
    pattern = r'^(?P<indent> *)'
    pattern += r'(?P<linetoken>' + COMMENT_CHARS[extension] + r" *" + LINETOKEN + r" *)"
    pattern += r'(?P<command>.*)'
    lineseg = re.search(pattern, line)
    if lineseg is not None:
        line = lineseg.group("indent") + lineseg.group("command")
    if LINETOKEN in line:
        raise RuntimeError(f"Uncommenting failed for line of {extension} file (a {LINETOKEN} tag remains):\n{line}\n"
                           f"Did you follow the pattern indent+comment+{LINETOKEN}+code (with nothing before the indent)?")
    return line


class FileTextFunction:
    """Function created from a file and generating the text file after
    replacement of the placeholders
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        assert filepath.exists(), "{filepath} does not exist"
        with filepath.open("r") as f:
            text = f.read()
        if LINETOKEN in text:
            lines = text.splitlines()
            ext = filepath.suffix.lower()
            lines = [(l if LINETOKEN not in l else uncomment_line(l, ext)) for l in lines]
            text = "\n".join(lines)
        self.placeholders = utils.Placeholder.finditer(text)
        # TODO check no repeat
        self._text = text
        self.parameters = {x.name for x in self.placeholders}

    def __call__(self, **kwargs):
        unexpected, missing = set(kwargs) - self.parameters, self.parameters - set(kwargs)
        if unexpected or missing:
            raise ValueError(f"Found unexpected arguments: {unexpected}\n and/or missing arguments {missing}.")
        return utils.Placeholder.sub(self._text, **kwargs)

    def __repr__(self):
        names = sorted(self.parameters)
        return f"{self.__class__.__name__}({self.filepath})({', '.join(names)})"


class FolderInstantiator:
    """Folder with instrumentation tokens, which can be instantiated.

    Parameters
    ----------
    folder: str/Path
        the instrumented folder to instantiate
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case).

    Caution
    -------
        The clean copy is generally located in /tmp and may not be accessible for
        computation in a cluster. You may want to create a clean copy yourself
        in the folder of your choice, or set the the TemporaryDirectoryCopy class
        (located in instrumentation.instantiate) CLEAN_COPY_DIRECTORY environment
        variable to a shared directory
    """

    def __init__(self, folder: Union[Path, str], clean_copy: bool = False) -> None:
        self._clean_copy = None
        self.folder = Path(folder).expanduser().absolute()
        assert self.folder.exists(), f"{folder} does not seem to exist"
        if clean_copy:
            self._clean_copy = utils.TemporaryDirectoryCopy(str(folder))
            self.folder = self._clean_copy.copyname
        self.file_functions: List[FileTextFunction] = []
        names = set()
        for fp in self.folder.glob("**/*"):  # TODO filter out all hidden files
            if fp.is_file() and fp.suffix.lower() in COMMENT_CHARS:
                file_func = FileTextFunction(fp)
                fnames = {ph.name for ph in file_func.placeholders}
                if fnames:
                    if fnames & names:
                        raise RuntimeError(f"Found {fp} placeholders in another file: {fnames & names}")
                    self.file_functions.append(file_func)
        assert self.file_functions, "Found no file with placeholders"
        self.file_functions = sorted(self.file_functions, key=operator.attrgetter("filepath"))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.folder}") with files:\n{self.file_functions}'

    @property
    def placeholders(self) -> List[utils.Placeholder]:
        return [p for f in self.file_functions for p in f.placeholders]

    def instantiate_to_folder(self, outfolder: Union[Path, str], **kwargs: Any) -> None:  # TODO change API to avoid outfolder
        # TODO check argument list as in file function
        outfolder = Path(outfolder).expanduser().absolute()
        assert outfolder != self.folder, "Do not instantiate on same folder!"
        symlink_folder_tree(self.folder, outfolder)
        for file_func in self.file_functions:
            inst_fp = outfolder / file_func.filepath.relative_to(self.folder)
            os.remove(str(inst_fp))  # remove symlink to avoid writing in original dir
            with inst_fp.open("w") as f:
                f.write(file_func(**{x: y for x, y in kwargs.items() if x in file_func.parameters}))

    @contextlib.contextmanager
    def instantiate(self, **kwargs: Any) -> Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as tempfolder:
            subtempfolder = Path(tempfolder) / self.folder.name
            self.instantiate_to_folder(subtempfolder, **kwargs)
            yield subtempfolder


class InstrumentedFunction(base.BaseFunction):
    """Converts a multi-argument function into a mono-argument multidimensional continuous function
    which can be optimized.

    Parameters
    ----------
    function: callable
        the callable to convert
    *args, **kwargs: Any
        Any argument. Arguments of type variables.SoftmaxCategorical or variables.Gaussian will be instrumented
        and others will be kept constant.

    Note
    ----
    - Tokens can be:
      - DiscreteToken(list_of_n_possible_values): converted into a n-dim array, corresponding to proba for each value
      - GaussianToken(mean, std, shape=None): a Gaussian variable (shape=None) or array.
    - This function can then be directly used in benchmarks *if it returns a float*.

    """

    def __init__(self, function: Callable, *args: Any, **kwargs: Any) -> None:
        assert callable(function)
        self.instrumentation = variables.Instrumentation(*args, **kwargs)
        super().__init__(dimension=self.instrumentation.dimension)
        # keep track of what is instrumented (but "how" is probably too long/complex)
        instrumented = [f"arg{k}" if name is None else name for k, name in enumerate(self.instrumentation.names)
                        if not isinstance(self.instrumentation.instruments[k], variables._Constant)]
        name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
        self._descriptors.update(name=name, instrumented=",".join(instrumented))
        self._function = function
        self.last_call_args: Optional[Tuple[Any, ...]] = None
        self.last_call_kwargs: Optional[Dict[str, Any]] = None

    def convert_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Get the arguments and keyword arguments corresponding to the data

        Parameters
        ----------
        data: np.ndarray
            input data
        deterministic: bool
            whether to process the data deterministically (some Variables such as SoftmaxCategorical are stochastic).
            If True, the output is the most likely output.
        """
        return self.instrumentation.data_to_arguments(data, deterministic=deterministic)

    def convert_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self.instrumentation.arguments_to_data(*args, **kwargs)

    def oracle_call(self, x: np.ndarray) -> Any:
        self.last_call_args, self.last_call_kwargs = self.convert_to_arguments(x, deterministic=False)
        return self._function(*self.last_call_args, **self.last_call_kwargs)

    def __call__(self, x: np.ndarray) -> Any:
        # BaseFunction __call__ method should generally not be overriden,
        # but here that would mess up with typing, and I would rather not constrain
        # user to return only floats.
        x = self.transform(x)
        return self.oracle_call(x)

    def get_summary(self, data: np.ndarray) -> Any:  # probably impractical for large arrays
        """Prints the summary corresponding to the provided data
        """
        strings = []
        names = self.instrumentation.names
        instruments = self.instrumentation.instruments
        splitted_data = utils.split_data(data, instruments)
        for k, (name, var, d) in enumerate(zip(names, instruments, splitted_data)):
            if not isinstance(var, variables._Constant):
                explanation = var.get_summary(d)
                sname = f"arg #{k + 1}" if name is None else f'kwarg "{name}"'
                strings.append(f"{sname}: {explanation}")
        return " - " + "\n - ".join(strings)
