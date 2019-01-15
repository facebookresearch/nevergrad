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
    comment_chars = {x: "//" for x in [".cpp", ".hpp", ".c", ".h"]}
    comment_chars.update({".py": r"#", ".m": r"%"})
    pattern = r'^(?P<indent> *)'
    pattern += r'(?P<linetoken>' + comment_chars[extension] + r" *" + LINETOKEN + r" *)"
    pattern += r'(?P<command>.*)'
    lineseg = re.search(pattern, line)
    if lineseg is not None:
        line = lineseg.group("indent") + lineseg.group("command")
    if LINETOKEN in line:
        raise RuntimeError(f"Uncommenting failed for line of {extension} file (a {LINETOKEN} tag remains):\n{line}\n"
                           f"Did you follow the pattern indent+comment+{LINETOKEN}+code (with nothing before the indent)?")
    return line


class InstrumentedFile(utils.Instrument):

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        assert filepath.exists(), "{filepath} does not exist"
        with filepath.open("r") as f:
            text = f.read()
        if "NG_" in text and LINETOKEN in text:  # assuming there is a token somewhere
            lines = text.splitlines()
            ext = filepath.suffix
            lines = [(l if LINETOKEN not in l else uncomment_line(l, ext)) for l in lines]
            text = "\n".join(lines)
        self.text, self.variables = utils.replace_tokens_by_placeholders(text)

    def process(self, data: np.ndarray, deterministic: bool = False) -> str:
        values = utils.process_instruments(self.variables, data, deterministic=deterministic)
        text = utils.replace_placeholders_by_values(self.text, values)
        return text

    def process_arg(self, arg: Any) -> ArrayLike:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return sum(t.dimension for t in self.variables)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.filepath})"

    def get_summary(self, data: np.ndarray) -> str:
        splitted_data = utils.split_data(data, self.variables)
        strings = [f"In file {self.filepath}"]
        lines = self.text.splitlines()
        for line_ind, line in enumerate(lines):
            matches = list(re.finditer(utils._HOLDER_PATTERN, line))
            if matches:
                strings.extend([f"- on line #{line_ind + 1}", line.strip()])
                for match in matches:
                    ind = int(match.group("index"))
                    strings.append("Placeholder {}: {}".format(ind, self.variables[ind].get_summary(splitted_data[ind])))
        return "\n".join(strings)


class InstrumentedFolder:  # should derive from base function?
    """Folder with instrumentation tokens, which can be instantiated.

    Parameters
    ----------
    folder: str/Path
        the instrumented folder to instantiate
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case).
    extensions: list
        extensions of the instrumented files which must be instantiated

    Caution
    -------
        The clean copy is generally located in /tmp and may not be accessible for
        computation in a cluster. You may want to create a clean copy yourself
        in the folder of your choice, or set the the TemporaryDirectoryCopy class
        (located in instrumentation.instantiate) CLEAN_COPY_DIRECTORY environment
        variable to a shared directory
    """

    def __init__(self, folder: Union[Path, str], clean_copy: bool = False, extensions: Optional[List[str]] = None) -> None:
        self._clean_copy = None
        self.folder = Path(folder).expanduser().absolute()
        assert self.folder.exists(), "{folder} does not seem to exist"
        if clean_copy:
            self._clean_copy = utils.TemporaryDirectoryCopy(str(folder))
            self.folder = self._clean_copy.copyname
        if extensions is None:
            extensions = [".py", "m", ".cpp", ".hpp", ".c", ".h"]
        self.instrumented_files: List[InstrumentedFile] = []
        for fp in self.folder.glob("**/*"):  # TODO filter out all hidden files
            if fp.is_file() and fp.suffix in extensions:
                instru_f = InstrumentedFile(fp)
                if instru_f.dimension:
                    self.instrumented_files.append(instru_f)
        assert self.instrumented_files, "Found no instrumented file"
        self.instrumented_files = sorted(self.instrumented_files, key=operator.attrgetter("filepath"))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.folder}") with files:\n{self.instrumented_files}'

    @property
    def dimension(self) -> int:
        return sum(i.dimension for i in self.instrumented_files)

    def instantiate_to_folder(self, data: np.ndarray, outfolder: Union[Path, str]) -> None:
        outfolder = Path(outfolder).expanduser().absolute()
        assert outfolder != self.folder, "Do not instantiate on same folder!"
        symlink_folder_tree(self.folder, outfolder)
        texts = utils.process_instruments(self.instrumented_files, data)  # instantiable files have same pattern as token
        for instantiable, text in zip(self.instrumented_files, texts):
            inst_fp = outfolder / instantiable.filepath.relative_to(self.folder)
            os.remove(str(inst_fp))  # remove symlink to avoid writing in original dir
            with inst_fp.open("w") as f:
                f.write(text)

    @contextlib.contextmanager
    def instantiate(self, data: np.ndarray) -> Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as tempfolder:
            subtempfolder = Path(tempfolder) / self.folder.name
            self.instantiate_to_folder(data, subtempfolder)
            yield subtempfolder

    def get_summary(self, data: np.ndarray) -> str:
        splitted_data = utils.split_data(data, self.instrumented_files)
        strings = []
        for ifile, sdata in zip(self.instrumented_files, splitted_data):
            strings.append(ifile.get_summary(sdata))
        return "\n\n".join(strings)


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
