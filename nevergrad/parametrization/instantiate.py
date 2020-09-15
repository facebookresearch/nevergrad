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
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import utils


LINETOKEN = "@nevergrad" + "@"  # Do not trigger an error when parsing this file...
COMMENT_CHARS = {".c": "//", ".h": "//", ".cpp": "//", ".hpp": "//", ".py": "#", ".m": "%"}


def _convert_to_string(data: tp.Any, extension: str) -> str:
    """Converts the data into a string to be injected in a file
    """
    if isinstance(data, np.ndarray):
        string = repr(data.tolist())
    else:
        string = repr(data)
    if extension in [".h", ".hpp", ".cpp", ".c"] and isinstance(data, np.ndarray):  # TODO: custom extensions are handled as python
        string = string.replace("[", "{").replace("]", "}")
    return string


class Placeholder:
    """Placeholder tokens to for external code instrumentation
    """

    pattern = r'NG_ARG' + r'{(?P<name>\w+?)(\|(?P<comment>.+?))?}'

    def __init__(self, name: str, comment: tp.Optional[str]) -> None:
        self.name = name
        self.comment = comment

    @classmethod
    def finditer(cls, text: str) -> tp.List['Placeholder']:
        prog = re.compile(cls.pattern)
        return [cls(x.group("name"), x.group("comment")) for x in prog.finditer(text)]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name!a}, {self.comment!a})'

    def __eq__(self, other: tp.Any) -> bool:
        if self.__class__ == other.__class__:
            return (self.name, self.comment) == (other.name, other.comment)
        return False

    @classmethod
    def sub(cls, text: str, extension: str, replacers: tp.Dict[str, tp.Any]) -> str:
        found: tp.Set[str] = set()
        kwargs = {x: _convert_to_string(y, extension) for x, y in replacers.items()}

        def _replacer(regex: tp.Match[str]) -> str:
            name = regex.group("name")
            if name in found:
                raise RuntimeError(f'Trying to remplace a second time placeholder "{name}"')
            if name not in kwargs:
                raise KeyError(f'Could not find a value for placeholder "{name}"')
            found.add(name)
            return str(kwargs[name])

        text = re.sub(cls.pattern, _replacer, text)
        missing = set(kwargs) - found
        if missing:
            raise RuntimeError(f"All values have not been consumed: {missing}")
        return text


def symlink_folder_tree(folder: tp.Union[Path, str], shadow_folder: tp.Union[Path, str]) -> None:
    """Utility for copying the tree structure of a folder and symlinking all files
    This can help creating lightweight copies of a project, for instantiating several
    copies with different parameters.
    """
    folder, shadow_folder = (Path(x).expanduser().resolve().absolute() for x in (folder, shadow_folder))
    shadow_folder.mkdir(parents=True, exist_ok=True)
    for fp in folder.iterdir():  # iterating is more efficient than globbing here
        shadow_fp = shadow_folder / fp.name
        if fp.is_dir():
            symlink_folder_tree(fp, shadow_fp)
        elif not shadow_fp.exists():
            shadow_fp.symlink_to(fp)


def uncomment_line(line: str, extension: str) -> str:
    if extension not in COMMENT_CHARS:
        raise RuntimeError(f'Unknown file type: {extension}\nDid you register it using {FolderFunction.register_file_type.__name__}?')
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
        deprecated_placeholders = ["NG_G{", "NG_OD{", "NG_SC{"]
        if any(x in text for x in deprecated_placeholders):
            raise RuntimeError(f"Found one of deprecated placeholders {deprecated_placeholders}. The API has now evolved to "
                               "a single placeholder NG_ARG{name|comment}, and FolderFunction now takes as many kwargs "
                               "as placeholders and must be instrumented before optimization.\n"
                               "Please refer to the README, PR #73 or issue #45 for more information")
        if LINETOKEN in text:
            lines = text.splitlines()
            ext = filepath.suffix.lower()
            lines = [(l if LINETOKEN not in l else uncomment_line(l, ext)) for l in lines]
            text = "\n".join(lines)
        self.placeholders = Placeholder.finditer(text)
        self._text = text
        self.parameters: tp.Set[str] = set()
        for x in self.placeholders:
            if x.name not in self.parameters:
                self.parameters.add(x.name)
            else:
                raise RuntimeError(f'Found duplicate placeholder (names must be unique) with name "{x.name}" in file:\n{self.filepath}')

    def __call__(self, **kwargs: tp.Any) -> str:
        testing.assert_set_equal(kwargs, self.parameters, err_msg="Wrong input parameters.")
        return Placeholder.sub(self._text, self.filepath.suffix, replacers=kwargs)

    def __repr__(self) -> str:
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

    def __init__(self, folder: tp.Union[Path, str], clean_copy: bool = False) -> None:
        self._clean_copy = None
        self.folder = Path(folder).expanduser().absolute()
        assert self.folder.exists(), f"{folder} does not seem to exist"
        if clean_copy:
            self._clean_copy = utils.TemporaryDirectoryCopy(str(folder))
            self.folder = self._clean_copy.copyname
        self.file_functions: tp.List[FileTextFunction] = []
        names: tp.Set[str] = set()
        for fp in self.folder.glob("**/*"):  # TODO filter out all hidden files (+ build files?)
            if fp.is_file() and fp.suffix.lower() in COMMENT_CHARS:
                file_func = FileTextFunction(fp)
                fnames = {ph.name for ph in file_func.placeholders}
                if fnames:
                    if fnames & names:
                        raise RuntimeError(f"Found {fp} placeholders in another file (names must be unique): {fnames & names}")
                    self.file_functions.append(file_func)
        assert self.file_functions, "Found no file with placeholders"
        self.file_functions = sorted(self.file_functions, key=operator.attrgetter("filepath"))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}("{self.folder}") with files:\n{self.file_functions}'

    @property
    def placeholders(self) -> tp.List[Placeholder]:
        return [p for f in self.file_functions for p in f.placeholders]

    def instantiate_to_folder(self, outfolder: tp.Union[Path, str], kwargs: tp.Dict[str, tp.Any]) -> None:
        testing.assert_set_equal(kwargs, {x.name for x in self.placeholders}, err_msg="Wrong input parameters.")
        outfolder = Path(outfolder).expanduser().absolute()
        assert outfolder != self.folder, "Do not instantiate on same folder!"
        symlink_folder_tree(self.folder, outfolder)
        for file_func in self.file_functions:
            inst_fp = outfolder / file_func.filepath.relative_to(self.folder)
            os.remove(str(inst_fp))  # remove symlink to avoid writing in original dir
            with inst_fp.open("w") as f:
                f.write(file_func(**{x: y for x, y in kwargs.items() if x in file_func.parameters}))

    @contextlib.contextmanager
    def instantiate(self, **kwargs: tp.Any) -> tp.Generator[Path, None, None]:
        with tempfile.TemporaryDirectory() as tempfolder:
            subtempfolder = Path(tempfolder) / self.folder.name
            self.instantiate_to_folder(subtempfolder, kwargs)
            yield subtempfolder


class FolderFunction:
    """Turns a folder into a parametrized function
    (with nevergrad tokens)

    Parameters
    ----------
    folder: Path/str
        path to the folder to instrument
    command: list
        command to run from inside the folder. The last line in stdout will
        be the output of the function.
        The command must be performed from just outside the instrument
        directory
    verbose: bool
        whether to print the run command and from where it is run.
    clean_copy: bool
        whether to create an initial clean temporary copy of the folder in order to avoid
        versioning problems (instantiations are lightweight symlinks in any case)

    Returns
    -------
    Any
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
    def __init__(self, folder: tp.Union[Path, str], command: tp.List[str], verbose: bool = False, clean_copy: bool = False) -> None:
        self.command = command
        self.verbose = verbose
        self.postprocessings = [get_last_line_as_float]
        self.instantiator = FolderInstantiator(folder, clean_copy=clean_copy)
        self.last_full_output: tp.Optional[str] = None

    @staticmethod
    def register_file_type(suffix: str, comment_chars: str) -> None:
        """Register a new file type to be used for token instrumentation by providing the relevant file suffix as well as
        the characters that indicate a comment."""
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        COMMENT_CHARS[suffix] = comment_chars

    @property
    def placeholders(self) -> tp.List[Placeholder]:
        return self.instantiator.placeholders

    def __call__(self, **kwargs: tp.Any) -> tp.Any:
        with self.instantiator.instantiate(**kwargs) as folder:
            if self.verbose:
                print(f"Running {self.command} from {folder.parent} which holds {folder}")
            output: tp.Any = utils.CommandFunction(self.command, cwd=folder.parent)()
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


def get_last_line_as_float(output: str) -> float:
    split_output = output.strip().splitlines()
    return float(split_output[-1])
