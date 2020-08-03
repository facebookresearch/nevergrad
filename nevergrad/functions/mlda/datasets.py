# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import contextlib
import typing as tp
from pathlib import Path
from unittest.mock import patch
import requests
import numpy as np
import pandas as pd
from scipy import misc


_NAMED_URLS = {"German towns": "http://people.sc.fsu.edu/~jburkardt/datasets/spaeth2/spaeth2_07.txt",
               "Ruspini": "https://rls.sites.oasis.unc.edu/s754/data/ruspini.txt",
               "Virus": "http://www.stats.ox.ac.uk/pub/PRNN/virus3.dat",
               "Employees": "https://astro.temple.edu/~alan/MMST/datasets2/samp05d.xls",
               "Landscape": "https://asterweb.jpl.nasa.gov/images/GDEM-10km-colorized.png"}


def get_cache_folder() -> Path:
    """Removes all cached datasets.
    This can be helpful in case of download issue
    """
    return Path(__file__).parent / "data"


def get_dataset_filepath(name: str) -> Path:
    if name not in _NAMED_URLS:
        raise ValueError(f'Dataset "{name}" is not available. Please choose among:\n{list(_NAMED_URLS.keys())}')
    url = _NAMED_URLS[name]
    url_name = Path(url).name
    path = get_cache_folder() / f"{name} - {url_name}"
    path.parent.mkdir(exist_ok=True)
    if path.exists():  # check not empty if exists
        with path.open("rb") as f:
            text = f.read()
        if not text.strip():
            path.unlink()
    if not path.exists():
        print(f'Downloading and caching external file "{name}" from url: {url}')
        try:
            response = requests.get(url, verify=True)
        except requests.exceptions.SSLError:
            warnings.warn(f"SSL verification failed for {url}, downloading without verification.")
            response = requests.get(url, verify=False)
        with path.open("wb") as f:
            f.write(response.content)
    return path


def get_data(name: str) -> tp.Union[np.ndarray, pd.DataFrame]:
    path = get_dataset_filepath(name)
    if name in ["German towns", "Ruspini", "Virus"]:
        return np.loadtxt(path)
    elif name == "Employees":  # proximity matrix
        return pd.read_excel(path)
    elif name == "Landscape":
        return misc.imread(path).dot([0.216, 0.7152, 0.0722])  # get brightness
    else:
        raise NameError(f'Unknown parsing for name "{name}"')


def _make_fake_get_data(name: str) -> tp.Union[np.ndarray, pd.DataFrame]:
    # Landscape is actually supposed to be exactly 10 times bigger (2160, 4320)
    sizes = {"Ruspini": (75, 2), "Virus": (38, 18), "Employees": (80, 81), "Landscape": (216, 432), "German towns": (89, 3)}
    data = np.zeros(sizes[name])
    return data if name != "Employees" else pd.DataFrame(data)


@contextlib.contextmanager
def mocked_data() -> tp.Iterator[tp.Any]:
    """Mocks all data that should be downloaded, in order to simplify testing
    """
    with patch("nevergrad.functions.mlda.datasets.get_data", new=_make_fake_get_data) as mocked:
        yield mocked


def make_perceptron_data(name: str) -> np.ndarray:
    """Creates the data (see https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view)
    """
    funcs = {"quadratic": lambda x: x**2, "sine": np.sin, "abs": np.abs, "heaviside": lambda x: x > 0}
    if name not in funcs:
        raise ValueError(f'Unknown name "{name}", available are:\n{list(funcs.keys())}')
    data: np.ndarray = np.zeros((50, 2))  # TODO: why?
    data[:, 0] = np.arange(-1, 1, .0408)
    data[:, 1] = funcs[name](data[:, 0])
    return data
