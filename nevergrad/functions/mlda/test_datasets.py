# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import typing as tp
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from ...common import testing
from . import datasets


def test_get_dataset_filepath() -> None:
    name = "__ Test Dataset __"
    assert name not in datasets._NAMED_URLS
    datasets._NAMED_URLS[name] = "file://{}".format(Path(__file__).absolute())
    with patch("requests.get") as req:
        req.return_value.content = b"blublu"
        filepath = datasets.get_dataset_filepath(name)
    assert filepath.exists()
    with filepath.open("r") as f:
        text = f.read()
    try:
        assert "blublu" in text, f"Found text:\n{text}"  # this is my file :)
    except AssertionError as e:
        raise e
    finally:
        filepath.unlink()
    np.testing.assert_raises(ValueError, datasets.get_dataset_filepath, "Bidule")
    del datasets._NAMED_URLS[name]
    assert not filepath.exists()
    assert name not in datasets._NAMED_URLS  # make sure it is clean


@testing.parametrized(
    german_towns=(
        "German towns",
        """#    Hey, this is a comment
#
320.9  13024  346.5
320.9  13024  346.5
""",
        [[320.9, 13024, 346.5], [320.9, 13024, 346.5]],
    ),
    ruspini=(
        "Ruspini",
        """     5    74
11    59""",
        [[5, 74], [11, 59]],
    ),
)
def test_get_data(name: str, text: str, expected: tp.List[tp.List[float]]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # create an example file
        filepath = Path(tmp) / "example.txt"
        with filepath.open("w") as f:
            f.write(text)
        # get the output
        with patch("nevergrad.functions.mlda.datasets.get_dataset_filepath") as path_getter:
            path_getter.return_value = filepath
            output = datasets.get_data(name)
    np.testing.assert_array_equal(output, expected)


@testing.parametrized(**{name: (name,) for name in datasets._NAMED_URLS})
def test_mocked_data(name: str) -> None:
    with datasets.mocked_data():  # this test makes sure we are able to mock all data, so that xp tests can run
        data = datasets.get_data(name)
        assert isinstance(data, (np.ndarray, pd.DataFrame))


def test_make_perceptron_data() -> None:
    for name, value in [("quadratic", 0.02028), ("sine", 0.14191), ("abs", 0.1424), ("heaviside", 1)]:
        data = datasets.make_perceptron_data(name)
        np.testing.assert_equal(data.shape, (50, 2))
        np.testing.assert_almost_equal(data[28, 0], 0.1424)
        np.testing.assert_almost_equal(data[28, 1], value, decimal=5, err_msg=f"Wrong value for {name}")


def test_xls_get_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # create an example file
        filepath = Path(tmp) / "example.xls"
        df = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        df.to_excel(filepath, index=False)
        # get the output
        with patch("nevergrad.functions.mlda.datasets.get_dataset_filepath") as path_getter:
            path_getter.return_value = filepath
            output = datasets.get_data("Employees")
    np.testing.assert_array_equal(output, df)
