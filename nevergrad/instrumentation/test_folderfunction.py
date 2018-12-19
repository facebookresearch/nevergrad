# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import numpy as np
from . import folderfunction


def test_folder_function() -> None:
    folder = Path(__file__).parents[1]
    func = folderfunction.FolderFunction(str(folder), ["python", "-m", "nevergrad.instrumentation.test_folderfunction"], clean_copy=True)
    output = func([12] * func.dimension)
    np.testing.assert_equal(output, 12)


if __name__ == "__main__":
    print("Hello World!")
    # @nevergrad@ print(NG_G{0,1})
