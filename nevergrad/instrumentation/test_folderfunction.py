# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import numpy as np
from . import folderfunction


def test_folder_function() -> None:
    folder = Path(__file__).parent / "examples" / "basic"
    func = folderfunction.FolderFunction(str(folder), ["python", "basic/script.py"], clean_copy=True)
    output = func(value1=98, value2=6)
    np.testing.assert_equal(output, 12)
