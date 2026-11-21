# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import typing as tp
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PACKAGE_DIR = ROOT / "nevergrad"

# data files read at runtime through `Path(__file__)`, they are not python modules
# so they only end up in an installed nevergrad if they are listed in `package_data`
RUNTIME_DATA_GLOBS = [
    "functions/images/*.png",  # nevergrad.functions.images.core.Image
    "functions/lsgo/cdatafiles/*.txt",  # nevergrad.functions.lsgo._funcs.read_data
    "functions/photonics/*.npy",  # nevergrad.functions.photonics.photonics.epscSi/solar
]


@pytest.mark.skipif(not (ROOT / "setup.py").exists(), reason="requires a source checkout")
def test_runtime_data_files_are_packaged(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "setup.py", "--quiet", "build_py", "--build-lib", str(tmp_path)],
        cwd=ROOT,
        check=True,
    )
    missing: tp.List[str] = []
    for pattern in RUNTIME_DATA_GLOBS:
        expected = sorted(p.relative_to(PACKAGE_DIR) for p in PACKAGE_DIR.glob(pattern))
        assert expected, f"no source file matches {pattern}"  # the list above is out of date
        missing.extend(str(p) for p in expected if not (tmp_path / "nevergrad" / p).exists())
    assert not missing, f"{len(missing)} data files missing from the built package, first ones: {missing[:3]}"
