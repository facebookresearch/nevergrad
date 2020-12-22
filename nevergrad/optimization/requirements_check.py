# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path
import nevergrad as ng
from nevergrad.parametrization import FolderFunction


if __name__ == "__main__":
    folder = Path(__file__).parents[1] / "parametrization" / "examples"
    func = FolderFunction(folder, [sys.executable, "examples/script.py"], clean_copy=True)
    instrumentation = ng.p.Instrumentation(
        value1=ng.p.Scalar(), value2=12, string=ng.p.Choice(["plop", "blublu", "plouf"])
    )
    opt = ng.optimizers.registry["OnePlusOne"](instrumentation, budget=4)
    opt.minimize(func)
    ng.families.ParametrizedOnePlusOne()
