# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import nevergrad as ng
from nevergrad.parametrization import FolderFunction


if __name__ == "__main__":
    folder = Path(__file__).parents[1] / "parametrization" / "examples"
    func = FolderFunction(folder, ["python", "examples/script.py"], clean_copy=True)
    instrumentation = ng.Instrumentation(value1=ng.var.Array(1).asscalar(),
                                         value2=12,
                                         string=ng.var.SoftmaxCategorical(["plop", "blublu", "plouf"]))
    opt = ng.optimizers.registry["OnePlusOne"](instrumentation, budget=4)
    opt.minimize(func)
    ng.families.ParametrizedOnePlusOne()
