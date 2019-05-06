# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from nevergrad import optimization
from nevergrad import instrumentation as inst


if __name__ == "__main__":
    folder = Path(__file__).parents[1] / "instrumentation" / "examples"
    func = inst.FolderFunction(folder, ["python", "examples/script.py"], clean_copy=True)
    instrumentation = inst.Instrumentation(value1=inst.var.Array(1).asscalar(),
                                           value2=12,
                                           string=inst.var.SoftmaxCategorical(["plop", "blublu", "plouf"]))
    opt = optimization.registry["OnePlusOne"](instrumentation, budget=4)
    opt.optimize(func)
