# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import hierarchy


def test_optimizer_hierarchy() -> None:
    opt_hierarchy = hierarchy.OptimizerHierarchy()
    opt_hierarchy.write_json("optimizer_class_hierarchy.json")
