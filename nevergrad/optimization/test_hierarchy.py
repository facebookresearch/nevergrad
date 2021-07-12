# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .hierarchy import build_hierarchy


def test_optimizer_hierarchy() -> None:
    build_hierarchy("optimizer_class_hierarchy.json")
