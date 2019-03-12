# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .functionlib import ArtificialFunction
from .base import BaseFunction
# this module is used by instrumentation for "BaseFunction"
# this init must therefore not import submodules which could
# require more dependencies that the "main" configuration.
