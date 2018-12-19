# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This is a very basic example of an instrumented python script
"""
discrete_value = 10
# @nevergrad@ discrete_value = NG_SC{1|10|100}
continuous_value = 90
# @nevergrad@ continuous_value = NG_G{90, 20}
print(abs(continuous_value - 100) * discrete_value)  # last print should provide the fitness value (minimization)
