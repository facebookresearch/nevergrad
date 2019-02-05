# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This is a very basic example of an instrumented python script
"""
value1 = 10
# @nevergrad@ value1 = NG_ARG{value1|this is a comment}
value2 = 90
# @nevergrad@ value2 = NG_ARG{value2}
string = "plop"
# @nevergrad@ string = NG_ARG{string}
print(12 if string == "blublu" else abs(value1 - 100) * value2)  # last print should provide the fitness value (minimization)
