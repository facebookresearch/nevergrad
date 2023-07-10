# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import core


def test_images_adversarial() -> None:
    func = next(core.ImageAdversarial.make_folder_functions(None, model="test"))
    x = np.zeros(func.image.shape)
    value = func(x)
    assert isinstance(value, float)
    assert value < np.inf
    other_func = func.copy().copy()
    value2 = other_func(x)
    assert value2 == value  # same function


def test_image_adversarial_eval() -> None:
    func = next(core.ImageAdversarial.make_folder_functions(None, model="test"))
    output = func.evaluation_function(func.parametrization)
    assert output == 1
    func.targeted = True
    output = func.evaluation_function(func.parametrization)
    assert output == 0


def test_images() -> None:
    func = core.Image()
    x = 7 * np.fabs(np.random.normal(size=func.domain_shape))
    # data = func.parametrization.spawn_child().set_standardized_data(x.flatten()).value
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
    other_func = func.copy()
    value2 = other_func(x)
    assert value == value2
