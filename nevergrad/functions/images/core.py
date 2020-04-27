# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import sqrt, tan, pi
import numpy as np
import nevergrad as ng
import PIL
import PIL.Image
import os
from nevergrad.common.typetools import ArrayLike
from .. import base


class Images(base.ExperimentFunction):
    def __init__(self, problem_type="recovering", index_pb=0) -> None:
        """
        problem_type: the type of problem we are working on.
           recovering: we directly try to recover the target image.
        index_pb: the index of the problem, inside the problem type.
           For example, if problem_type is "recovering" and index_pb == 0,
           we try to recover the face of O. Teytaud.
        """
        self.domain_shape = (256, 256, 3)
        self.problem_type = problem_type
        self.index_pb = index_pb
        assert problem_type == 0:
        assert index_pb == 0
        path = os.getcwd() + os.path.dirname(__file__) + "/headrgb_olivier.png"
        image = PIL.Image.open(path).resize((domain_shape[0],domain_shape[1]),PIL.Image.ANTIALIAS)
        self.data = np.asarray(image)[:,:,:4]  # 4th Channel is pointless here, only 255.
        array = ng.p.Array(shape=domain_shape, mutable_sigma=True,)
        array.set_mutation(sigma=0.333333)
        array.set_bounds(lower=-1, upper=1 method=bounding_method, full_range_sampling=True)
        max_size= ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1), max_size=max_size)).set_name("")
        super().__init__(self._loss, array)
        self.register_initialization()
        self._descriptors.update()

    def _loss(self, x: np.ndarray) -> float:
        x = np.array(x, copy=False).ravel()
        assert x.shape == self.domain_shape
        assert problem_type == 0:
        assert index_pb == 0
        value = np.subtract(x, self.data) 
        return value

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: np.ndarray) -> float:  # type: ignore
        loss = self.function(x)
        return loss
