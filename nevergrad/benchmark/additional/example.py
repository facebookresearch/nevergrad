# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nevergrad.functions import ArtificialFunction
from nevergrad import instrumentation as inst
from nevergrad.optimization import registry as optimregistry
from nevergrad.benchmark import registry as xpregistry
from nevergrad.benchmark import Experiment
# this file implements:
# - an additional test function: CustomFunction
# - an additional optimizer: NewOptimizer
# - an addition experiment plan: additional_experiment
# it can be used with the --imports parameters if nevergrad.benchmark commandline function


class CustomFunction(inst.InstrumentedFunction):
    """Example of a new test function
    """

    def __init__(self, offset):
        super().__init__(self.oracle_call, inst.var.Array(1).asscalar())
        self.offset = offset
        # add your own function descriptors (from base class, we already get "dimension" etc...)
        # those will be recorded during benchmarks
        self._descriptors.update(offset=offset)

    def oracle_call(self, x):  # np.ndarray as input
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        return (x - self.offset)**2


@optimregistry.register  # register optimizers in the optimization registry
class NewOptimizer(optimregistry["NoisyBandit"]):
    pass


@xpregistry.register  # register experiments in the experiment registry
def additional_experiment():  # The signature can also include a seed argument if need be (see experiments.py)
    funcs = [ArtificialFunction(name="sphere", block_dimension=10), CustomFunction(2)]
    for budget in [10, 100]:
        for optimizer in ["NewOptimizer", "RandomSearch"]:
            for func in funcs:  # 2 realizations of the same function
                yield Experiment(func, optimizer=optimizer, budget=budget, num_workers=1)
