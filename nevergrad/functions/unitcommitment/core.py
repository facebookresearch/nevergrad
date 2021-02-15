# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad as ng
from ..base import ExperimentFunction


class UnitCommitmentProblem(ExperimentFunction):
    """Model that uses conventional implementation for semi-continuous variables
    The model is adopted from Pyomo model 1: Conventional implementation for semi-continuous variables
    (https://jckantor.github.io/ND-Pyomo-Cookbook/04.06-Unit-Commitment.html)
    The constraints are added to the objective with a heavy penalty for violation.

    Parameters
    ----------
    num_timepoints: int
        number of time points
    num_generators: int
        number of generators
    penalty_weight: float
        weight to penalize for violation of constraints
    """

    def __init__(
        self,
        problem_name: str = "semi-continuous",
        num_timepoints: int = 13,
        num_generators: int = 3,
        penalty_weight: float = 10000,
    ) -> None:
        if problem_name not in ["semi-continuous"]:
            raise NotImplementedError

        # Demand for certain time period
        self.num_timepoints = num_timepoints
        self.demands = np.random.uniform(low=100, high=200, size=(self.num_timepoints,))

        # Generating units
        self.num_generators = num_generators
        self.p_max = 2 * max(self.demands) / self.num_generators
        self.p_min = 0.6 * self.p_max

        # Cost
        self.cost_a = np.clip(np.random.randn(self.num_generators) * 0.2 + 0.5, a_min=0, a_max=None)
        self.cost_b = np.random.uniform(low=0, high=10, size=(self.num_generators,))

        self.penalty_weight = penalty_weight

        param_operational_output = ng.p.Array(shape=(self.num_generators, self.num_timepoints)).set_bounds(
            0, self.p_max
        )
        # param_operational_states = ng.p.Array(shape=(self.num_generators, self.num_timepoints)).set_bounds(0, 1).set_integer_casting() #
        param_operational_states = ng.p.Choice([0, 1], repetitions=self.num_timepoints * self.num_generators)
        instru = ng.p.Instrumentation(
            operational_output=param_operational_output,
            operational_states=param_operational_states,
        ).set_name("")
        super().__init__(self.unit_commitment_obj_with_penalization, instru)

    def unit_commitment_obj_with_penalization(self, operational_output, operational_states):
        operational_states = np.array(operational_states).reshape(self.num_generators, self.num_timepoints)
        demand_penalty, lb_penalty, ub_penalty = 0, 0, 0
        # From demand constraint
        demand_penalty = np.sum(np.abs(np.sum(operational_output, axis=0) - self.demands))
        # From semi_continuous_constraints
        lb_penalty = np.sum(
            np.clip(self.p_min * operational_states - operational_output, 0, a_max=None),
            axis=None,
        )
        ub_penalty = np.sum(
            np.clip(operational_output - self.p_max * operational_states, 0, a_max=None),
            axis=None,
        )
        # Running cost
        running_cost = np.sum(
            np.sum(operational_output, axis=1) * self.cost_a
            + np.sum(operational_states, axis=1) * self.cost_b
        )
        ret = running_cost + (demand_penalty + lb_penalty + ub_penalty) * self.penalty_weight
        return ret
