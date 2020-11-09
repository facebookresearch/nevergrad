import numpy as np
import nevergrad as ng
from ..base import ExperimentFunction

class UnitCommitmentProblem1(ExperimentFunction):
    """Model that uses conventional implementation for emi-continuous variables
    The model is adopted from Pyomo model 1: Conventional implementation for emi-continuous variables
    (https://jckantor.github.io/ND-Pyomo-Cookbook/04.06-Unit-Commitment.html)
    The constraints are added to the objective with a heavy penalty for violation.

    Parameters
    ----------
    T_points: int
        number of time points
    N_generators: int
        number of generators
    penalty_weight: float
        weight to penalize for violation of constraints
    """
    def __init__(self, T_points: int = 13,
                 N_generators: int = 3,
                 penalty_weight : float = 10000
                 ) -> None:
        params = {x: y for x, y in locals().items() if x not in ["self", "__class__"]}  # for copying
        # Demand for certain time period
        self.T_points = T_points
        self.demands = np.array([100 + 100*np.random.uniform() for t in range(self.T_points)])

        # Generating units
        self.N_generators = N_generators
        self.p_max = 2*max(self.demands) / self.N_generators
        self.p_min = 0.6*self.p_max

        # Cost
        self.cost_a = np.array([0.5 + 0.2*np.random.randn() for n in range(self.N_generators)])
        self.cost_b = np.array([10*np.random.uniform() for n in range(self.N_generators)])

        self.penalty_weight = penalty_weight

        param_x = ng.p.Array(shape=(self.N_generators, self.T_points)).set_bounds(0, self.p_max)
        param_u = ng.p.Array(shape=(self.N_generators, self.T_points)).set_bounds(0, 1).set_integer_casting()
        instru = ng.p.Instrumentation(x=param_x, u=param_u).set_name("")
        super().__init__(self.unit_commitment_obj_with_penalization, instru)
        self.register_initialization(**params)
        self._descriptors.update(T_points=T_points, N_generators=N_generators)


    def unit_commitment_obj_with_penalization(self, x, u):
        demand_penalty, lb_penalty, ub_penalty = 0, 0, 0
        # From demand constraint
        demand_penalty = np.sum(np.abs(np.sum(x, axis=0) - self.demands))
        # From semi_continuous_constraints
        lb_penalty = np.sum(np.clip(self.p_min*u - x, 0, a_max=None), axis=None)
        ub_penalty = np.sum(np.clip(x - self.p_max*u, 0, a_max=None), axis=None)
        # Running cost
        running_cost = np.sum(np.sum(x, axis=1) * self.cost_a + np.sum(u, axis=1) * self.cost_b)
        return running_cost + (demand_penalty + lb_penalty + ub_penalty) * self.penalty_weight

