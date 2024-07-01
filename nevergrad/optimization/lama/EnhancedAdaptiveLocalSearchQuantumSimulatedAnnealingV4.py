import numpy as np


class EnhancedAdaptiveLocalSearchQuantumSimulatedAnnealingV4:
    def __init__(
        self,
        budget=10000,
        initial_temp=1.0,
        cooling_rate=0.999,
        explore_ratio=0.1,
        perturb_range=0.1,
        local_search_iters=10,
    ):
        self.budget = budget
        self.dim = 5
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.explore_ratio = explore_ratio
        self.perturb_range = perturb_range
        self.local_search_iters = local_search_iters

    def _quantum_step(self, x):
        explore_range = self.explore_ratio * (5.0 - (-5.0))
        return x + np.random.uniform(-explore_range, explore_range, size=self.dim)

    def _local_search_step(self, x, func, search_range=0.1):
        candidate_x = x
        candidate_f = func(candidate_x)

        for _ in range(self.local_search_iters):
            perturb_range = search_range * np.exp(-_ / self.local_search_iters)  # Adaptive perturbation range
            new_candidate_x = candidate_x + np.random.uniform(-perturb_range, perturb_range, size=self.dim)
            new_candidate_x = np.clip(new_candidate_x, -5.0, 5.0)
            new_candidate_f = func(new_candidate_x)
            if new_candidate_f < candidate_f:
                candidate_x = new_candidate_x
                candidate_f = new_candidate_f

        return candidate_x, candidate_f

    def _acceptance_probability(self, candidate_f, current_f):
        return np.exp((current_f - candidate_f) / self.temp)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        current_x = np.random.uniform(-5.0, 5.0, size=self.dim)
        current_f = func(current_x)

        for i in range(self.budget):
            candidate_x = self._quantum_step(current_x)
            candidate_x = np.clip(candidate_x, -5.0, 5.0)
            candidate_x, candidate_f = self._local_search_step(candidate_x, func, search_range=0.1)

            if candidate_f < current_f or np.random.rand() < self._acceptance_probability(
                candidate_f, current_f
            ):
                current_x = candidate_x
                current_f = candidate_f

            if current_f < self.f_opt:
                self.f_opt = current_f
                self.x_opt = current_x

            self.temp *= self.cooling_rate

        return self.f_opt, self.x_opt
