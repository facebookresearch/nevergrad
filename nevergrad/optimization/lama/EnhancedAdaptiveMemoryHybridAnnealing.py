import numpy as np


class EnhancedAdaptiveMemoryHybridAnnealing:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        evaluations = 0

        T_initial = 1.0  # Initial temperature
        T_min = 1e-5  # Minimum temperature
        alpha = 0.97  # Cooling rate, slightly less aggressive cooling

        x_current = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f_current = func(x_current)
        evaluations += 1

        # Memory for storing best solutions
        memory_size = 20  # Increased memory size for diversity
        memory = np.zeros((memory_size, self.dim))
        memory_scores = np.full(memory_size, np.Inf)
        memory[0] = x_current
        memory_scores[0] = f_current

        # Adaptive memory factor
        memory_factor = 0.2  # Increased memory influence

        T = T_initial
        while evaluations < self.budget and T > T_min:
            for i in range(memory_size):
                if np.random.rand() < memory_factor:
                    x_candidate = memory[i] + T * np.random.randn(self.dim)
                else:
                    x_candidate = x_current + T * np.random.randn(self.dim)
                x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)

                # Hybrid component: local search around the candidate
                local_search_range = 0.1 * T
                x_local_candidate = x_candidate + local_search_range * np.random.randn(self.dim)
                x_local_candidate = np.clip(x_local_candidate, func.bounds.lb, func.bounds.ub)

                f_candidate = func(x_candidate)
                f_local_candidate = func(x_local_candidate)
                evaluations += 2

                # Use the better of the candidate and local candidate
                if f_local_candidate < f_candidate:
                    x_candidate = x_local_candidate
                    f_candidate = f_local_candidate

                if f_candidate < f_current or np.exp((f_current - f_candidate) / T) > np.random.rand():
                    x_current = x_candidate
                    f_current = f_candidate

                    # Update memory with better solutions
                    worst_idx = np.argmax(memory_scores)
                    if f_current < memory_scores[worst_idx]:
                        memory[worst_idx] = x_current
                        memory_scores[worst_idx] = f_current

                    if f_current < self.f_opt:
                        self.f_opt = f_current
                        self.x_opt = x_current

            T *= alpha
            # Adapt memory factor based on temperature
            memory_factor = max(0.1, memory_factor * (1 - T / T_initial))

        return self.f_opt, self.x_opt
