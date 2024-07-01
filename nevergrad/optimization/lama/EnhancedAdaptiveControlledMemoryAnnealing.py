import numpy as np


class EnhancedAdaptiveControlledMemoryAnnealing:
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
        alpha = 0.95  # Cooling rate, more aggressive cooling
        beta = 2.0  # Higher control parameter for better acceptance probability

        x_current = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f_current = func(x_current)
        evaluations += 1

        # Memory for storing best solutions
        memory_size = 20  # Larger memory size for more diversity
        memory = np.zeros((memory_size, self.dim))
        memory_scores = np.full(memory_size, np.Inf)
        memory[0] = x_current
        memory_scores[0] = f_current

        T = T_initial
        while evaluations < self.budget and T > T_min:
            for i in range(memory_size):
                if np.random.rand() < 0.5:
                    # Disturbance around current best memory solution
                    x_candidate = memory[np.argmin(memory_scores)] + T * np.random.randn(self.dim)
                else:
                    # Random memory selection
                    x_candidate = memory[i] + T * np.random.randn(self.dim)

                x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)
                f_candidate = func(x_candidate)
                evaluations += 1

                if f_candidate < f_current or np.exp(beta * (f_current - f_candidate) / T) > np.random.rand():
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
            # Dynamically adjust control parameter beta
            if evaluations < self.budget / 3:
                beta = 1.5  # Initial phase: higher exploration
            elif evaluations < 2 * self.budget / 3:
                beta = 1.0  # Middle phase: balanced exploration and exploitation
            else:
                beta = 2.5  # Final phase: higher acceptance for local search refinement

        return self.f_opt, self.x_opt
