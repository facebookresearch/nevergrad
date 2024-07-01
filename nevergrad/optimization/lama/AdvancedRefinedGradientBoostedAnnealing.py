import numpy as np


class AdvancedRefinedGradientBoostedAnnealing:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        evaluations = 0

        T_initial = 1.0  # Initial temperature
        T_min = 1e-6  # Minimum temperature
        alpha_initial = 0.96  # Cooling rate for initial phase

        x_current = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f_current = func(x_current)
        evaluations += 1

        # Memory for storing best solutions
        memory_size = 20
        memory = np.zeros((memory_size, self.dim))
        memory_scores = np.full(memory_size, np.Inf)
        memory[0] = x_current
        memory_scores[0] = f_current

        T = T_initial

        while evaluations < self.budget and T > T_min:
            for _ in range(memory_size):
                # Generate candidate solution
                if np.random.rand() < 0.5:
                    x_candidate = memory[np.argmin(memory_scores)] + T * np.random.randn(self.dim)
                else:
                    x_candidate = memory[np.random.randint(memory_size)] + T * np.random.randn(self.dim)

                x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)
                f_candidate = func(x_candidate)
                evaluations += 1

                # Perform Metropolis acceptance criterion
                if f_candidate < f_current or np.exp((f_current - f_candidate) / T) > np.random.rand():
                    x_current = x_candidate
                    f_current = f_candidate

                    worst_idx = np.argmax(memory_scores)
                    if f_current < memory_scores[worst_idx]:
                        memory[worst_idx] = x_current
                        memory_scores[worst_idx] = f_current

                    if f_current < self.f_opt:
                        self.f_opt = f_current
                        self.x_opt = x_current

            # Adaptive cooling schedule
            if evaluations < self.budget // 4:
                alpha = 0.98  # Phase 1
            elif evaluations < self.budget // 2:
                alpha = 0.97  # Phase 2
            elif evaluations < 3 * self.budget // 4:
                alpha = 0.96  # Phase 3
            else:
                alpha = 0.95  # Phase 4
            T *= alpha

            # Gradient-based local search refinement
            if evaluations % (self.budget // 10) == 0:
                x_best_memory = memory[np.argmin(memory_scores)]
                x_best_memory = self._local_refinement(func, x_best_memory)
                f_best_memory = func(x_best_memory)
                evaluations += 1
                if f_best_memory < self.f_opt:
                    self.f_opt = f_best_memory
                    self.x_opt = x_best_memory

            # Dimensional adjustment with adaptive step size
            if evaluations % (self.budget // 8) == 0:
                x_best_memory = memory[np.argmin(memory_scores)]
                x_best_memory = self._dimensional_adjustment(func, x_best_memory)
                f_best_memory = func(x_best_memory)
                evaluations += 1
                if f_best_memory < self.f_opt:
                    self.f_opt = f_best_memory
                    self.x_opt = x_best_memory

            # Improved periodic exploration boost
            if evaluations % (self.budget // 6) == 0:
                best_memory_idx = np.argmin(memory_scores)
                for _ in range(memory_size // 2):
                    if np.random.rand() < 0.25:
                        x_candidate = memory[best_memory_idx] + np.random.uniform(-1, 1, self.dim)
                    else:
                        x_candidate = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)
                    f_candidate = func(x_candidate)
                    evaluations += 1
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = x_candidate

                    worst_idx = np.argmax(memory_scores)
                    if f_candidate < memory_scores[worst_idx]:
                        memory[worst_idx] = x_candidate
                        memory_scores[worst_idx] = f_candidate

            # Enhanced memory-based exploration with diversity encouragement
            if evaluations % (self.budget // 5) == 0:
                for _ in range(memory_size // 3):
                    x_candidate = memory[np.random.randint(memory_size)] + np.random.uniform(
                        -0.5, 0.5, self.dim
                    )
                    x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)
                    f_candidate = func(x_candidate)
                    evaluations += 1
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = x_candidate

                    worst_idx = np.argmax(memory_scores)
                    if f_candidate < memory_scores[worst_idx]:
                        memory[worst_idx] = x_candidate
                        memory_scores[worst_idx] = f_candidate

        return self.f_opt, self.x_opt

    def _local_refinement(self, func, x, iters=50, step_size=0.005):
        for _ in range(iters):
            gradient = self._approximate_gradient(func, x)
            x -= step_size * gradient
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
        return x

    def _approximate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        fx = func(x)
        for i in range(self.dim):
            x_eps = np.copy(x)
            x_eps[i] += epsilon
            grad[i] = (func(x_eps) - fx) / epsilon
        return grad

    def _dimensional_adjustment(self, func, x, step_factor=0.1):
        new_x = np.copy(x)
        for i in range(self.dim):
            new_x[i] += step_factor * (np.random.uniform(-1, 1) * (func.bounds.ub[i] - func.bounds.lb[i]))
        new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
        return new_x
