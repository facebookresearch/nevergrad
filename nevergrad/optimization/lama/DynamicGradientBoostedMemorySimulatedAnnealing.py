import numpy as np


class DynamicGradientBoostedMemorySimulatedAnnealing:
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
        alpha_initial = 0.97  # Initial cooling rate
        beta_initial = 1.5  # Initial control parameter for acceptance probability

        x_current = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f_current = func(x_current)
        evaluations += 1

        # Memory for storing best solutions
        memory_size = 20  # Balanced memory size for diversity
        memory = np.zeros((memory_size, self.dim))
        memory_scores = np.full(memory_size, np.Inf)
        memory[0] = x_current
        memory_scores[0] = f_current

        T = T_initial
        beta = beta_initial

        # Define phases for dynamic adaptation
        phase1 = self.budget // 4  # Initial exploration phase
        phase2 = self.budget // 2  # Intensive search phase
        phase3 = 3 * self.budget // 4  # Exploitation phase

        while evaluations < self.budget and T > T_min:
            for _ in range(memory_size):
                if np.random.rand() < 0.5:
                    # Disturbance around current best memory solution
                    x_candidate = memory[np.argmin(memory_scores)] + T * np.random.randn(self.dim)
                else:
                    # Random memory selection
                    x_candidate = memory[np.random.randint(memory_size)] + T * np.random.randn(self.dim)

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

            T *= alpha_initial

            # Dynamic adjustment of beta and alpha for better exploration-exploitation balance
            if evaluations < phase1:
                beta = 2.0  # Higher exploration phase
                alpha = 0.99  # Slower cooling for thorough exploration
            elif evaluations < phase2:
                beta = 1.5  # Balanced phase
                alpha = 0.97  # Standard cooling rate
            elif evaluations < phase3:
                beta = 1.0  # Transition to exploitation
                alpha = 0.95  # Faster cooling for convergence
            else:
                beta = 2.5  # Higher acceptance for local search refinement
                alpha = 0.92  # Even faster cooling for final convergence

            # Gradient-based local search refinement more frequently
            if evaluations % (self.budget // 8) == 0 and evaluations < self.budget:
                x_best_memory = memory[np.argmin(memory_scores)]
                x_best_memory = self._local_refinement(func, x_best_memory)
                f_best_memory = func(x_best_memory)
                evaluations += 1
                if f_best_memory < self.f_opt:
                    self.f_opt = f_best_memory
                    self.x_opt = x_best_memory

            # Dimensional adjustment to escape local minima and diversify search
            if evaluations % (self.budget // 5) == 0 and evaluations < self.budget:
                x_best_memory = memory[np.argmin(memory_scores)]
                x_best_memory = self._dimensional_adjustment(func, x_best_memory)
                f_best_memory = func(x_best_memory)
                evaluations += 1
                if f_best_memory < self.f_opt:
                    self.f_opt = f_best_memory
                    self.x_opt = x_best_memory

            # Periodic exploration boost
            if evaluations % (self.budget // 4) == 0:
                for _ in range(memory_size // 2):  # Half the memory size for exploration boost
                    x_candidate = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    f_candidate = func(x_candidate)
                    evaluations += 1
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = x_candidate

                    # Update memory with better solutions
                    worst_idx = np.argmax(memory_scores)
                    if f_candidate < memory_scores[worst_idx]:
                        memory[worst_idx] = x_candidate
                        memory_scores[worst_idx] = f_candidate

        return self.f_opt, self.x_opt

    def _local_refinement(self, func, x, iters=150, step_size=0.007):
        for _ in range(iters):
            gradient = self._approximate_gradient(func, x)
            x -= step_size * gradient  # Gradient descent step
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

    def _dimensional_adjustment(self, func, x, step_factor=0.12):
        new_x = np.copy(x)
        for i in range(self.dim):
            new_x[i] += step_factor * (np.random.uniform(-1, 1) * (func.bounds.ub[i] - func.bounds.lb[i]))
        new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
        return new_x
