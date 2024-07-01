import numpy as np


class HybridGradientMemoryAnnealing:
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
        alpha = 0.97  # Cooling rate, balanced cooling
        beta_initial = 1.5  # Initial control parameter for acceptance probability

        x_current = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f_current = func(x_current)
        evaluations += 1

        # Memory for storing best solutions
        memory_size = 10  # Moderate memory size for diversity
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

            # Gradient-based local search to refine the best memory solution
            if evaluations < self.budget and T < 0.5:
                x_best = memory[np.argmin(memory_scores)]
                for _ in range(10):  # Perform 10 gradient descent steps
                    gradient = self._approximate_gradient(func, x_best)
                    x_best -= 0.01 * gradient  # Gradient descent step
                    x_best = np.clip(x_best, func.bounds.lb, func.bounds.ub)
                    f_best = func(x_best)
                    evaluations += 1
                    if f_best < self.f_opt:
                        self.f_opt = f_best
                        self.x_opt = x_best

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

            # Memory Enrichment
            if evaluations % (memory_size * 5) == 0:
                self._enhance_memory(func, memory, memory_scores, evaluations)

        return self.f_opt, self.x_opt

    def _approximate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        fx = func(x)
        for i in range(self.dim):
            x_eps = np.copy(x)
            x_eps[i] += epsilon
            grad[i] = (func(x_eps) - fx) / epsilon
        return grad

    def _enhance_memory(self, func, memory, memory_scores, evaluations):
        # Enhancing memory by local optimization around best memory points
        for i in range(len(memory)):
            local_T = 0.1  # Low disturbance for local search
            x_local = memory[i]
            f_local = memory_scores[i]
            for _ in range(5):  # Local search iterations
                x_candidate = x_local + local_T * np.random.randn(self.dim)
                x_candidate = np.clip(x_candidate, func.bounds.lb, func.bounds.ub)
                f_candidate = func(x_candidate)
                evaluations += 1
                if evaluations >= self.budget:
                    break

                if f_candidate < f_local:
                    x_local = x_candidate
                    f_local = f_candidate

            memory[i] = x_local
            memory_scores[i] = f_local
