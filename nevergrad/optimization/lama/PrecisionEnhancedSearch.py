import numpy as np


class PrecisionEnhancedSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize with random point with smaller intervals to avoid large initial dispersion
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Adaptive parameters with precision enhancements
        scale = 1.0  # Starting scale more conservative
        min_scale = 0.001  # Fine precision for advanced refinement
        adaptive_decay = 0.95  # Slower decay rate to sustain exploration
        exploration_probability = 0.7  # Higher initial exploration
        exploitation_boost = 0.1  # Boost exploitation gradually

        # Use a memory mechanism to remember past good positions
        memory_size = 5
        memory = [current_point.copy() for _ in range(memory_size)]
        memory_f = [current_f for _ in range(memory_size)]

        # Main optimization loop
        for i in range(1, self.budget):
            scale *= adaptive_decay
            scale = max(min_scale, scale)

            # Decide between exploration and exploitation
            if np.random.rand() < exploration_probability:
                # Global exploration
                candidate = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                # Local exploitation around a remembered good position
                memory_index = np.random.choice(range(memory_size))
                perturbation = np.random.normal(0, scale, self.dim)
                candidate = memory[memory_index] + perturbation
                candidate = np.clip(candidate, -5.0, 5.0)

            candidate_f = func(candidate)

            # Update memory if better
            max_memory_f = max(memory_f)
            if candidate_f < max_memory_f:
                worst_index = memory_f.index(max_memory_f)
                memory[worst_index] = candidate
                memory_f[worst_index] = candidate_f

            # Update global best
            if candidate_f < self.f_opt:
                self.f_opt = candidate_f
                self.x_opt = candidate

            # Adjust exploration probability and boost exploitation
            exploration_probability *= 1.0 - exploitation_boost
            exploitation_boost += 0.002  # Increase exploitation boost gradually

        return self.f_opt, self.x_opt
