import numpy as np


class RefinedMultiStageAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Start with a random point in the search space
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)

        # Update if the initial guess is better
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Set initial scale and adaptive rates
        scale = 0.5
        global_scale = 0.1
        local_scale = 0.01

        # Adaptive scale factors
        exploration_scale_factor = 1.1
        exploitation_scale_factor = 0.9

        # Temperature for simulated annealing like probability acceptance
        temperature = 1.0
        min_temperature = 0.01
        temperature_decay = 0.95

        for i in range(1, self.budget):
            # Calculate current temperature
            temperature = max(min_temperature, temperature * temperature_decay)

            # Choose strategy: global or local search
            if np.random.rand() < 0.5:
                # Global search with larger mutations
                candidate = current_point + np.random.normal(0, global_scale, self.dim)
            else:
                # Local search with small mutations
                candidate = current_point + np.random.normal(0, local_scale, self.dim)

            candidate = np.clip(candidate, -5.0, 5.0)  # Ensure candidate remains within bounds
            candidate_f = func(candidate)

            # Apply acceptance criteria
            if candidate_f < current_f or np.exp((current_f - candidate_f) / temperature) > np.random.rand():
                current_point = candidate
                current_f = candidate_f

                # Update optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

                # Adjust scale based on the search type used
                if np.random.rand() < 0.5:
                    global_scale *= exploration_scale_factor
                else:
                    local_scale *= exploration_scale_factor
            else:
                global_scale *= exploitation_scale_factor
                local_scale *= exploitation_scale_factor

            # Clamp scale values to prevent them from becoming too large or too small
            global_scale = np.clip(global_scale, 0.05, 1.0)
            local_scale = np.clip(local_scale, 0.001, 0.1)

        return self.f_opt, self.x_opt
