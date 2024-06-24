import numpy as np


class RefinedAdaptivePrecisionDivideSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize center point
        center_point = np.random.uniform(-5.0, 5.0, self.dim)
        center_f = func(center_point)
        if center_f < self.f_opt:
            self.f_opt = center_f
            self.x_opt = center_point

        # Division strategy parameters
        num_divisions = 5  # Reduce the number of initial divisions to focus more on refinement
        division_size = 10.0 / num_divisions
        refine_factor = 0.75  # Increase refinement factor for more aggressive focusing
        initial_exploration_steps = max(
            1, self.budget // (num_divisions**self.dim) // 2
        )  # Balance between initial exploration and refinement

        # Generate a grid around the center point and explore each grid division
        grid_offsets = np.linspace(-5.0, 5.0, num_divisions)
        for offset_dims in np.ndindex(*(num_divisions,) * self.dim):
            local_center = center_point + np.array([grid_offsets[dim] for dim in offset_dims])
            local_center = np.clip(local_center, -5.0, 5.0)  # Ensure it is within bounds
            local_scale = division_size / 2

            # Local search within the grid division
            for _ in range(initial_exploration_steps):
                candidate = local_center + np.random.uniform(-local_scale, local_scale, self.dim)
                candidate_f = func(candidate)
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # Refinement phase
            refinement_budget = (self.budget - initial_exploration_steps * num_divisions**self.dim) // (
                num_divisions**self.dim
            )
            for _ in range(refinement_budget):
                refined_scale = local_scale * refine_factor
                refined_candidate = self.x_opt + np.random.uniform(-refined_scale, refined_scale, self.dim)
                refined_candidate_f = func(refined_candidate)
                if refined_candidate_f < self.f_opt:
                    self.f_opt = refined_candidate_f
                    self.x_opt = refined_candidate

        return self.f_opt, self.x_opt
