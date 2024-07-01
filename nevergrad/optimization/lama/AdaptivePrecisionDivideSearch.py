import numpy as np


class AdaptivePrecisionDivideSearch:
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
        num_divisions = 10
        division_size = 10.0 / num_divisions
        refine_factor = 0.9  # Factor to reduce division size for further refinements
        exploration_steps = self.budget // (num_divisions**self.dim)  # Exploration steps per division

        # Generate a grid around the center point and explore each grid division
        grid_offsets = np.linspace(-5.0, 5.0, num_divisions)
        for offset_dims in np.ndindex(*(num_divisions,) * self.dim):
            local_center = center_point + np.array([grid_offsets[dim] for dim in offset_dims])
            local_center = np.clip(local_center, -5.0, 5.0)  # Ensure it is within bounds
            local_scale = division_size / 2

            # Local search within the grid division
            for _ in range(exploration_steps):
                candidate = local_center + np.random.uniform(-local_scale, local_scale, self.dim)
                candidate_f = func(candidate)
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # Refine the division size for further precision
            division_size *= refine_factor

        return self.f_opt, self.x_opt
