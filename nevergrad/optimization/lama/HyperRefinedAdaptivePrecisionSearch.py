import numpy as np


class HyperRefinedAdaptivePrecisionSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Utilize an initial center-based strategy to determine a good starting point
        center_point = np.random.uniform(-5.0, 5.0, self.dim)
        center_f = func(center_point)
        if center_f < self.f_opt:
            self.f_opt = center_f
            self.x_opt = center_point

        # Define adaptive grid dynamics
        num_divisions = 3  # Smaller to begin a finer initial partitioning
        division_size = 10.0 / num_divisions
        refine_factor = 0.5  # Stronger focus refinement
        adaptive_budget = self.budget

        # Start with a wider grid and progressively refine
        for iteration in range(1, 4):  # Deepening the focus through iterations
            grid_offsets = np.linspace(-division_size, division_size, num_divisions)
            best_local_center = self.x_opt
            best_local_f = self.f_opt

            # Search each division based on the current best
            for offset_dims in np.ndindex(*(num_divisions,) * self.dim):
                local_center = best_local_center + np.array([grid_offsets[dim] for dim in offset_dims])
                local_center = np.clip(local_center, -5.0, 5.0)  # Ensure it is within bounds
                local_budget = max(1, adaptive_budget // (num_divisions**self.dim))

                # Explore this division
                for _ in range(local_budget):
                    candidate = local_center + np.random.uniform(-division_size, division_size, self.dim)
                    candidate_f = func(candidate)
                    if candidate_f < best_local_f:
                        best_local_f = candidate_f
                        best_local_center = candidate

                adaptive_budget -= local_budget

            # Update the best found in this iteration
            if best_local_f < self.f_opt:
                self.f_opt = best_local_f
                self.x_opt = best_local_center

            # Refine grid and division size for next iteration
            division_size *= refine_factor  # Narrower focus
            num_divisions = 2  # Less divisions, more focus

        return self.f_opt, self.x_opt
