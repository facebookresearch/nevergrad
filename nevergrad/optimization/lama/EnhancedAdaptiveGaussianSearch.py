import numpy as np


class EnhancedAdaptiveGaussianSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        # Initialize variables
        self.f_opt = np.inf
        self.x_opt = None
        # Start with a random point in the search space
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)

        # Update optimal solution if the initial guess is better
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Set initial scale of the Gaussian perturbations
        scale = 1.0
        # Introduce memory to remember past successful steps
        memory = []

        # Main optimization loop
        for i in range(self.budget - 1):
            # Generate a new candidate by perturbing the current point
            if np.random.rand() < 0.2 and memory:
                # With a 20% chance, jump to a historically good point
                candidate = memory[np.random.randint(len(memory))] + np.random.normal(
                    0, scale * 0.5, self.dim
                )
            else:
                candidate = current_point + np.random.normal(0, scale, self.dim)

            # Ensure the candidate stays within bounds
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_f = func(candidate)

            # If the candidate is better, move there and adjust the perturbation scale
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f
                scale *= 1.2  # Increase scale to explore further
                memory.append(candidate)  # Remember successful step

                # Update the optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate
                # Limit memory size to avoid excessive growth
                if len(memory) > 10:
                    memory.pop(0)

            # If not better, decrease the perturbation scale to refine search
            else:
                scale *= 0.85  # Encourage more localized search

        return self.f_opt, self.x_opt
