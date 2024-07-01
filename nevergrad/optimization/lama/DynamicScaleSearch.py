import numpy as np


class DynamicScaleSearch:
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

        # Set initial scale of the Gaussian perturbations and memory
        init_scale = 0.5
        scale = init_scale
        memory = []
        adapt_rate = 0.1  # Adaptive rate for scale adjustment

        # Main optimization loop
        for i in range(self.budget - 1):
            # Dynamic adjustment of the scale based on progress
            if i % 100 == 0 and i > 0:
                scale = max(scale * 0.9, 0.01)  # Reduce scale to fine-tune search

            # Generate a new candidate by perturbing the current point
            candidate = current_point + np.random.normal(0, scale, self.dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_f = func(candidate)

            # If the candidate is better, move there
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f
                memory.append(candidate)  # Add to memory if successful

                # Update optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

                # Increase scale when making progress
                scale += adapt_rate
                # Limit memory size
                if len(memory) > 20:
                    memory.pop(0)
            else:
                # Occasionally jump to a remembered good solution
                if memory and np.random.rand() < 0.05:
                    current_point = memory[np.random.randint(len(memory))]
                # Decrease scale when no progress
                scale = max(scale * 0.95, 0.01)  # Avoid scale becoming too small

        return self.f_opt, self.x_opt
