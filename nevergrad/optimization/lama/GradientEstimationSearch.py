import numpy as np


class GradientEstimationSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dimension = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialization
        x_current = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        f_current = func(x_current)
        self.f_opt = f_current
        self.x_opt = x_current

        # Small step for gradient approximation
        epsilon = 1e-5
        learning_rate = 0.1

        evaluations = 1

        while evaluations < self.budget:
            # Gradient estimation
            gradients = np.zeros(self.dimension)
            for i in range(self.dimension):
                x_step = np.array(x_current)
                x_step[i] += epsilon
                f_step = func(x_step)
                evaluations += 1
                gradients[i] = (f_step - f_current) / epsilon

                if evaluations >= self.budget:
                    break

            # Update the current point
            x_new = x_current - learning_rate * gradients
            # Maintain within bounds
            x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
            f_new = func(x_new)
            evaluations += 1

            # Check if a new optimum has been found
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new

            # Update current position
            x_current = x_new
            f_current = f_new

        return self.f_opt, self.x_opt
