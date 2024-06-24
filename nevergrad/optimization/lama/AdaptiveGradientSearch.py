import numpy as np


class AdaptiveGradientSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize parameters
        self.f_opt = np.inf
        self.x_opt = None

        # Random initial point
        x = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        # Learning rate adaptation parameters
        alpha = 0.1  # Initial learning rate
        beta = 0.9  # Momentum term
        epsilon = 1e-8  # Small term to avoid division by zero

        velocity = np.zeros_like(x)

        for i in range(self.budget):
            # Evaluate the function at the current point
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()

            # Estimate gradient via finite differences
            grad = np.zeros_like(x)
            perturbation = 1e-5
            for j in range(self.dim):
                x_perturb = x.copy()
                x_perturb[j] += perturbation
                grad[j] = (func(x_perturb) - f) / perturbation

            # Update the velocity and position
            velocity = beta * velocity - alpha * grad
            x = x + velocity

            # Ensure x stays within bounds
            x = np.clip(x, self.lower_bound, self.upper_bound)

            # Adapt the learning rate based on the improvement
            if i > 0 and (prev_f - f) / abs(prev_f) > 0.01:
                alpha *= 1.05  # Increase learning rate if improvement is significant
            else:
                alpha *= 0.7  # Decrease learning rate if improvement is not significant

            prev_f = f

        return self.f_opt, self.x_opt
