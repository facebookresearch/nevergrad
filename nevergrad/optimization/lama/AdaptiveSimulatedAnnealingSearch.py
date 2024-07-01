import numpy as np


class AdaptiveSimulatedAnnealingSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Bounds of the search space
        temperature = 1.0  # Initial temperature for simulated annealing
        cooling_rate = 0.99  # Cooling rate for the annealing schedule
        step_size = 0.5  # Initial step size

        # Generate an initial point randomly
        x_current = np.random.uniform(lb, ub, self.dim)
        f_current = func(x_current)
        self.f_opt = f_current
        self.x_opt = x_current

        for i in range(1, self.budget):
            # Cooling down the temperature
            temperature *= cooling_rate

            # Generate a new point by perturbing the current point
            perturbation = np.random.normal(0, step_size, self.dim)
            x_new = x_current + perturbation
            x_new = np.clip(x_new, lb, ub)  # Ensure new points are within bounds

            # Evaluate the new point
            f_new = func(x_new)

            # Calculate the probability of accepting the new point
            if f_new < f_current:
                accept = True
            else:
                # Acceptance probability in case the new function value is worse
                # It depends on the difference between new and current function values and the temperature
                delta = f_new - f_current
                probability = np.exp(-delta / temperature)
                accept = np.random.rand() < probability

            # Accept the new point if it is better or by the criterion of simulated annealing
            if accept:
                x_current = x_new
                f_current = f_new
                # If a new optimum is found, update the best known values
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_current

            # Adaptively adjust the step size based on acceptance
            if accept:
                step_size *= 1.1  # Increase step size if moving in a good direction
            else:
                step_size *= 0.9  # Decrease step size if stuck or not making progress

        return self.f_opt, self.x_opt
