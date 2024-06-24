import numpy as np


class AdaptiveGradientSampling:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5

    def __call__(self, func):
        # Initialize parameters
        self.f_opt = np.inf
        self.x_opt = None
        lb, ub = -5.0, 5.0  # Bounds of the search space

        # Initial random point
        x_current = np.random.uniform(lb, ub, self.dim)
        f_current = func(x_current)

        # Update best found solution if it's better
        if f_current < self.f_opt:
            self.f_opt = f_current
            self.x_opt = x_current

        # Adaptive step size
        step_size = 0.5

        # Gradient approximation parameters
        epsilon = 1e-5
        for i in range(self.budget - 1):
            gradients = np.zeros(self.dim)
            for j in range(self.dim):
                x_temp = np.array(x_current)
                x_temp[j] += epsilon
                f_temp = func(x_temp)
                gradients[j] = (f_temp - f_current) / epsilon

            # Normalize the gradient vector to make it step-size independent
            norm = np.linalg.norm(gradients)
            if norm == 0:
                gradients = np.random.normal(0, 1, self.dim)  # random restart if gradient is zero
            else:
                gradients /= norm

            # Update current point with adaptive step
            x_new = x_current - step_size * gradients
            x_new = np.clip(x_new, lb, ub)  # Ensure new points are within bounds

            # Evaluate new point
            f_new = func(x_new)

            # Update current point if new point is better
            if f_new < f_current:
                x_current = x_new
                f_current = f_new
                step_size *= 1.1  # Increase step size slightly as we are in a good direction

                # Update the best found solution
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_current
            else:
                step_size *= 0.9  # Reduce step size as there was no improvement

        return self.f_opt, self.x_opt
