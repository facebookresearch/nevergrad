import numpy as np


class AdaptiveLocalSearchOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.initial_step_size = (upper_bound - lower_bound) / 4
        self.min_step_size = (upper_bound - lower_bound) / 1000

    def local_search(self, func, current_point, step_size):
        """Perform a local search from the current point with the given step size."""
        best_point = current_point
        best_value = func(current_point)
        self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(self.dimension):
                for direction in [-1, 1]:
                    new_point = np.copy(current_point)
                    new_point[i] += direction * step_size
                    # Ensure new_point stays within bounds
                    new_point = np.clip(new_point, self.bounds[0], self.bounds[1])

                    new_value = func(new_point)
                    self.evaluations += 1
                    if new_value < best_value:
                        best_value = new_value
                        best_point = new_point

                    if self.evaluations >= self.budget:
                        return best_point, best_value

            if np.array_equal(best_point, current_point):
                break
            current_point = best_point

        return best_point, best_value

    def __call__(self, func):
        # Initialize at a random starting point
        current_point = np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
        step_size = self.initial_step_size
        self.evaluations = 0

        best_point, best_value = self.local_search(func, current_point, step_size)

        # Perform iterative reduction of step size and local search
        while step_size > self.min_step_size and self.evaluations < self.budget:
            step_size *= 0.5
            new_point, new_value = self.local_search(func, best_point, step_size)

            if new_value < best_value:
                best_value = new_value
                best_point = new_point

        return best_value, best_point
