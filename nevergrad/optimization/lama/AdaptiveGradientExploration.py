import numpy as np


class AdaptiveGradientExploration:
    def __init__(self, budget):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.learning_rate = 0.1
        self.epsilon = 1e-8

        def random_vector():
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

        def gradient_estimate(x, h=1e-5):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] += h
                x2[i] -= h
                grad[i] = (func(x1) - func(x2)) / (2 * h)
            return grad

        x = random_vector()
        f = func(x)
        if f < self.f_opt:
            self.f_opt = f
            self.x_opt = x

        for i in range(1, self.budget):
            grad = gradient_estimate(x)
            adapt_lr = self.learning_rate / (np.sqrt(i) + self.epsilon)
            perturbation = random_vector() * adapt_lr
            new_x = x - adapt_lr * grad + perturbation

            new_x = np.clip(new_x, self.bounds[0], self.bounds[1])
            new_f = func(new_x)

            if new_f < self.f_opt:
                self.f_opt = new_f
                self.x_opt = new_x
                x = new_x
            else:
                x = random_vector()  # Restart exploration from random point

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = AdaptiveGradientExploration(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
