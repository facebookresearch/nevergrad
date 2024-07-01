import numpy as np


class MomentumGradientExploration:
    def __init__(self, budget):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.learning_rate = 0.1
        self.epsilon = 1e-8
        self.exploration_prob = 0.3  # Increased exploration probability
        self.momentum_factor = 0.9  # Momentum factor for gradient-based updates

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

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

        momentum = np.zeros(self.dim)

        for i in range(1, self.budget):
            if np.random.rand() < self.exploration_prob:
                # Perform random exploration
                x = random_vector()
                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x
            else:
                # Perform gradient-based exploitation with momentum
                grad = gradient_estimate(x)
                adapt_lr = self.learning_rate / (np.sqrt(i) + self.epsilon)
                momentum = self.momentum_factor * momentum + adapt_lr * grad
                perturbation = np.random.randn(self.dim) * adapt_lr  # Random perturbation

                new_x = x - momentum + perturbation
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
# optimizer = MomentumGradientExploration(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
