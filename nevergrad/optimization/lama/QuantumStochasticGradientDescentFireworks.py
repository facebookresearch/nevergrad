import numpy as np


class QuantumStochasticGradientDescentFireworks:
    def __init__(self, budget=1000, num_sparks=10, num_iterations=100, learning_rate=0.1, momentum=0.9):
        self.budget = budget
        self.num_sparks = num_sparks
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimensions = 5
        bounds = func.bounds
        fireworks = np.random.uniform(bounds.lb, bounds.ub, size=(self.num_sparks, dimensions))
        best_firework = fireworks[0]
        velocities = np.zeros_like(fireworks)

        for _ in range(self.num_iterations):
            for firework in fireworks:
                f = func(firework)
                if f < func(best_firework):
                    best_firework = firework

            for i, firework in enumerate(fireworks):
                gradient = np.zeros(dimensions)
                for _ in range(self.num_sparks):
                    spark = firework + np.random.normal(0, 1, size=dimensions)
                    spark = np.clip(spark, bounds.lb, bounds.ub)
                    gradient += (func(spark) - func(firework)) * (spark - firework)

                velocities[i] = self.momentum * velocities[i] + self.learning_rate * gradient
                fireworks[i] += velocities[i]
                fireworks[i] = np.clip(fireworks[i], bounds.lb, bounds.ub)

        self.f_opt = func(best_firework)
        self.x_opt = best_firework

        return self.f_opt, self.x_opt
