import numpy as np


class QuantumFireworksAlgorithm:
    def __init__(self, budget=1000, num_sparks=10, num_iterations=100, amplification_factor=1.5):
        self.budget = budget
        self.num_sparks = num_sparks
        self.num_iterations = num_iterations
        self.amplification_factor = amplification_factor

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimensions = 5
        bounds = func.bounds
        fireworks = np.random.uniform(bounds.lb, bounds.ub, size=(self.num_sparks, dimensions))
        best_firework = fireworks[0]

        for _ in range(self.num_iterations):
            for firework in fireworks:
                f = func(firework)
                if f < func(best_firework):
                    best_firework = firework

            for i, firework in enumerate(fireworks):
                for _ in range(self.num_sparks):
                    spark = firework + np.random.normal(0, 1, size=dimensions) * self.amplification_factor
                    spark = np.clip(spark, bounds.lb, bounds.ub)
                    f_spark = func(spark)
                    if f_spark < func(firework):
                        fireworks[i] = spark
                        if f_spark < func(best_firework):
                            best_firework = spark

        self.f_opt = func(best_firework)
        self.x_opt = best_firework

        return self.f_opt, self.x_opt
