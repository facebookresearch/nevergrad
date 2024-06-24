import numpy as np


class QuantumDynamicallyAdaptiveFireworksAlgorithm:
    def __init__(
        self,
        budget=1000,
        num_sparks=10,
        num_iterations=100,
        amplification_factor=1.5,
        divergence_threshold=0.2,
    ):
        self.budget = budget
        self.num_sparks = num_sparks
        self.num_iterations = num_iterations
        self.amplification_factor = amplification_factor
        self.divergence_threshold = divergence_threshold

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimensions = 5
        bounds = func.bounds
        fireworks = np.random.uniform(bounds.lb, bounds.ub, size=(self.num_sparks, dimensions))
        best_firework = fireworks[0]
        num_successful_sparks = np.zeros(self.num_sparks)

        for _ in range(self.num_iterations):
            for firework in fireworks:
                f = func(firework)
                if f < func(best_firework):
                    best_firework = firework

            for i, firework in enumerate(fireworks):
                successful_sparks = 0
                for _ in range(self.num_sparks):
                    spark = firework + np.random.normal(0, 1, size=dimensions) * self.amplification_factor
                    spark = np.clip(spark, bounds.lb, bounds.ub)
                    f_spark = func(spark)
                    if f_spark < func(firework):
                        fireworks[i] = spark
                        successful_sparks += 1
                        if f_spark < func(best_firework):
                            best_firework = spark

                num_successful_sparks[i] = successful_sparks

            avg_sparks = np.mean(num_successful_sparks)
            if avg_sparks < self.divergence_threshold * self.num_sparks:
                for i, firework in enumerate(fireworks):
                    fireworks[i] = fireworks[i] + np.random.normal(0, 1, size=dimensions) * (
                        self.amplification_factor / 2
                    )

        self.f_opt = func(best_firework)
        self.x_opt = best_firework

        return self.f_opt, self.x_opt
