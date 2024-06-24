import numpy as np


class EnhancedQuantumFireworksAlgorithmV2:
    def __init__(
        self,
        budget=1000,
        num_fireworks=10,
        num_sparks=5,
        num_iterations=100,
        mutation_rate=0.1,
        explosion_rate=0.1,
    ):
        self.budget = budget
        self.num_fireworks = num_fireworks
        self.num_sparks = num_sparks
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.explosion_rate = explosion_rate

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimensions = 5
        bounds = func.bounds
        fireworks = np.random.uniform(bounds.lb, bounds.ub, size=(self.num_fireworks, dimensions))
        best_firework = fireworks[0]
        explosion_sizes = np.ones(self.num_fireworks)

        for _ in range(self.num_iterations):
            for firework in fireworks:
                f = func(firework)
                if f < func(best_firework):
                    best_firework = firework

                for _ in range(self.num_sparks):
                    selected_firework = np.random.choice(range(self.num_fireworks))
                    spark = (
                        fireworks[selected_firework]
                        + np.random.normal(0, 1, size=dimensions) * explosion_sizes[selected_firework]
                    )
                    spark = np.clip(spark, bounds.lb, bounds.ub)
                    f_spark = func(spark)

                    if f_spark < f:
                        fireworks[selected_firework] = spark
                        f = f_spark
                        if f < func(best_firework):
                            best_firework = spark

            # Introduce random mutation with adaptive explosion sizes
            for i in range(self.num_fireworks):
                if np.random.rand() < self.mutation_rate:
                    fireworks[i] = np.random.uniform(bounds.lb, bounds.ub, dimensions)

                fireworks[i] = np.clip(fireworks[i], bounds.lb, bounds.ub)
                explosion_sizes[i] = np.clip(
                    explosion_sizes[i] * (1 + self.explosion_rate * np.random.normal()), 1, None
                )

        self.f_opt = func(best_firework)
        self.x_opt = best_firework

        return self.f_opt, self.x_opt
