import numpy as np


class ERADS_AdvancedDynamic:
    def __init__(
        self, budget, population_size=50, F_min=0.5, F_max=0.8, CR=0.9, memory_factor=0.25, adaptive=True
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_min = F_min  # Minimum scaling factor for mutation
        self.F_max = F_max  # Maximum scaling factor for mutation
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Factor to integrate memory into mutation
        self.adaptive = adaptive  # Flag to enable/disable dynamic adaptation of F

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Memory for successful mutation directions

        while evaluations < self.budget:
            F_current = self.F_min + (self.F_max - self.F_min) * np.sin(
                np.pi * evaluations / self.budget
            )  # Sinusoidal adaptation of F

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Mutant vector creation with memory influence
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover to create trial vector
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Selection step
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i
                    # Update memory with successful mutation direction
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
