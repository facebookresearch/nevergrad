import numpy as np


class ERADS_UltraDynamicMaxHyper:
    def __init__(
        self,
        budget,
        population_size=50,
        F_base=0.55,
        F_amp=0.25,
        CR=0.98,
        memory_factor=0.3,
        adaptation_factor=0.02,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base scaling factor for mutation
        self.F_amp = F_amp  # Amplitude factor for dynamic adaptability of mutation
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory to guide mutation based on past successful steps
        self.adaptation_factor = (
            adaptation_factor  # Factor to enhance adaptation during mutation and crossover
        )

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Initialize memory to store successful mutation directions

        while evaluations < self.budget:
            # Dynamic adaptation of the scaling factor using a sinusoidal function for robust exploration
            t = evaluations / self.budget
            F_current = self.F_base + self.F_amp * np.sin(np.pi * t)  # Sinusoidal function for F

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i

                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

            # Adapt CR and memory factor based on periodicity of the search space exploration
            if (evaluations / self.budget) % 0.1 == 0:
                self.CR = min(1, self.CR + self.adaptation_factor * (0.5 - np.random.random()))
                self.memory_factor = max(0, self.memory_factor - self.adaptation_factor)

        return self.f_opt, self.x_opt
