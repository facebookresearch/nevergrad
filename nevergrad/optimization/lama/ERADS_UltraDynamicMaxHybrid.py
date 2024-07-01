import numpy as np


class ERADS_UltraDynamicMaxHybrid:
    def __init__(
        self, budget, population_size=100, F_init=0.5, F_end=0.8, CR_init=0.9, CR_end=0.6, memory_factor=0.4
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial scaling factor for mutation
        self.F_end = F_end  # Final scaling factor for mutation
        self.CR_init = CR_init  # Initial crossover probability
        self.CR_end = CR_end  # Final crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory factor to guide mutation based on past successful steps

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
            # Linear interpolation for F and CR over the course of evaluations
            t = evaluations / self.budget
            F_current = self.F_init + (self.F_end - self.F_init) * t
            CR_current = self.CR_init + (self.CR_end - self.CR_init) * t

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Mutant vector calculation includes best solution and memory factor
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover to create a trial vector
                trial = np.where(np.random.rand(self.dimension) < CR_current, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Selection based on fitness improvement
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i

                    # Update memory with successful mutation direction
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * (mutant - population[i])

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
