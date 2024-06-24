import numpy as np


class ERADS_AdaptivePlus:
    def __init__(
        self, budget, population_size=50, F_init=0.5, F_end=0.8, CR_init=0.9, CR_end=0.7, memory_factor=0.3
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial scaling factor for mutation
        self.F_end = F_end  # Final scaling factor for mutation, adjusting over time
        self.CR_init = CR_init  # Initial crossover probability
        self.CR_end = CR_end  # Final crossover probability, dynamically adjusting
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_factor = memory_factor  # Memory factor to guide mutation based on past successful steps

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        memory = np.zeros((self.population_size, self.dimension))  # Initialize memory for each individual

        while evaluations < self.budget:
            # Adaptive scaling factor and crossover probability
            progress = evaluations / self.budget
            F_current = self.F_init + (self.F_end - self.F_init) * progress
            CR_current = self.CR_init + (self.CR_end - self.CR_init) * progress

            for i in range(self.population_size):
                # Mutation using differential evolution strategy
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]

                # Differential mutation incorporating adaptive memory
                mutant = x1 + F_current * (x2 - x3 + self.memory_factor * memory[i])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])  # Enforcing bounds

                # Crossover operation
                trial = np.where(np.random.rand(self.dimension) < CR_current, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Selection and memory update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    memory[i] = (1 - self.memory_factor) * memory[i] + self.memory_factor * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        # Obtain the best result
        best_index = np.argmin(fitness)
        return fitness[best_index], population[best_index]
