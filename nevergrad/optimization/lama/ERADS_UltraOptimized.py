import numpy as np


class ERADS_UltraOptimized:
    def __init__(
        self, budget, population_size=100, F_init=0.6, F_end=0.3, CR_init=0.9, CR_end=0.6, memory_factor=0.3
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial scaling factor for mutation
        self.F_end = F_end  # Final scaling factor for mutation, adjusting over time
        self.CR_init = CR_init  # Initial crossover probability
        self.CR_end = CR_end  # Final crossover probability, adjusting over time
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
            progress = evaluations / self.budget
            F_current = self.F_init + (self.F_end - self.F_init) * progress
            CR_current = self.CR_init + (self.CR_end - self.CR_init) * progress

            for i in range(self.population_size):
                # Selection of three distinct random population indices different from current index i
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Create the mutant vector incorporating memory of past successful mutations
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(
                    mutant, self.bounds[0], self.bounds[1]
                )  # Ensure mutant remains within bounds

                # Crossover operation to generate trial vector
                trial = np.where(np.random.rand(self.dimension) < CR_current, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Selection: Replace the old vector if the trial vector has better fitness
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i  # Update the index of the best solution found

                    # Update memory with the successful mutation direction scaled by F_current
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
