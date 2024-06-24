import numpy as np


class ERADS_Precision:
    def __init__(
        self,
        budget,
        population_size=40,
        F_init=0.55,
        F_end=0.9,
        CR_init=0.95,
        CR_end=0.7,
        memory_factor=0.3,
        elite_factor=0.1,
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
        self.elite_factor = elite_factor  # Percentage of population considered elite

    def __call__(self, func):
        # Initialize population uniformly within the bounds and calculate their fitness
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        elite_size = int(self.population_size * self.elite_factor)

        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Initialize memory

        while evaluations < self.budget:
            # Update scaling factor and crossover probability based on progress
            progress = evaluations / self.budget
            F_current = self.F_init + (self.F_end - self.F_init) * progress
            CR_current = self.CR_init + (self.CR_end - self.CR_init) * progress

            # Sort population based on fitness and define elite members
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 2, replace=False
                )
                x1, x2 = population[indices]
                elite_member = elite_population[np.random.randint(0, elite_size)]

                # Differential mutation with elite member influence and memory factor
                mutant = x1 + F_current * (
                    elite_member - x1 + x2 - population[i] + self.memory_factor * memory
                )
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])  # Ensure within bounds

                # Binomial crossover
                trial = np.where(np.random.rand(self.dimension) < CR_current, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Selection step
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )
                    if f_trial < fitness[np.argmin(fitness)]:
                        fitness[np.argmin(fitness)] = f_trial
                        population[np.argmin(fitness)] = trial

                if evaluations >= self.budget:
                    break

        # Return the best solution found
        best_index = np.argmin(fitness)
        return fitness[best_index], population[best_index]
