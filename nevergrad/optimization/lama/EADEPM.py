import numpy as np


class EADEPM:
    def __init__(
        self, budget, population_size=50, F_init=0.5, F_end=0.8, CR=0.9, memory_size=10, memory_factor=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_init = F_init  # Initial differential weight
        self.F_end = F_end  # Final differential weight for linear adaptation
        self.CR = CR  # Crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space bounds
        self.memory_size = memory_size  # Number of memory slots
        self.memory_factor = memory_factor  # Proportion of memory influence
        self.memory = []

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[np.argmin(fitness)]

                # Use memory in mutation if available
                memory_effect = (
                    np.sum(self.memory, axis=0) * self.memory_factor
                    if self.memory
                    else np.zeros(self.dimension)
                )

                # Mutation strategy incorporating memory
                mutant = x1 + F_current * (best - x1 + x2 - x3 + memory_effect)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Successful trials update the population and memory
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                    # Update memory with successful mutation vectors
                    if len(self.memory) < self.memory_size:
                        self.memory.append(mutant - population[i])
                    else:
                        # Replace a random memory component
                        self.memory[np.random.randint(len(self.memory))] = mutant - population[i]

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
