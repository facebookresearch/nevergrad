import numpy as np


class ERADS_UltraDynamicMaxSupreme:
    def __init__(self, budget, population_size=55, F_init=0.5, F_end=0.95, CR=0.9, memory_factor=0.35):
        self.budget = budget
        self.population_size = population_size  # Adjusted for optimal exploration-exploitation
        self.F_init = F_init  # More conservative initial mutation factor
        self.F_end = F_end  # Higher final mutation factor for forceful late exploration
        self.CR = CR  # Adapted crossover probability for increased diversity
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space limits
        self.memory_factor = (
            memory_factor  # Adapted memory factor for enhanced exploitation of successful directions
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
            # Linearly adapt the mutation factor throughout optimization
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                # Prevent selection of the current index
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Generate mutant vector with dynamic memory usage
                mutant = x1 + F_current * (best - x1 + x2 - x3) + self.memory_factor * memory
                mutant = np.clip(
                    mutant, self.bounds[0], self.bounds[1]
                )  # Ensure mutant remains within bounds

                # Crossover to create the trial solution
                trial = np.where(np.random.rand(self.dimension) < self.CR, mutant, population[i])
                f_trial = func(trial)
                evaluations += 1

                # Evaluate and potentially replace the current individual
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_index = i

                    # Update memory based on successful mutation
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * (mutant - population[i])

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
