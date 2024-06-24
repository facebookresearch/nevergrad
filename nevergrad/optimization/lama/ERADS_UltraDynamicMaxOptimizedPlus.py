import numpy as np


class ERADS_UltraDynamicMaxOptimizedPlus:
    def __init__(self, budget, population_size=60, F_init=0.58, F_end=0.88, CR=0.93, memory_factor=0.38):
        self.budget = budget
        self.population_size = (
            population_size  # Slightly increased for better exploration-exploitation balance
        )
        self.F_init = F_init  # More aggressive initial mutation factor
        self.F_end = F_end  # Higher final mutation factor for late-stage intensification
        self.CR = CR  # Adjusted crossover probability for more robust exploration
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Search space boundaries
        self.memory_factor = (
            memory_factor  # Enhanced memory factor for leveraging successful mutation patterns
        )

    def __call__(self, func):
        # Initialize population within defined bounds
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(
            self.dimension
        )  # Memory initialization for storing direction of successful mutations

        while evaluations < self.budget:
            # Adapt mutation factor over the optimization period
            F_current = self.F_init + (self.F_end - self.F_init) * (evaluations / self.budget)

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Mutant vector generation incorporating memory of successful mutations
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover to create the trial solution
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

                    # Update memory with successful mutation direction
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
