import numpy as np


class ERADS_UltraDynamicMaxRefinedPlus:
    def __init__(self, budget, population_size=55, F_init=0.55, F_end=0.88, CR=0.92, memory_factor=0.35):
        self.budget = budget
        self.population_size = (
            population_size  # Adjusted for better balance between exploration and exploitation
        )
        self.F_init = F_init  # Starting mutation factor
        self.F_end = (
            F_end  # Ending mutation factor, heightened to extend aggressive search later into the runtime
        )
        self.CR = CR  # Crossover probability, slightly reduced for finer-grained exploration
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # Defined bounds for the problem
        self.memory_factor = memory_factor  # Increased to enhance influence of successful directions

    def __call__(self, func):
        # Initialize population within the search space
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        evaluations = self.population_size
        memory = np.zeros(self.dimension)  # Memory for storing efficacious mutations

        while evaluations < self.budget:
            F_current = self.F_init + (self.F_end - self.F_init) * (
                evaluations / self.budget
            )  # Mutation factor adaptation

            for i in range(self.population_size):
                indices = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                x1, x2, x3 = population[indices]
                best = population[best_index]

                # Differential mutation considering memory
                mutant = x1 + F_current * (best - x1 + x2 - x3 + self.memory_factor * memory)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover to generate the trial solution
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

                    # Update memory with the successful mutation
                    memory = (1 - self.memory_factor) * memory + self.memory_factor * F_current * (
                        mutant - population[i]
                    )

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
